import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from scipy.optimize import least_squares

from nuscenes_edge_demo import (
    CAMERAS, bilinear, draw_projected_points, load_frame, point_to_fixed_line,
    point_to_line_with_index, project, se3, se3_vector,
)


def trimmed_mean(values, fraction=0.2):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if not len(values):
        return 1e6
    trim = int(len(values) * fraction)
    return float(np.mean(values[trim:len(values) - trim or None]))


def rotation_align(source, target):
    source = source / (np.linalg.norm(source) + 1e-12)
    target = target / (np.linalg.norm(target) + 1e-12)
    axis = np.cross(source, target)
    sine = np.linalg.norm(axis)
    cosine = np.clip(np.dot(source, target), -1.0, 1.0)
    if sine < 1e-9:
        return np.eye(3)
    axis /= sine
    R, _ = cv2.Rodrigues(axis * np.arctan2(sine, cosine))
    return R


def main():
    ap = argparse.ArgumentParser(description="Joint six-camera body-frame LiDAR calibration")
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--out", default="/workspace/results/joint_body")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--noise-rpy-deg", type=float, nargs=3, default=[3.0, -3.0, 4.0])
    ap.add_argument("--noise-translation-m", type=float, nargs=3, default=[0.06, -0.04, 0.08])
    ap.add_argument("--coarse-span-deg", type=float, default=5.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    samples, token = [], scene["first_sample_token"]
    while token and len(samples) < args.frames:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]

    # This perturbation is common in the body frame. Each camera sees its
    # conjugated form, exactly as a shared LiDAR mounting drift would appear.
    body_manual_noise = se3(np.r_[np.deg2rad(args.noise_rpy_deg), args.noise_translation_m])
    frames = []
    for camera in CAMERAS:
        for sample in samples:
            frame = load_frame(nusc, sample, camera, args.dataroot)
            frame["camera"] = camera
            frame["T_manual"] = (np.linalg.inv(frame["T_ego_camera"]) @ body_manual_noise @ frame["T_ego_camera"])
            frames.append(frame)

    def transform(frame, body_delta):
        return (np.linalg.inv(frame["T_ego_camera"]) @ se3(body_delta) @
                frame["T_ego_camera"] @ frame["T_manual"])

    # Static LiDAR depth edges are sampled sparsely for a bounded global search.
    sampled_edges = [f["edge_pts"][::max(1, len(f["edge_pts"]) // 80)] for f in frames]

    def distance_score(x):
        per_view = []
        for frame, points in zip(frames, sampled_edges):
            uv, valid = project(points, transform(frame, x), frame["K"], frame["shape"])
            if valid.sum() >= 8:
                per_view.append(float(np.median(bilinear(frame["dist"], uv[valid]))))
        return trimmed_mean(per_view)

    def grid_search(center, span_deg, step_deg):
        span = np.broadcast_to(np.asarray(span_deg, dtype=np.float64), 3)
        step = np.broadcast_to(np.asarray(step_deg, dtype=np.float64), 3)
        values = [np.arange(-span[i], span[i] + 1e-6, step[i]) for i in range(3)]
        candidates = []
        for dr in values[0]:
            for dp in values[1]:
                for dy in values[2]:
                    x = center.copy()
                    x[:3] += np.deg2rad([dr, dp, dy])
                    candidates.append((distance_score(x), x))
        candidates.sort(key=lambda pair: pair[0])
        return candidates[0], candidates[:8]

    # Ground/IMU initialization makes roll and pitch observable without relying
    # on texture edges. It intentionally leaves yaw for the image/LiDAR search.
    ground_true = np.median(np.asarray([f["ground_normal_ego"] for f in frames]), axis=0)
    ground_manual = body_manual_noise[:3, :3] @ ground_true
    R_ground = rotation_align(ground_manual, ground_true)
    ground_x = np.zeros(6)
    ground_x[:3] = cv2.Rodrigues(R_ground)[0][:, 0]
    # Search yaw broadly, but keep gravity-constrained roll/pitch local.
    (coarse_score, coarse_x), top_hypotheses = grid_search(ground_x, [0.75, 0.75, args.coarse_span_deg], [0.25, 0.25, 0.5])
    (fine_score, fine_x), _ = grid_search(coarse_x, [0.25, 0.25, 0.75], [0.10, 0.10, 0.10])

    def establish_matches(x):
        total = 0
        for frame in frames:
            # load_frame already keeps these two arrays on the same stride.
            source = frame["pts"]
            tangent = frame["tangents"]
            uv, valid = project(source, transform(frame, x), frame["K"], frame["shape"])
            distances, indices = point_to_line_with_index(uv[valid], frame["segments"], tangent[valid], max_angle_deg=40.0)
            keep = np.zeros(len(source), dtype=bool)
            keep[valid] = distances < 14.0
            frame["match_pts"] = source[keep]
            frame["line_indices"] = indices[distances < 14.0]
            total += len(frame["match_pts"])
        return total

    def edge_residual(x):
        values = []
        for frame in frames:
            pts = frame.get("match_pts", np.empty((0, 3)))
            indices = frame.get("line_indices", np.empty(0, dtype=np.int32))
            if not len(pts):
                continue
            uv, valid = project(pts, transform(frame, x), frame["K"], frame["shape"])
            residual = np.full(len(pts), 50.0)
            residual[valid] = point_to_fixed_line(uv[valid], frame["segments"], indices[valid])
            values.append(residual)
        return np.concatenate(values) if values else np.full(1, 50.0)

    x = fine_x
    match_counts = []
    for _ in range(2):
        match_counts.append(establish_matches(x))
        def residual(candidate):
            prior = 0.12 * (candidate - x) / np.array([0.04, 0.04, 0.04, 0.10, 0.10, 0.10])
            return np.r_[edge_residual(candidate), prior]
        result = least_squares(
            residual, x, loss="huber", f_scale=3.0,
            bounds=(-np.array([0.16, 0.16, 0.16, 0.30, 0.30, 0.30]),
                    np.array([0.16, 0.16, 0.16, 0.30, 0.30, 0.30])), max_nfev=120,
        )
        x = result.x

    expected = se3_vector(np.linalg.inv(body_manual_noise))
    report = {
        "scene": scene["name"], "frames_per_camera": len(samples), "noise_frame": "body",
        "manual_body_noise": se3_vector(body_manual_noise).tolist(),
        "ground_initialized_body_correction": ground_x.tolist(),
        "expected_body_correction": expected.tolist(), "coarse_body_correction": coarse_x.tolist(),
        "coarse_distance_px": coarse_score, "fine_body_correction": fine_x.tolist(),
        "fine_distance_px": fine_score, "estimated_body_correction": x.tolist(),
        "matching_points_per_outer_iteration": match_counts,
        "initial_fixed_line_px": float(np.mean(edge_residual(fine_x))),
        "final_fixed_line_px": float(np.mean(edge_residual(x))), "cameras": {},
    }
    for camera in CAMERAS:
        camera_frames = [f for f in frames if f["camera"] == camera]
        residuals = []
        for frame in camera_frames:
            pts = frame["match_pts"]
            if len(pts):
                uv, valid = project(pts, transform(frame, x), frame["K"], frame["shape"])
                r = np.full(len(pts), 50.0)
                r[valid] = point_to_fixed_line(uv[valid], frame["segments"], frame["line_indices"][valid])
                residuals.extend(r.tolist())
        first = camera_frames[0]
        manual_image = draw_projected_points(first["image"], first["all_pts"], first["K"], transform(first, np.zeros(6)), radius=1)
        coarse_image = draw_projected_points(first["image"], first["all_pts"], first["K"], transform(first, fine_x), radius=1)
        final_image = draw_projected_points(first["image"], first["all_pts"], first["K"], transform(first, x), radius=1)
        cv2.imwrite(str(out / f"{camera}_manual.jpg"), manual_image)
        cv2.imwrite(str(out / f"{camera}_coarse.jpg"), coarse_image)
        cv2.imwrite(str(out / f"{camera}_final.jpg"), final_image)
        report["cameras"][camera] = {"matches": int(sum(len(f["match_pts"]) for f in camera_frames)), "final_point_to_line_px": float(np.mean(residuals)) if residuals else 50.0}

    # Boundary minima or a near tie make the global estimate non-publishable.
    runner_up_gap = float(top_hypotheses[1][0] - coarse_score)
    coarse_limits = np.array([0.75, 0.75, args.coarse_span_deg])
    boundary = bool(np.any(np.abs(np.rad2deg(coarse_x[:3] - ground_x[:3])) >= coarse_limits - 1e-6))
    report["coarse_runner_up_gap_px"] = runner_up_gap
    report["coarse_winner_on_boundary"] = boundary
    report["publish"] = bool(not boundary and runner_up_gap > 0.15)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
