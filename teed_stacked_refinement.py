"""Refine an ICP+hand-eye body correction with TEED and stacked ground returns.

This is an evaluation pipeline: nuScenes ego poses motion-compensate short
LiDAR histories, while the initial body correction comes from raw-scan ICP plus
hand-eye.  The image term is a dynamic TEED distance field, not fixed matches.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.optimize import least_squares

from nuscenes_edge_demo import CAMERAS, bilinear, project, se3, se3_vector
from stacked_lidar_demo import dynamic_mask, estimate_ground_mask, sensor_to_global
from teed_edge_cache import TEEDCache
from teed_ground_inspection import collect_sweeps, intensity_bev_contour
from teed_model import TEED


def caption(image, text):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (min(1100, canvas.shape[1]), 42), (0, 0, 0), -1)
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def body_to_camera(frame, body_delta):
    return (np.linalg.inv(frame["T_ego_camera"]) @ se3(body_delta) @
            frame["T_ego_camera"] @ frame["T_manual"])


def rotation_align(source, target):
    source = source / (np.linalg.norm(source) + 1e-12)
    target = target / (np.linalg.norm(target) + 1e-12)
    axis = np.cross(source, target)
    sine, cosine = np.linalg.norm(axis), np.clip(np.dot(source, target), -1.0, 1.0)
    if sine < 1e-9:
        return np.eye(3)
    axis /= sine
    return cv2.Rodrigues(axis * np.arctan2(sine, cosine))[0]


def collect_samples(nusc, scene, offsets):
    requested, offset = set(offsets), 0
    selected, token = [], scene["first_sample_token"]
    while token and offset <= max(requested):
        sample = nusc.get("sample", token)
        if offset in requested:
            selected.append((offset, sample))
        token, offset = sample["next"], offset + 1
    missing = requested - {index for index, _ in selected}
    if missing:
        raise ValueError(f"sample offsets outside scene: {sorted(missing)}")
    return selected


def build_frame(nusc, sample, camera_name, dataroot, sweeps, cache, intensity_percentile, teed_percentile,
                bev_resolution, max_points):
    camera_sd = nusc.get("sample_data", sample["data"][camera_name])
    camera = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
    reference_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])
    image = cv2.imread(str(Path(dataroot) / camera_sd["filename"]))
    if image is None:
        raise FileNotFoundError(camera_sd["filename"])
    K = np.asarray(camera["camera_intrinsic"], dtype=np.float64)
    camera_points, ego_points, intensity = [], [], []
    for lidar_sd, annotation_sample in collect_sweeps(nusc, sample, sweeps):
        lidar = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        raw = np.fromfile(Path(dataroot) / lidar_sd["filename"], dtype=np.float32).reshape(-1, 5)
        global_points = sensor_to_global(raw[:, :3], lidar, pose)
        movable = dynamic_mask(nusc, annotation_sample, global_points)
        global_points = global_points[~movable]
        camera_points.append((global_points - np.asarray(reference_pose["translation"])) @ Quaternion(reference_pose["rotation"]).rotation_matrix)
        ego_points.append((global_points - np.asarray(reference_pose["translation"])) @ Quaternion(reference_pose["rotation"]).rotation_matrix)
        intensity.append(raw[~movable, 3])

    # camera_points are temporarily in ego coordinates; use sensor calibration once.
    points_ego = np.concatenate(camera_points)
    intensity = np.concatenate(intensity)
    R_camera = Quaternion(camera["rotation"]).rotation_matrix
    points_camera = (points_ego - np.asarray(camera["translation"])) @ R_camera
    uv, visible = project(points_camera, np.eye(4), K, image.shape)
    ground = estimate_ground_mask(points_ego) & visible
    cutoff = float(np.percentile(intensity[ground], intensity_percentile))
    high_intensity = ground & (intensity >= cutoff)
    # Refinement uses every camera, so retain markings around the full vehicle
    # rather than the front-camera inspection window.
    candidate = intensity_bev_contour(points_ego, high_intensity, bev_resolution,
                                      x_min=-70.0, x_max=70.0, y_min=-70.0, y_max=70.0)
    ids = np.flatnonzero(candidate)
    if len(ids) > max_points:
        ids = ids[::int(np.ceil(len(ids) / max_points))]
    probability = cache.probability(image, camera_sd["token"])
    threshold = float(np.percentile(probability, teed_percentile))
    edge = probability >= threshold
    distance = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    T_ego_camera = np.eye(4)
    T_ego_camera[:3, :3], T_ego_camera[:3, 3] = R_camera, np.asarray(camera["translation"])
    return {
        "camera": camera_name, "image": image, "shape": image.shape, "K": K,
        "points": points_camera[ids], "distance": distance, "edge": edge,
        "high_intensity_count": int(high_intensity.sum()), "candidate_count": int(candidate.sum()),
        "sampled_count": int(len(ids)),
        "intensity_cutoff": cutoff, "teed_threshold": threshold,
        "ground_normal_ego": np.linalg.svd(points_ego[ground] - np.median(points_ego[ground], axis=0), full_matrices=False)[2][-1],
        "T_ego_camera": T_ego_camera,
    }


def residuals(frames, body_delta):
    values = []
    for frame in frames:
        uv, valid = project(frame["points"], body_to_camera(frame, body_delta), frame["K"], frame["shape"])
        residual = np.full(len(frame["points"]), 30.0)
        if valid.any():
            residual[valid] = bilinear(frame["distance"], uv[valid])
        values.append(residual)
    return np.concatenate(values) if values else np.full(1, 30.0)


def score(frames, body_delta):
    values = residuals(frames, body_delta)
    return {
        "mean_px": float(np.mean(values)), "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)), "points": int(len(values)),
    }


def draw_overlay(frame, body_delta, text):
    canvas = frame["image"].copy()
    canvas[frame["edge"]] = (0, 160, 255)
    uv, valid = project(frame["points"], body_to_camera(frame, body_delta), frame["K"], frame["shape"])
    for point in uv[valid].astype(int):
        cv2.circle(canvas, tuple(point), 2, (0, 255, 0), -1, cv2.LINE_AA)
    return caption(canvas, text)


def main():
    parser = argparse.ArgumentParser(description="TEED stacked-ground refinement after ICP+hand-eye")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--coarse-json", required=True)
    parser.add_argument("--out", default="/workspace/results/teed_stacked_refinement")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--train-offsets", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--holdout-offsets", type=int, nargs="+", default=[10, 12])
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--intensity-percentile", type=float, default=90.0)
    parser.add_argument("--bev-resolution", type=float, default=0.10)
    parser.add_argument("--teed-percentile", type=float, default=95.0)
    parser.add_argument("--cache-dir", default="/workspace/results/teed_cache_scene_0")
    parser.add_argument("--max-points-per-view", type=int, default=400)
    parser.add_argument("--max-rotation-step-deg", type=float, default=1.5)
    parser.add_argument("--max-translation-step-m", type=float, default=0.02,
                        help="Ground markings weakly constrain translation; keep this local")
    parser.add_argument("--refine-translation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gravity-constrain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gravity-tolerance-deg", type=float, default=0.50)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if set(args.train_offsets) & set(args.holdout_offsets):
        parser.error("train and holdout offsets must not overlap")

    report_coarse = json.loads(Path(args.coarse_json).read_text())
    if "estimated_body_correction" not in report_coarse or "manual_body_noise" not in report_coarse:
        parser.error("coarse JSON must come from lidar_icp_handeye.py")
    coarse = np.asarray(report_coarse["estimated_body_correction"], dtype=np.float64)
    manual_noise = np.asarray(report_coarse["manual_body_noise"], dtype=np.float64)
    expected = se3_vector(np.linalg.inv(se3(manual_noise)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model = TEED().to(device)
    model.load_state_dict(torch.load("/workspace/models/teed_biped_epoch5.pth", map_location=device, weights_only=True), strict=True)
    model.eval()
    cache = TEEDCache(args.cache_dir, model, device)

    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    manual_transform = se3(manual_noise)
    train_frames, holdout_frames = [], []
    for collection, target in ((collect_samples(nusc, scene, args.train_offsets), train_frames),
                               (collect_samples(nusc, scene, args.holdout_offsets), holdout_frames)):
        for _, sample in collection:
            for camera in CAMERAS:
                frame = build_frame(nusc, sample, camera, args.dataroot, args.sweeps, cache,
                                    args.intensity_percentile, args.teed_percentile, args.bev_resolution,
                                    args.max_points_per_view)
                frame["T_manual"] = np.linalg.inv(frame["T_ego_camera"]) @ manual_transform @ frame["T_ego_camera"]
                target.append(frame)

    initial_train, initial_holdout = score(train_frames, coarse), score(holdout_frames, coarse)
    ground_true = np.median(np.asarray([frame["ground_normal_ego"] for frame in train_frames]), axis=0)
    if ground_true[2] < 0:
        ground_true = -ground_true
    ground_manual = manual_transform[:3, :3] @ ground_true
    ground_correction = np.zeros(6)
    ground_correction[:3] = cv2.Rodrigues(rotation_align(ground_manual, ground_true))[0][:, 0]
    rotation_step = np.deg2rad(args.max_rotation_step_deg)
    gravity_step = np.deg2rad(args.gravity_tolerance_deg)
    translation_step = args.max_translation_step_m if args.refine_translation else 1e-4
    step = np.array([rotation_step, rotation_step, rotation_step,
                     translation_step, translation_step, translation_step])
    center = coarse.copy()
    if args.gravity_constrain:
        center[:2] = ground_correction[:2]
        step[:2] = gravity_step
    lower, upper = center - step, center + step

    def objective(candidate):
        # The TEED distance field is evaluated at every candidate transform,
        # which dynamically rebuilds the image-edge association.
        prior = 0.08 * (candidate - center) / step
        return np.r_[residuals(train_frames, candidate), prior]

    # Gravity initialization can intentionally move roll/pitch outside the
    # ICP estimate's local box, so start from the valid constrained center.
    result = least_squares(objective, center, bounds=(lower, upper), loss="huber", f_scale=2.5, max_nfev=100)
    refined = result.x
    final_train, final_holdout = score(train_frames, refined), score(holdout_frames, refined)
    boundary_dimensions = np.isclose(refined, lower, atol=1e-5) | np.isclose(refined, upper, atol=1e-5)
    boundary_hit = bool(np.any(boundary_dimensions[:3]) or (args.refine_translation and np.any(boundary_dimensions[3:])))
    train_view, holdout_view = train_frames[0], holdout_frames[0]
    cv2.imwrite(str(out / "train_coarse.jpg"), draw_overlay(train_view, coarse, "Train: ICP+hand-eye coarse"))
    cv2.imwrite(str(out / "train_refined.jpg"), draw_overlay(train_view, refined, "Train: TEED stacked-ground refined"))
    cv2.imwrite(str(out / "holdout_coarse.jpg"), draw_overlay(holdout_view, coarse, "Holdout: ICP+hand-eye coarse"))
    cv2.imwrite(str(out / "holdout_refined.jpg"), draw_overlay(holdout_view, refined, "Holdout: TEED stacked-ground refined"))
    report = {
        "mode": "ICP+hand-eye coarse followed by dynamic TEED stacked-ground refinement",
        "coarse_json": str(args.coarse_json), "scene": scene["name"], "device": str(device),
        "train_offsets": args.train_offsets, "holdout_offsets": args.holdout_offsets, "sweeps": args.sweeps,
        "intensity_percentile": args.intensity_percentile, "bev_resolution_m": args.bev_resolution,
        "cache": {"dir": str(args.cache_dir), "hits": cache.hits, "inferred": cache.misses},
        "manual_body_noise": manual_noise.tolist(), "expected_body_correction": expected.tolist(),
        "gravity_initialized_body_correction": ground_correction.tolist(),
        "gravity_constrain": args.gravity_constrain, "refine_translation": args.refine_translation,
        "coarse_body_correction": coarse.tolist(), "refined_body_correction": refined.tolist(),
        "coarse_error_rpy_deg": np.rad2deg(coarse[:3] - expected[:3]).tolist(),
        "refined_error_rpy_deg": np.rad2deg(refined[:3] - expected[:3]).tolist(),
        "train_score_coarse": initial_train, "train_score_refined": final_train,
        "holdout_score_coarse": initial_holdout, "holdout_score_refined": final_holdout,
        "optimizer": {"success": bool(result.success), "message": result.message, "nfev": int(result.nfev),
                      "bounds_step": step.tolist(), "boundary_hit": boundary_hit},
        "publish_attitude": bool(result.success and final_holdout["median_px"] < initial_holdout["median_px"] and
                                 final_holdout["p90_px"] < initial_holdout["p90_px"] and not boundary_hit),
        "publish": bool(args.refine_translation and result.success and final_holdout["median_px"] < initial_holdout["median_px"] and
                        final_holdout["p90_px"] < initial_holdout["p90_px"] and not boundary_hit),
        "views": {
            "train": [{"camera": f["camera"], "high_intensity": f["high_intensity_count"],
                       "candidates": f["candidate_count"], "sampled": f["sampled_count"],
                       "teed_threshold": f["teed_threshold"]} for f in train_frames],
            "holdout": [{"camera": f["camera"], "high_intensity": f["high_intensity_count"],
                         "candidates": f["candidate_count"], "sampled": f["sampled_count"],
                         "teed_threshold": f["teed_threshold"]} for f in holdout_frames],
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
