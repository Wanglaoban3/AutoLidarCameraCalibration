import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def se3(x):
    R, _ = cv2.Rodrigues(np.asarray(x[:3], dtype=np.float64))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, x[3:]
    return T


def se3_vector(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return np.concatenate([rvec[:, 0], T[:3, 3]])


def sensor_to_global(points, calibrated, pose):
    p = points @ Quaternion(calibrated["rotation"]).rotation_matrix.T + np.asarray(calibrated["translation"])
    return p @ Quaternion(pose["rotation"]).rotation_matrix.T + np.asarray(pose["translation"])


def global_to_sensor(points, calibrated, pose):
    p = (points - np.asarray(pose["translation"])) @ Quaternion(pose["rotation"]).rotation_matrix
    return (p - np.asarray(calibrated["translation"])) @ Quaternion(calibrated["rotation"]).rotation_matrix


def image_edge_distance(image):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edge = cv2.Canny(gray, 70, 160)
    # Combine Canny texture edges with LSD long line segments.
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    line_mask = np.zeros_like(edge)
    segments = []
    if lines is not None:
        for x1, y1, x2, y2 in np.round(lines[:, 0]).astype(int):
            if np.hypot(x2 - x1, y2 - y1) < 20:
                continue
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            segments.append((x1, y1, x2, y2))
    edge = cv2.bitwise_or(edge, line_mask)
    segments.sort(key=lambda s: np.hypot(s[2] - s[0], s[3] - s[1]), reverse=True)
    return cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3), edge, np.asarray(segments[:200], dtype=np.float64)


def bilinear(image, uv):
    h, w = image.shape[:2]
    x, y = np.clip(uv[:, 0], 0, w - 1.001), np.clip(uv[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    return ((1 - wx) * (1 - wy) * image[y0, x0] + wx * (1 - wy) * image[y0, x1]
            + (1 - wx) * wy * image[y1, x0] + wx * wy * image[y1, x1])


def project(points, T, K, shape):
    q = (T[:3, :3] @ points.T + T[:3, 3:4]).T
    valid = q[:, 2] > 0.5
    uv = np.zeros((len(points), 2), dtype=np.float64)
    uv[valid] = (q[valid] @ K.T)[:, :2] / q[valid, 2:3]
    h, w = shape[:2]
    valid &= (uv[:, 0] >= 2) & (uv[:, 0] < w - 2) & (uv[:, 1] >= 2) & (uv[:, 1] < h - 2)
    return uv, valid


def lidar_occlusion_edges(points_cam, K, shape):
    uv, valid = project(points_cam, np.eye(4), K, shape)
    pts, uv = points_cam[valid], uv[valid]
    if len(pts) < 8:
        return pts, np.zeros((len(pts), 2))
    tree = cKDTree(uv)
    scores = np.zeros(len(pts))
    for i, ids in enumerate(tree.query_ball_point(uv, 18.0)):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) > 2:
            scores[i] = np.max(np.abs(pts[ids, 2] - pts[i, 2]))
    keep = scores > 0.6
    if keep.sum() < 30:
        keep = scores >= np.percentile(scores, 80)
    pts, uv = pts[keep], uv[keep]
    tangents = np.zeros((len(pts), 2))
    tree = cKDTree(uv)
    for i, ids in enumerate(tree.query_ball_point(uv, 25.0)):
        if len(ids) < 3:
            continue
        local = uv[np.asarray(ids)] - uv[i]
        _, _, vh = np.linalg.svd(local, full_matrices=False)
        tangents[i] = vh[0]
    return pts, tangents


def point_to_line_with_index(uv, segments, tangents=None, max_angle_deg=35.0):
    if len(segments) == 0 or len(uv) == 0:
        return np.full(len(uv), 100.0), np.full(len(uv), -1, dtype=np.int32)
    seg = segments[:500]
    a, b = seg[:, :2], seg[:, 2:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1) + 1e-9
    delta = uv[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(delta * ab[None, :, :], axis=2) / denom[None, :], 0.0, 1.0)
    q = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    distances = np.linalg.norm(uv[:, None, :] - q, axis=2)
    if tangents is not None and len(tangents) == len(uv):
        line_dir = ab / np.sqrt(denom)[:, None]
        tangent_norm = np.linalg.norm(tangents, axis=1)
        valid_tangent = tangent_norm > 1e-6
        tangent_dir = tangents.copy()
        tangent_dir[valid_tangent] /= tangent_norm[valid_tangent, None]
        alignment = np.abs(tangent_dir @ line_dir.T)
        # Edge directions are undirected, so compare absolute dot products.
        distances[valid_tangent] += 1000.0 * (alignment[valid_tangent] < np.cos(np.deg2rad(max_angle_deg)))
    indices = np.argmin(distances, axis=1)
    return distances[np.arange(len(uv)), indices], indices


def point_to_line(uv, segments):
    return point_to_line_with_index(uv, segments)[0]


def point_to_fixed_line(uv, segments, indices):
    if len(uv) == 0:
        return np.empty(0)
    valid_idx = (indices >= 0) & (indices < len(segments))
    out = np.full(len(uv), 100.0)
    seg = segments[indices[valid_idx]]
    a, b = seg[:, :2], seg[:, 2:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1) + 1e-9
    delta = uv[valid_idx] - a
    t = np.clip(np.sum(delta * ab, axis=1) / denom, 0.0, 1.0)
    q = a + t[:, None] * ab
    out[valid_idx] = np.linalg.norm(uv[valid_idx] - q, axis=1)
    return out


def coarse_rotation_search(frames, transform_fn, span_deg=8.0, step_deg=2.0):
    """Find a basin using the image distance field, without fixed correspondences."""
    values = np.arange(-span_deg, span_deg + 1e-6, step_deg)
    # A sparse, deterministic subset keeps the grid search bounded.
    samples = []
    for frame in frames:
        pts = frame["edge_pts"]
        samples.append(pts[::max(1, len(pts) // 180)])

    def score(x):
        scores = []
        T = transform_fn(x)
        for frame, pts in zip(frames, samples):
            uv, valid = project(pts, T, frame["K"], frame["shape"])
            if valid.sum() < 8:
                continue
            scores.append(float(np.median(bilinear(frame["dist"], uv[valid]))))
        return float(np.median(scores)) if scores else 1e6

    best = (1e6, np.zeros(6))
    for r in values:
        for p in values:
            for y in values:
                x = np.zeros(6, dtype=np.float64)
                x[:3] = np.deg2rad([r, p, y])
                value = score(x)
                if value < best[0]:
                    best = (value, x)
    # Refine the best grid cell at half-degree resolution.
    center = np.rad2deg(best[1][:3])
    fine = np.arange(-2.0, 2.01, 0.5)
    for dr in fine:
        for dp in fine:
            for dy in fine:
                x = np.zeros(6, dtype=np.float64)
                x[:3] = np.deg2rad(center + [dr, dp, dy])
                value = score(x)
                if value < best[0]:
                    best = (value, x)
    return best[1], best[0]


def draw_segments(image, segments, color=(255, 180, 0), thickness=1):
    canvas = image.copy()
    for x1, y1, x2, y2 in segments:
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return canvas


def draw_matches(image, frame, delta, transform_fn, color):
    canvas = image.copy()
    if len(frame["line_indices"]):
        canvas = draw_segments(canvas, frame["segments"][np.unique(frame["line_indices"])], (0, 220, 255), 2)
    uv, valid = project(frame["pts"], transform_fn(delta), frame["K"], frame["shape"])
    for u, v in uv[valid].astype(int):
        cv2.circle(canvas, (int(u), int(v)), 3, color, -1, cv2.LINE_AA)
    return canvas


def caption(image, text):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 620), 40), (0, 0, 0), -1)
    cv2.putText(canvas, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def contact_sheet(images):
    h = min(image.shape[0] for image in images)
    w = min(image.shape[1] for image in images)
    resized = [cv2.resize(image, (w // 2, h // 2)) for image in images]
    return np.vstack([np.hstack(resized[:2]), np.hstack(resized[2:])])


def draw_projected_points(image, points, K, T, color_mode="depth", radius=1):
    canvas = image.copy()
    uv, valid = project(points, T, K, image.shape)
    q = (T[:3, :3] @ points.T + T[:3, 3:4]).T
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        return canvas
    depth = q[ids, 2]
    normalized = np.clip((depth - 2.0) / 60.0, 0.0, 1.0)
    for j, idx in enumerate(ids):
        if color_mode == "depth":
            value = np.array([[int(normalized[j] * 255)]], dtype=np.uint8)
            color = tuple(int(v) for v in cv2.applyColorMap(value, cv2.COLORMAP_TURBO)[0, 0])
        else:
            color = (0, 0, 255)
        cv2.circle(canvas, tuple(uv[idx].astype(int)), radius, color, -1, cv2.LINE_AA)
    return canvas


def remove_dynamic_points(nusc, sample, global_points):
    dynamic = np.zeros(len(global_points), dtype=bool)
    dynamic_prefixes = ("vehicle.", "human.", "animal.", "movable_object.")
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if not ann["category_name"].startswith(dynamic_prefixes):
            continue
        center = np.asarray(ann["translation"], dtype=np.float64)
        size = np.asarray(ann["size"], dtype=np.float64)
        R = Quaternion(ann["rotation"]).rotation_matrix
        local = (global_points - center) @ R
        dynamic |= np.all(np.abs(local) <= (size[None, :] * 0.5 + 0.15), axis=1)
    return global_points[~dynamic], int(dynamic.sum())


def ground_normal(points_ego):
    candidates = points_ego[(points_ego[:, 2] < -0.5) & (points_ego[:, 2] > -3.5)]
    if len(candidates) < 30:
        return np.array([0.0, 0.0, 1.0])
    # The low-percentile slab is robust enough here because road points are the
    # dominant horizontal structure below the sensor rig.
    center = np.median(candidates, axis=0)
    _, _, vh = np.linalg.svd(candidates - center, full_matrices=False)
    normal = vh[-1]
    return normal if normal[2] >= 0 else -normal


def load_frame(nusc, sample, cam_name, dataroot):
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    raw = np.fromfile(Path(dataroot) / lidar_sd["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]
    global_points = sensor_to_global(raw, lidar_cs, lidar_pose)
    global_points, dynamic_count = remove_dynamic_points(nusc, sample, global_points)
    ego_points = (global_points - np.asarray(lidar_pose["translation"])) @ Quaternion(lidar_pose["rotation"]).rotation_matrix
    ground = ground_normal(ego_points)
    global_points = global_points[ego_points[:, 2] > 0.25]
    sd = nusc.get("sample_data", sample["data"][cam_name])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    image = cv2.imread(str(Path(dataroot) / sd["filename"]))
    dist, edge, segments = image_edge_distance(image)
    K = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
    p_cam = global_to_sensor(global_points, cs, pose)
    _, valid = project(p_cam, np.eye(4), K, image.shape)
    pts, tangents = lidar_occlusion_edges(p_cam[valid], K, image.shape)
    T_ego_camera = np.eye(4)
    T_ego_camera[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    T_ego_camera[:3, 3] = np.asarray(cs["translation"])
    return {"all_pts": p_cam[valid], "edge_pts": pts, "pts": pts[::4], "tangents": tangents[::4], "dist": dist, "segments": segments, "image": image, "edge": edge, "K": K, "shape": image.shape, "filename": sd["filename"], "dynamic_removed": dynamic_count, "ground_normal_ego": ground, "T_ego_camera": T_ego_camera}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--out", default="/workspace/results/nuscenes")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--noise-rpy-deg", type=float, nargs=3, default=[0.86, -0.57, 1.15], metavar=("ROLL", "PITCH", "YAW"))
    ap.add_argument("--noise-translation-m", type=float, nargs=3, default=[0.06, -0.04, 0.08], metavar=("X", "Y", "Z"))
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
    report = {"scene": scene["name"], "frames": len(samples), "cameras": {}}
    ego_corrections = {}
    for cam_name in CAMERAS:
        datasets = [load_frame(nusc, sample, cam_name, args.dataroot) for sample in samples]
        manual_noise = np.concatenate([np.deg2rad(args.noise_rpy_deg), args.noise_translation_m])
        T_manual = se3(manual_noise)
        x0 = np.zeros(6)

        def current_transform(delta):
            return se3(delta) @ T_manual

        coarse_x, coarse_score = coarse_rotation_search(datasets, current_transform)
        # Rebuild correspondences at the coarse basin; matching at the manual
        # pose is unreliable when the initial attitude error is several degrees.
        matching_start = coarse_x
        # Establish stable point-to-line candidates using the manual extrinsic;
        # this prevents unrelated LSD segments from becoming correspondences.
        for d in datasets:
            uv0, valid0 = project(d["pts"], current_transform(matching_start), d["K"], d["shape"])
            keep = np.zeros(len(d["pts"]), dtype=bool)
            distances, line_indices = point_to_line_with_index(uv0[valid0], d["segments"], d["tangents"][valid0])
            candidate = distances < 20.0
            if candidate.sum() < 30:
                distances, line_indices = point_to_line_with_index(
                    uv0[valid0], d["segments"], d["tangents"][valid0], max_angle_deg=55.0
                )
                candidate = distances < 25.0
            keep[valid0] = candidate
            d["pts"] = d["pts"][keep]
            d["line_indices"] = line_indices[candidate]

        def per_frame_residual(x, frames):
            values = []
            for d in frames:
                uv, valid = project(d["pts"], current_transform(x), d["K"], d["shape"])
                r = np.full(len(d["pts"]), 100.0)
                r[valid] = point_to_fixed_line(uv[valid], d["segments"], d["line_indices"][valid])
                values.append(r)
            return values

        def edge_residual(x, frames):
            values = per_frame_residual(x, frames)
            return np.concatenate(values) if values else np.array([100.0])

        def residual(x, frames):
            prior = 0.25 * (x - matching_start) / np.array([0.03, 0.03, 0.03, 0.08, 0.08, 0.08])
            return np.concatenate([edge_residual(x, frames), prior])

        def optimize(frames, start):
            return least_squares(
                lambda x: residual(x, frames), start, loss="huber", f_scale=3.0,
                bounds=(-np.array([0.20, 0.20, 0.20, 0.30, 0.30, 0.30]),
                        np.array([0.20, 0.20, 0.20, 0.30, 0.30, 0.30])), max_nfev=150
            )

        initial = float(np.mean(edge_residual(x0, datasets)))
        stage_one = optimize(datasets, matching_start)
        frame_medians = np.array([np.median(r) if len(r) else np.inf for r in per_frame_residual(stage_one.x, datasets)])
        finite = frame_medians[np.isfinite(frame_medians)]
        median = np.median(finite)
        mad = np.median(np.abs(finite - median)) + 1e-6
        active = [d for d, score in zip(datasets, frame_medians) if len(d["pts"]) >= 20 and score <= median + 2.5 * mad]
        if len(active) < 3:
            active = datasets
        result = optimize(active, stage_one.x)
        first = datasets[0]
        overlay = first["image"].copy()
        for state, color in ((x0, (0, 0, 255)), (result.x, (0, 255, 0))):
            uv_state, valid_state = project(first["pts"], current_transform(state), first["K"], first["shape"])
            for u, v in uv_state[valid_state][::4].astype(int):
                cv2.circle(overlay, (int(u), int(v)), 2, color, -1)
        T_ego_delta = first["T_ego_camera"] @ se3(result.x) @ np.linalg.inv(first["T_ego_camera"])
        ego_corrections[cam_name] = se3_vector(T_ego_delta)
        report["cameras"][cam_name] = {"frames": len(datasets), "frames_kept": len(active), "image": first["filename"], "dynamic_points_removed_per_frame": [int(d["dynamic_removed"]) for d in datasets], "points_used_per_frame": [int(len(d["pts"])) for d in datasets], "frame_median_residual_px": frame_medians.tolist(), "manual_noise": manual_noise.tolist(), "expected_correction": se3_vector(np.linalg.inv(T_manual)).tolist(), "coarse_correction": matching_start.tolist(), "coarse_distance_px": float(coarse_score), "estimated_correction": result.x.tolist(), "ego_frame_correction": ego_corrections[cam_name].tolist(), "initial_point_to_line_px": initial, "final_point_to_line_px": float(np.mean(edge_residual(result.x, active)))}
        cv2.imwrite(str(out / f"{cam_name}_edges.png"), first["edge"])
        cv2.imwrite(str(out / f"{cam_name}_overlay.jpg"), overlay)
        raw_panel = caption(first["image"], "1. Camera image")
        line_panel = caption(draw_segments(first["image"], first["segments"]), "2. Canny + LSD long segments")
        initial_panel = caption(draw_matches(first["image"], first, x0, current_transform, (0, 0, 255)), "3. Manual extrinsic: fixed point-to-line matches")
        final_panel = caption(draw_matches(first["image"], first, result.x, current_transform, (0, 255, 0)), "4. Coarse-search + optimized delta_T")
        cv2.imwrite(str(out / f"{cam_name}_pipeline.jpg"), contact_sheet([raw_panel, line_panel, initial_panel, final_panel]))
        full_initial = caption(draw_projected_points(first["image"], first["all_pts"], first["K"], current_transform(x0), radius=1), "Full non-ground LiDAR projection: manual extrinsic")
        full_coarse = caption(draw_projected_points(first["image"], first["all_pts"], first["K"], current_transform(matching_start), radius=1), "Full non-ground LiDAR projection: distance-field coarse search")
        full_final = caption(draw_projected_points(first["image"], first["all_pts"], first["K"], current_transform(result.x), radius=1), "Full non-ground LiDAR projection: optimized delta_T")
        edge_initial = caption(draw_projected_points(first["image"], first["edge_pts"], first["K"], current_transform(x0), color_mode="edge", radius=3), "LiDAR depth-discontinuity edge projection: manual extrinsic")
        edge_coarse = caption(draw_projected_points(first["image"], first["edge_pts"], first["K"], current_transform(matching_start), color_mode="edge", radius=3), "LiDAR depth-discontinuity edge projection: distance-field coarse search")
        edge_final = caption(draw_projected_points(first["image"], first["edge_pts"], first["K"], current_transform(result.x), color_mode="edge", radius=3), "LiDAR depth-discontinuity edge projection: optimized delta_T")
        cv2.imwrite(str(out / f"{cam_name}_full_lidar_initial.jpg"), full_initial)
        cv2.imwrite(str(out / f"{cam_name}_full_lidar_coarse.jpg"), full_coarse)
        cv2.imwrite(str(out / f"{cam_name}_full_lidar_final.jpg"), full_final)
        cv2.imwrite(str(out / f"{cam_name}_lidar_edges_initial.jpg"), edge_initial)
        cv2.imwrite(str(out / f"{cam_name}_lidar_edges_coarse.jpg"), edge_coarse)
        cv2.imwrite(str(out / f"{cam_name}_lidar_edges_final.jpg"), edge_final)
    correction_matrix = np.asarray(list(ego_corrections.values()))
    consensus = np.median(correction_matrix, axis=0)
    for cam_name, correction in ego_corrections.items():
        camera = report["cameras"][cam_name]
        delta = correction - consensus
        rotation_deviation = float(np.linalg.norm(delta[:3]))
        translation_deviation = float(np.linalg.norm(delta[3:]))
        retained_ratio = camera["frames_kept"] / camera["frames"]
        reasons = []
        if sum(camera["points_used_per_frame"]) < 300:
            reasons.append("insufficient_edge_points")
        if retained_ratio < 0.35:
            reasons.append("insufficient_consistent_frames")
        if camera["final_point_to_line_px"] > 12.0:
            reasons.append("high_point_to_line_residual")
        if rotation_deviation > 0.03 or translation_deviation > 0.12:
            reasons.append("cross_camera_inconsistency")
        camera["health"] = {"retained_frame_ratio": retained_ratio, "rotation_deviation_rad": rotation_deviation, "translation_deviation_m": translation_deviation, "action": "publish_candidate" if not reasons else "freeze", "reasons": reasons}
    report["vehicle_consensus"] = {"ego_frame_correction": consensus.tolist(), "policy": "publish only cameras whose health.action is publish_candidate"}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    rows = []
    for cam_name, camera in report["cameras"].items():
        health = camera["health"]
        rows.append(
            f"<tr><td>{cam_name}</td><td>{camera['frames_kept']}/{camera['frames']}</td>"
            f"<td>{camera['initial_point_to_line_px']:.2f} -> {camera['final_point_to_line_px']:.2f}</td>"
            f"<td>{health['action']}</td><td>{', '.join(health['reasons']) or '-'}</td>"
            f"<td><a href='{cam_name}_pipeline.jpg'>pipeline</a> | <a href='{cam_name}_full_lidar_initial.jpg'>manual cloud</a> | <a href='{cam_name}_full_lidar_coarse.jpg'>coarse cloud</a> | <a href='{cam_name}_lidar_edges_coarse.jpg'>coarse edges</a></td></tr>"
        )
    html = """<!doctype html><html><head><meta charset='utf-8'><title>LiDAR-Camera Calibration Report</title>
<style>body{font-family:Arial,sans-serif;margin:28px;color:#18212b}table{border-collapse:collapse;width:100%%}td,th{border:1px solid #ccd3db;padding:8px;text-align:left}th{background:#edf2f6}.publish_candidate{color:#067647}.freeze{color:#b42318}</style></head><body>
<h1>Multi-frame LiDAR-Camera Calibration</h1><p>Scene: %s. Window: %d frames. Each pipeline image is: camera image, LSD segments, manual-extrinsic matches, optimized matches.</p>
<h2>Six-camera health gate</h2><table><tr><th>Camera</th><th>Retained frames</th><th>Point-to-line px</th><th>Action</th><th>Reasons</th><th>Visuals</th></tr>%s</table>
<h2>Ego-frame consensus correction</h2><pre>%s</pre><p><a href='report.json'>Raw JSON report</a></p></body></html>""" % (report["scene"], report["frames"], "".join(rows), json.dumps(report["vehicle_consensus"], indent=2))
    (out / "index.html").write_text(html)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
