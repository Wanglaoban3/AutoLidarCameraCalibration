"""Inspect motion-compensated LiDAR history and ground-return candidates.

The reference image is the last sample in the selected window.  Earlier scans
are transformed through nuScenes global poses into that image's camera frame.
This makes the visualisation useful for judging whether stacking gives enough
ground structure to constrain roll/pitch before using it in calibration.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from nuscenes_edge_demo import global_to_sensor, project


def sensor_to_global(points, calibrated, pose):
    points = points @ Quaternion(calibrated["rotation"]).rotation_matrix.T
    points += np.asarray(calibrated["translation"])
    return points @ Quaternion(pose["rotation"]).rotation_matrix.T + np.asarray(pose["translation"])


def dynamic_mask(nusc, sample, global_points):
    """Return points inside annotated movable objects for this sample."""
    mask = np.zeros(len(global_points), dtype=bool)
    prefixes = ("vehicle.", "human.", "animal.", "movable_object.")
    for token in sample["anns"]:
        annotation = nusc.get("sample_annotation", token)
        if not annotation["category_name"].startswith(prefixes):
            continue
        center = np.asarray(annotation["translation"], dtype=np.float64)
        extent = np.asarray(annotation["size"], dtype=np.float64) * 0.5 + 0.15
        local = (global_points - center) @ Quaternion(annotation["rotation"]).rotation_matrix
        mask |= np.all(np.abs(local) <= extent, axis=1)
    return mask


def estimate_ground_mask(points_ego):
    """Robust local plane fit in the reference ego frame.

    Lane markings are not guaranteed to have a distinct nuScenes intensity, so
    this deliberately reports high-return *candidates*, not lane detections.
    """
    if len(points_ego) < 30:
        return np.zeros(len(points_ego), dtype=bool)
    # nuScenes ego-pose origin is near the ground, unlike a LiDAR sensor frame
    # where road returns are normally around z=-1.8 m.
    initial = ((points_ego[:, 0] > -15.0) & (points_ego[:, 0] < 70.0) &
               (np.abs(points_ego[:, 1]) < 25.0) & (points_ego[:, 2] < 0.5) &
               (points_ego[:, 2] > -1.5))
    if initial.sum() < 30:
        return np.zeros(len(points_ego), dtype=bool)
    candidates = points_ego[initial]
    center = np.median(candidates, axis=0)
    _, _, vh = np.linalg.svd(candidates - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    distance = np.abs((points_ego - center) @ normal)
    return initial & (distance < 0.18)


def draw_points(image, uv, mask, color, radius=2):
    canvas = image.copy()
    for point in uv[mask].astype(int):
        cv2.circle(canvas, tuple(point), radius, color, -1, cv2.LINE_AA)
    return canvas


def visible_surface_edges(points_camera, K, shape, cell_size=4):
    """Detect depth jumps only on the closest stacked return in each image cell.

    A radius search over all stacked points incorrectly regards temporally
    separated front/back surfaces as one local discontinuity.  The z-buffer
    leaves one visible surface per cell before comparing neighbouring depths.
    """
    uv, valid = project(points_camera, np.eye(4), K, shape)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return np.empty((0, 3), dtype=np.float64)
    h, w = shape[:2]
    rows, cols = (h + cell_size - 1) // cell_size, (w + cell_size - 1) // cell_size
    depth = np.full((rows, cols), np.inf)
    point_ids = np.full((rows, cols), -1, dtype=np.int32)
    for index in ids:
        col, row = (uv[index] / cell_size).astype(int)
        z = points_camera[index, 2]
        if z < depth[row, col]:
            depth[row, col], point_ids[row, col] = z, index
    selected = point_ids >= 0
    edge = np.zeros_like(selected)
    for dr, dc in ((0, 1), (1, 0)):
        a = (slice(0, rows - dr), slice(0, cols - dc))
        b = (slice(dr, rows), slice(dc, cols))
        present = selected[a] & selected[b]
        difference = np.zeros_like(depth[a])
        difference[present] = np.abs(depth[a][present] - depth[b][present])
        threshold = np.maximum(0.8, 0.04 * np.minimum(depth[a], depth[b]))
        hit = present & (difference > threshold)
        edge[a] |= hit
        edge[b] |= hit
    return points_camera[point_ids[edge]]


def caption(image, text):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (min(1050, canvas.shape[1]), 42), (0, 0, 0), -1)
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize historical LiDAR stacking on a reference camera frame")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--out", default="/workspace/results/stacked_lidar")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--frames", type=int, default=10, help="History window including the reference scan")
    parser.add_argument("--sweeps", type=int, default=0, help="Use this many consecutive LiDAR sweeps ending at the reference scan")
    parser.add_argument("--stack-counts", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--camera", default="CAM_FRONT")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    samples, token = [], scene["first_sample_token"]
    while token and len(samples) < args.frames:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]
    if args.camera not in samples[-1]["data"]:
        parser.error(f"camera {args.camera} is unavailable")

    # The last sample is the reference; every preceding sample is genuine history.
    ref_sample = samples[-1]
    ref_sd = nusc.get("sample_data", ref_sample["data"][args.camera])
    ref_camera = nusc.get("calibrated_sensor", ref_sd["calibrated_sensor_token"])
    ref_pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])
    image = cv2.imread(str(Path(args.dataroot) / ref_sd["filename"]))
    if image is None:
        raise FileNotFoundError(ref_sd["filename"])
    K = np.asarray(ref_camera["camera_intrinsic"], dtype=np.float64)

    # Keyframes have 0.5 s spacing.  For short-range densification, follow the
    # sample-data chain instead: nuScenes LiDAR sweeps are approximately 20 Hz.
    keyframe_history = []
    sample_token = ref_sample["token"]
    required_keyframes = max(2, int(np.ceil(max(args.sweeps, 1) / 10)) + 2)
    while sample_token and len(keyframe_history) < required_keyframes:
        sample = nusc.get("sample", sample_token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        keyframe_history.append((lidar_sd["timestamp"], sample))
        sample_token = sample["prev"]

    if args.sweeps:
        if args.sweeps < 1:
            parser.error("--sweeps must be positive")
        lidar_sweeps = []
        token = ref_sample["data"]["LIDAR_TOP"]
        while token and len(lidar_sweeps) < args.sweeps:
            lidar_sd = nusc.get("sample_data", token)
            nearest_sample = min(keyframe_history, key=lambda pair: abs(pair[0] - lidar_sd["timestamp"]))[1]
            lidar_sweeps.append((lidar_sd, nearest_sample))
            token = lidar_sd["prev"]
        scan_inputs = list(reversed(lidar_sweeps))
        source = "consecutive_sweeps"
    else:
        scan_inputs = [
            (nusc.get("sample_data", sample["data"]["LIDAR_TOP"]), sample)
            for sample in samples
        ]
        source = "keyframe_scans"

    history = []
    for lidar_sd, annotation_sample in scan_inputs:
        lidar_sensor = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        raw = np.fromfile(Path(args.dataroot) / lidar_sd["filename"], dtype=np.float32).reshape(-1, 5)
        global_points = sensor_to_global(raw[:, :3], lidar_sensor, lidar_pose)
        # Sweeps do not have their own annotations.  The closest 2 Hz keyframe
        # supplies a conservative movable-object mask for visualization.
        movable = dynamic_mask(nusc, annotation_sample, global_points)
        global_points, intensity = global_points[~movable], raw[~movable, 3]
        ref_ego = (global_points - np.asarray(ref_pose["translation"])) @ Quaternion(ref_pose["rotation"]).rotation_matrix
        ref_camera_points = global_to_sensor(global_points, ref_camera, ref_pose)
        history.append({
            "camera": ref_camera_points, "ego": ref_ego, "intensity": intensity,
            "raw_points": len(raw), "dynamic_removed": int(movable.sum()),
        })

    counts = sorted({min(len(history), count) for count in args.stack_counts if count > 0})
    report = {
        "scene": scene["name"], "camera": args.camera, "reference_sample": ref_sample["token"],
        "source": source, "history_frames_available": len(history) - 1, "window_frames": len(history),
        "dynamic_removed_per_scan": [item["dynamic_removed"] for item in history], "stacks": {},
    }
    for count in counts:
        # Keep the reference scan and its nearest count-1 historical scans.
        selected = history[-count:]
        points = np.concatenate([item["camera"] for item in selected])
        ego_points = np.concatenate([item["ego"] for item in selected])
        intensity = np.concatenate([item["intensity"] for item in selected])
        uv, visible = project(points, np.eye(4), K, image.shape)
        ground = estimate_ground_mask(ego_points) & visible
        # This is intentionally a conservative inspection cue, not a lane label.
        threshold = float(np.percentile(intensity[ground], 90)) if ground.sum() >= 20 else float("inf")
        bright_ground = ground & (intensity >= threshold)
        edge_points = visible_surface_edges(points, K, image.shape)

        full = image.copy()
        for point in uv[visible].astype(int):
            cv2.circle(full, tuple(point), 1, (0, 90, 255), -1, cv2.LINE_AA)
        ground_view = draw_points(image, uv, ground, (0, 220, 255), 2)
        lane_view = draw_points(ground_view, uv, bright_ground, (255, 255, 0), 3)
        edge_view = image.copy()
        edge_uv, edge_visible = project(edge_points, np.eye(4), K, image.shape)
        edge_view = draw_points(edge_view, edge_uv, edge_visible, (0, 0, 255), 3)
        panels = [
            caption(full, f"Static LiDAR: reference + {count - 1} historical scans"),
            caption(lane_view, "Ground returns: yellow; high-reflectivity candidates: cyan"),
            caption(edge_view, "Z-buffered LiDAR depth-discontinuity candidates"),
        ]
        h, w = image.shape[:2]
        sheet = np.hstack([cv2.resize(panel, (w // 2, h // 2)) for panel in panels])
        stem = f"{args.camera}_history_{count:02d}"
        cv2.imwrite(str(out / f"{stem}_full.jpg"), panels[0])
        cv2.imwrite(str(out / f"{stem}_ground.jpg"), panels[1])
        cv2.imwrite(str(out / f"{stem}_edges.jpg"), panels[2])
        cv2.imwrite(str(out / f"{stem}_comparison.jpg"), sheet)
        report["stacks"][str(count)] = {
            "scans": count, "historical_scans": count - 1, "static_points": int(len(points)),
            "projected_points": int(visible.sum()), "ground_projected_points": int(ground.sum()),
            "high_reflectivity_ground_candidates": int(bright_ground.sum()),
            "depth_discontinuity_candidates": int(len(edge_points)),
            "ground_intensity_p90": threshold if np.isfinite(threshold) else None,
        }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
