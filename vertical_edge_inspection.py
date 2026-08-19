"""Visualize stacked static LiDAR vertical-edge candidates before tracking."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from nuscenes_edge_demo import project
from stacked_lidar_demo import dynamic_mask, estimate_ground_mask, sensor_to_global
from teed_ground_inspection import collect_sweeps
from teed_vertical_roll_refinement import vertical_column_mask, vertical_depth_edges


def caption(image, text):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (min(1120, canvas.shape[1]), 42), (0, 0, 0), -1)
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw(image, points, K, color, radius=2):
    uv, visible = project(points, np.eye(4), K, image.shape)
    canvas = image.copy()
    for point in uv[visible].astype(int):
        cv2.circle(canvas, tuple(point), radius, color, -1, cv2.LINE_AA)
    return canvas, int(visible.sum())


def sample_at_offset(nusc, scene, offset):
    token = scene["first_sample_token"]
    for _ in range(offset):
        sample = nusc.get("sample", token)
        if not sample["next"]:
            raise ValueError("reference offset exceeds scene")
        token = sample["next"]
    return nusc.get("sample", token)


def scene_timing(nusc, scene):
    token, count, first, last = scene["first_sample_token"], 0, None, None
    while token:
        sample = nusc.get("sample", token)
        timestamp = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["timestamp"]
        first = timestamp if first is None else first
        last, count, token = timestamp, count + 1, sample["next"]
    return count, (last - first) / 1e6


def main():
    parser = argparse.ArgumentParser(description="Inspect stacked LiDAR vertical contours")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--out", default="/workspace/results/vertical_edge_inspection")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--reference-offset", type=int, default=9)
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--sweep-counts", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--column-cell-size", type=float, default=0.5)
    parser.add_argument("--column-min-height", type=float, default=1.2)
    parser.add_argument("--column-min-points", type=int, default=5)
    args = parser.parse_args()
    if not args.sweep_counts or min(args.sweep_counts) < 1:
        parser.error("--sweep-counts must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    scene_samples, scene_duration_s = scene_timing(nusc, scene)
    reference = sample_at_offset(nusc, scene, args.reference_offset)
    camera_sd = nusc.get("sample_data", reference["data"][args.camera])
    camera = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
    reference_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])
    image = cv2.imread(str(Path(args.dataroot) / camera_sd["filename"]))
    if image is None:
        raise FileNotFoundError(camera_sd["filename"])
    K = np.asarray(camera["camera_intrinsic"], dtype=np.float64)
    R_camera = Quaternion(camera["rotation"]).rotation_matrix
    history = []
    for lidar_sd, annotation_sample in collect_sweeps(nusc, reference, max(args.sweep_counts)):
        lidar = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        raw = np.fromfile(Path(args.dataroot) / lidar_sd["filename"], dtype=np.float32).reshape(-1, 5)
        global_points = sensor_to_global(raw[:, :3], lidar, pose)
        movable = dynamic_mask(nusc, annotation_sample, global_points)
        ego = (global_points[~movable] - np.asarray(reference_pose["translation"])) @ Quaternion(reference_pose["rotation"]).rotation_matrix
        history.append({"ego": ego, "dynamic_removed": int(movable.sum())})
    report = {"scene": scene["name"], "scene_keyframes": scene_samples, "scene_duration_s": scene_duration_s,
              "camera": args.camera, "reference_sample": reference["token"],
              "reference_offset": args.reference_offset, "sweeps_available": len(history),
              "dynamic_removed_per_sweep": [item["dynamic_removed"] for item in history], "stacks": {}}
    for count in sorted(set(min(len(history), value) for value in args.sweep_counts)):
        ego = np.concatenate([item["ego"] for item in history[-count:]])
        camera_points = (ego - np.asarray(camera["translation"])) @ R_camera
        ground = estimate_ground_mask(ego)
        column = (~ground) & vertical_column_mask(ego, args.column_cell_size,
                                                    args.column_min_height, args.column_min_points)
        edges = vertical_depth_edges(camera_points, column, K, image.shape)
        full, full_visible = draw(image, camera_points, K, (0, 140, 255), 1)
        columns, column_visible = draw(image, camera_points[column], K, (255, 255, 0), 2)
        edge_view, edge_visible = draw(image, edges, K, (0, 0, 255), 3)
        panels = [caption(full, f"1 Static stacked LiDAR: {count} sweeps"),
                  caption(columns, "2 3D vertical-column support"),
                  caption(edge_view, "3 Vertical depth-discontinuity edges")]
        height, width = image.shape[:2]
        sheet = np.hstack([cv2.resize(panel, (width // 2, height // 2)) for panel in panels])
        stem = f"{args.camera}_sweeps_{count:02d}"
        cv2.imwrite(str(out / f"{stem}_comparison.jpg"), sheet)
        cv2.imwrite(str(out / f"{stem}_all_points.jpg"), panels[0])
        cv2.imwrite(str(out / f"{stem}_column_support.jpg"), panels[1])
        cv2.imwrite(str(out / f"{stem}_vertical_edges.jpg"), panels[2])
        report["stacks"][str(count)] = {"static_points": int(len(ego)), "visible_points": full_visible,
                                         "column_support_points": int(column.sum()),
                                         "column_support_visible": column_visible,
                                         "vertical_edge_points": int(len(edges)),
                                         "vertical_edge_visible": edge_visible}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
