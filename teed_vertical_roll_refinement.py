"""Roll-only refinement from stacked LiDAR vertical contours and cached TEED edges."""
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
from teed_ground_inspection import collect_sweeps
from teed_model import TEED


def collect_samples(nusc, scene, offsets):
    wanted, result, token, offset = set(offsets), [], scene["first_sample_token"], 0
    while token and offset <= max(wanted):
        sample = nusc.get("sample", token)
        if offset in wanted:
            result.append((offset, sample))
        token, offset = sample["next"], offset + 1
    missing = wanted - {index for index, _ in result}
    if missing:
        raise ValueError(f"sample offsets outside scene: {sorted(missing)}")
    return result


def vertical_depth_edges(points_camera, allowed, K, shape, cell_size=4, min_vertical_cells=3):
    """Closest-surface depth jumps that persist vertically in the image."""
    uv, visible = project(points_camera, np.eye(4), K, shape)
    ids = np.flatnonzero(visible & allowed)
    if not len(ids):
        return np.empty((0, 3), dtype=np.float64)
    height, width = shape[:2]
    rows, cols = (height + cell_size - 1) // cell_size, (width + cell_size - 1) // cell_size
    depth = np.full((rows, cols), np.inf)
    point_ids = np.full((rows, cols), -1, dtype=np.int32)
    for index in ids:
        col, row = (uv[index] / cell_size).astype(int)
        if points_camera[index, 2] < depth[row, col]:
            depth[row, col], point_ids[row, col] = points_camera[index, 2], index
    present = point_ids >= 0
    edge = np.zeros_like(present)
    left, right = (slice(None), slice(0, cols - 1)), (slice(None), slice(1, cols))
    both = present[left] & present[right]
    difference = np.zeros((rows, cols - 1))
    difference[both] = np.abs(depth[left][both] - depth[right][both])
    threshold = np.maximum(0.8, 0.04 * np.minimum(depth[left], depth[right]))
    hit = both & (difference > threshold)
    edge[left] |= hit
    edge[right] |= hit
    support = cv2.filter2D(edge.astype(np.uint8), -1, np.ones((min_vertical_cells, 1), np.uint8))
    selected = edge & (support >= min_vertical_cells)
    return points_camera[point_ids[selected]]


def vertical_column_mask(points_ego, cell_size=0.5, min_height=1.2, min_points=5):
    """Keep returns supported by a vertically extended 3D column."""
    range_xy = np.linalg.norm(points_ego[:, :2], axis=1)
    region = ((range_xy > 3.0) & (range_xy < 55.0) &
              (points_ego[:, 2] > -0.4) & (points_ego[:, 2] < 4.0))
    ids = np.flatnonzero(region)
    selected = np.zeros(len(points_ego), dtype=bool)
    if not len(ids):
        return selected
    cells = np.floor(points_ego[ids, :2] / cell_size).astype(np.int32)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    z_min = np.full(len(count), np.inf)
    z_max = np.full(len(count), -np.inf)
    np.minimum.at(z_min, inverse, points_ego[ids, 2])
    np.maximum.at(z_max, inverse, points_ego[ids, 2])
    valid_column = (count >= min_points) & ((z_max - z_min) >= min_height)
    selected[ids] = valid_column[inverse]
    return selected


def body_to_camera(frame, body_delta):
    return np.linalg.inv(frame["T_ego_camera"]) @ se3(body_delta) @ frame["T_ego_camera"] @ frame["T_manual"]


def build_frame(nusc, sample, camera_name, dataroot, sweeps, cache, teed_percentile, max_points,
                column_cell_size, column_min_height, column_min_points):
    camera_sd = nusc.get("sample_data", sample["data"][camera_name])
    camera = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
    reference_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])
    image = cv2.imread(str(Path(dataroot) / camera_sd["filename"]))
    if image is None:
        raise FileNotFoundError(camera_sd["filename"])
    K = np.asarray(camera["camera_intrinsic"], dtype=np.float64)
    stacked = []
    for lidar_sd, annotation_sample in collect_sweeps(nusc, sample, sweeps):
        lidar = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        raw = np.fromfile(Path(dataroot) / lidar_sd["filename"], dtype=np.float32).reshape(-1, 5)
        global_points = sensor_to_global(raw[:, :3], lidar, pose)
        stacked.append(global_points[~dynamic_mask(nusc, annotation_sample, global_points)])
    points_ego = np.concatenate(stacked)
    points_ego = (points_ego - np.asarray(reference_pose["translation"])) @ Quaternion(reference_pose["rotation"]).rotation_matrix
    R_camera = Quaternion(camera["rotation"]).rotation_matrix
    points_camera = (points_ego - np.asarray(camera["translation"])) @ R_camera
    ground = estimate_ground_mask(points_ego)
    vertical_region = ~ground & vertical_column_mask(points_ego, column_cell_size,
                                                      column_min_height, column_min_points)
    points = vertical_depth_edges(points_camera, vertical_region, K, image.shape)
    if len(points) > max_points:
        points = points[::int(np.ceil(len(points) / max_points))]
    probability = cache.probability(image, camera_sd["token"])
    threshold = float(np.percentile(probability, teed_percentile))
    gx = cv2.Sobel(probability, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(probability, cv2.CV_32F, 0, 1, ksize=3)
    vertical_image_edge = (probability >= threshold) & (np.abs(gx) >= 0.75 * np.hypot(gx, gy))
    distance = cv2.distanceTransform((vertical_image_edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    T_ego_camera = np.eye(4)
    T_ego_camera[:3, :3], T_ego_camera[:3, 3] = R_camera, np.asarray(camera["translation"])
    return {"camera": camera_name, "image": image, "shape": image.shape, "K": K, "points": points,
            "distance": distance, "edge": vertical_image_edge, "candidate_count": int(len(points)),
            "teed_threshold": threshold, "T_ego_camera": T_ego_camera}


def residuals(frames, body_delta):
    output = []
    for frame in frames:
        if not len(frame["points"]):
            continue
        uv, valid = project(frame["points"], body_to_camera(frame, body_delta), frame["K"], frame["shape"])
        values = np.full(len(frame["points"]), 30.0)
        values[valid] = bilinear(frame["distance"], uv[valid])
        output.append(values)
    return np.concatenate(output) if output else np.full(1, 30.0)


def score(frames, body_delta):
    values = residuals(frames, body_delta)
    return {"mean_px": float(np.mean(values)), "median_px": float(np.median(values)),
            "p90_px": float(np.percentile(values, 90)), "points": int(len(values))}


def draw_overlay(frame, body_delta, title):
    canvas = frame["image"].copy()
    canvas[frame["edge"]] = (0, 160, 255)
    if len(frame["points"]):
        uv, valid = project(frame["points"], body_to_camera(frame, body_delta), frame["K"], frame["shape"])
        for point in uv[valid].astype(int):
            cv2.circle(canvas, tuple(point), 2, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (min(1100, canvas.shape[1]), 42), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Refine roll from cached TEED vertical edges and stacked LiDAR")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--initial-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", default="/workspace/results/teed_vertical_roll")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--train-offsets", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--holdout-offsets", type=int, nargs="+", default=[10, 12])
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--teed-percentile", type=float, default=95.0)
    parser.add_argument("--max-points-per-view", type=int, default=400)
    parser.add_argument("--column-cell-size", type=float, default=0.5)
    parser.add_argument("--column-min-height", type=float, default=1.2)
    parser.add_argument("--column-min-points", type=int, default=5)
    parser.add_argument("--max-roll-step-deg", type=float, default=1.2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if set(args.train_offsets) & set(args.holdout_offsets):
        parser.error("train and holdout offsets must not overlap")
    prior = json.loads(Path(args.initial_json).read_text())
    initial = np.asarray(prior.get("refined_body_correction", prior.get("estimated_body_correction")), dtype=np.float64)
    if initial.shape != (6,):
        parser.error("initial JSON must contain a six-vector body correction")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    model = TEED().to(device)
    model.load_state_dict(torch.load("/workspace/models/teed_biped_epoch5.pth", map_location=device, weights_only=True), strict=True)
    model.eval()
    cache = TEEDCache(args.cache_dir, model, device)
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    transform = se3(np.asarray(prior["manual_body_noise"], dtype=np.float64))
    frames_by_split = []
    for offsets in (args.train_offsets, args.holdout_offsets):
        frames = []
        for _, sample in collect_samples(nusc, scene, offsets):
            for camera in CAMERAS:
                frame = build_frame(nusc, sample, camera, args.dataroot, args.sweeps, cache,
                                    args.teed_percentile, args.max_points_per_view, args.column_cell_size,
                                    args.column_min_height, args.column_min_points)
                frame["T_manual"] = np.linalg.inv(frame["T_ego_camera"]) @ transform @ frame["T_ego_camera"]
                frames.append(frame)
        frames_by_split.append(frames)
    train_frames, holdout_frames = frames_by_split
    lower, upper = initial.copy(), initial.copy()
    step = np.deg2rad(args.max_roll_step_deg)
    lower[0], upper[0] = initial[0] - step, initial[0] + step
    initial_train, initial_holdout = score(train_frames, initial), score(holdout_frames, initial)
    result = least_squares(lambda x: residuals(train_frames, np.r_[x[0], initial[1:]]), [initial[0]],
                           bounds=([lower[0]], [upper[0]]), loss="huber", f_scale=2.5, max_nfev=80)
    refined = initial.copy()
    refined[0] = result.x[0]
    final_train, final_holdout = score(train_frames, refined), score(holdout_frames, refined)
    expected = (np.asarray(prior["expected_body_correction"], dtype=np.float64) if "expected_body_correction" in prior
                else se3_vector(np.linalg.inv(transform)))
    boundary_hit = bool(np.isclose(refined[0], lower[0], atol=1e-5) or np.isclose(refined[0], upper[0], atol=1e-5))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "train_initial.jpg"), draw_overlay(train_frames[0], initial, "Train: vertical contours, initial"))
    cv2.imwrite(str(out / "train_refined.jpg"), draw_overlay(train_frames[0], refined, "Train: vertical contours, roll refined"))
    cv2.imwrite(str(out / "holdout_initial.jpg"), draw_overlay(holdout_frames[0], initial, "Holdout: vertical contours, initial"))
    cv2.imwrite(str(out / "holdout_refined.jpg"), draw_overlay(holdout_frames[0], refined, "Holdout: vertical contours, roll refined"))
    report = {"mode": "roll-only: stacked LiDAR vertical depth edges + cached TEED vertical edges",
              "scene": scene["name"], "sweeps": args.sweeps, "train_offsets": args.train_offsets,
              "holdout_offsets": args.holdout_offsets, "initial_body_correction": initial.tolist(),
              "vertical_column_gate": {"cell_size_m": args.column_cell_size,
                                       "min_height_m": args.column_min_height,
                                       "min_points": args.column_min_points},
              "refined_body_correction": refined.tolist(), "expected_body_correction": expected.tolist(),
              "initial_error_rpy_deg": np.rad2deg(initial[:3] - expected[:3]).tolist(),
              "refined_error_rpy_deg": np.rad2deg(refined[:3] - expected[:3]).tolist(),
              "train_score_initial": initial_train, "train_score_refined": final_train,
              "holdout_score_initial": initial_holdout, "holdout_score_refined": final_holdout,
              "optimizer": {"success": bool(result.success), "nfev": int(result.nfev), "boundary_hit": boundary_hit},
              "cache": {"dir": str(args.cache_dir), "hits": cache.hits, "inferred": cache.misses},
              "publish_roll": bool(result.success and not boundary_hit and
                                   final_holdout["median_px"] < initial_holdout["median_px"] and
                                   final_holdout["p90_px"] < initial_holdout["p90_px"]),
              "views": [{"camera": f["camera"], "candidates": f["candidate_count"]} for f in train_frames + holdout_frames]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
