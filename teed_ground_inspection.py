"""Inspect TEED image edges that agree with stacked LiDAR ground returns."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from nuscenes_edge_demo import global_to_sensor, project
from stacked_lidar_demo import dynamic_mask, estimate_ground_mask, sensor_to_global
from teed_model import TEED


TEED_BGR_MEAN = np.array([104.007, 116.669, 122.679], dtype=np.float32)


def caption(image, text):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (min(1150, canvas.shape[1]), 42), (0, 0, 0), -1)
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_points(image, uv, mask, color, radius):
    canvas = image.copy()
    for point in uv[mask].astype(int):
        cv2.circle(canvas, tuple(point), radius, color, -1, cv2.LINE_AA)
    return canvas


def teed_probability(image, model, device):
    height, width = image.shape[:2]
    resized_width = (width + 7) // 8 * 8
    resized_height = (height + 7) // 8 * 8
    resized = cv2.resize(image, (resized_width, resized_height)) if (resized_width, resized_height) != (width, height) else image
    tensor = torch.from_numpy((resized.astype(np.float32) - TEED_BGR_MEAN).transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor)[-1])[0, 0].float().cpu().numpy()
    return cv2.resize(probability, (width, height), interpolation=cv2.INTER_CUBIC)


def collect_sweeps(nusc, reference_sample, count):
    keyframes, token = [], reference_sample["token"]
    needed = max(2, int(np.ceil(count / 10)) + 2)
    while token and len(keyframes) < needed:
        sample = nusc.get("sample", token)
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        keyframes.append((lidar["timestamp"], sample))
        token = sample["prev"]
    scans, token = [], reference_sample["data"]["LIDAR_TOP"]
    while token and len(scans) < count:
        lidar = nusc.get("sample_data", token)
        annotation_sample = min(keyframes, key=lambda item: abs(item[0] - lidar["timestamp"]))[1]
        scans.append((lidar, annotation_sample))
        token = lidar["prev"]
    return list(reversed(scans))


def intensity_bev_contour(points_ego, high_intensity, resolution=0.10,
                          x_min=0.0, x_max=70.0, y_min=-25.0, y_max=25.0):
    """Select high-reflectance returns at the boundary of BEV marking regions."""
    x0, y0, width_m, height_m = x_min, y_min, x_max - x_min, y_max - y_min
    width, height = int(width_m / resolution), int(height_m / resolution)
    gx = ((points_ego[:, 0] - x0) / resolution).astype(np.int32)
    gy = ((points_ego[:, 1] - y0) / resolution).astype(np.int32)
    in_grid = high_intensity & (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    if not np.any(in_grid):
        return np.zeros(len(points_ego), dtype=bool)

    occupancy = np.zeros((height, width), dtype=np.uint8)
    occupancy[gy[in_grid], gx[in_grid]] = 255
    # Join repeated returns from a painted region before retaining only its rim.
    closed = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    markings = np.zeros_like(closed)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 8:
            markings[labels == label] = 255
    rim = cv2.morphologyEx(markings, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    selected = np.zeros(len(points_ego), dtype=bool)
    selected[in_grid] = rim[gy[in_grid], gx[in_grid]] > 0
    return selected


def main():
    parser = argparse.ArgumentParser(description="Fuse TEED image edges with stacked LiDAR ground intensity")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--out", default="/workspace/results/teed_ground")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--reference-offset", type=int, default=9, help="Keyframe offset from the scene start")
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--weights", default="/workspace/models/teed_biped_epoch5.pth")
    parser.add_argument("--intensity-percentile", type=float, default=90.0)
    parser.add_argument("--teed-threshold", type=float, default=None, help="Absolute probability threshold; overrides --teed-percentile")
    parser.add_argument("--teed-percentile", type=float, default=95.0, help="Adaptive TEED threshold percentile")
    parser.add_argument("--match-radius-px", type=int, default=4)
    parser.add_argument("--bev-resolution", type=float, default=0.10,
                        help="metres per cell for high-intensity ground contours")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.sweeps < 1:
        parser.error("--sweeps must be positive")
    if not 0 < args.intensity_percentile < 100:
        parser.error("--intensity-percentile must be between 0 and 100")
    if not 0 < args.teed_percentile < 100:
        parser.error("--teed-percentile must be between 0 and 100")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    token = scene["first_sample_token"]
    for _ in range(args.reference_offset):
        sample = nusc.get("sample", token)
        if not sample["next"]:
            parser.error("--reference-offset exceeds this scene")
        token = sample["next"]
    reference_sample = nusc.get("sample", token)
    camera_sd = nusc.get("sample_data", reference_sample["data"][args.camera])
    camera = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
    reference_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])
    image = cv2.imread(str(Path(args.dataroot) / camera_sd["filename"]))
    if image is None:
        raise FileNotFoundError(camera_sd["filename"])
    K = np.asarray(camera["camera_intrinsic"], dtype=np.float64)

    camera_points, ego_points, intensities, removed = [], [], [], []
    for lidar_sd, annotation_sample in collect_sweeps(nusc, reference_sample, args.sweeps):
        lidar = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        raw = np.fromfile(Path(args.dataroot) / lidar_sd["filename"], dtype=np.float32).reshape(-1, 5)
        global_points = sensor_to_global(raw[:, :3], lidar, pose)
        movable = dynamic_mask(nusc, annotation_sample, global_points)
        global_points = global_points[~movable]
        camera_points.append(global_to_sensor(global_points, camera, reference_pose))
        ego_points.append((global_points - np.asarray(reference_pose["translation"])) @ Quaternion(reference_pose["rotation"]).rotation_matrix)
        intensities.append(raw[~movable, 3])
        removed.append(int(movable.sum()))
    points = np.concatenate(camera_points)
    ego_points = np.concatenate(ego_points)
    intensity = np.concatenate(intensities)
    uv, visible = project(points, np.eye(4), K, image.shape)
    ground = estimate_ground_mask(ego_points) & visible
    intensity_cutoff = float(np.percentile(intensity[ground], args.intensity_percentile))
    high_intensity = ground & (intensity >= intensity_cutoff)
    lidar_contour = intensity_bev_contour(ego_points, high_intensity, args.bev_resolution)

    requested_device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu")
    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"TEED weight is missing: {weights}")
    model = TEED().to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    probability = np.clip(teed_probability(image, model, device), 0.0, 1.0)
    teed_threshold = (args.teed_threshold if args.teed_threshold is not None
                      else float(np.percentile(probability, args.teed_percentile)))
    edge_mask = probability >= teed_threshold
    radius = max(1, args.match_radius_px)
    local_max = cv2.dilate(probability, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8))
    point_probability = np.zeros(len(points), dtype=np.float32)
    projected = np.flatnonzero(visible)
    pixel = np.rint(uv[projected]).astype(int)
    pixel[:, 0] = np.clip(pixel[:, 0], 0, image.shape[1] - 1)
    pixel[:, 1] = np.clip(pixel[:, 1], 0, image.shape[0] - 1)
    point_probability[projected] = local_max[pixel[:, 1], pixel[:, 0]]
    accepted = high_intensity & (point_probability >= teed_threshold)

    probability_view = cv2.applyColorMap(np.uint8(np.clip(probability, 0, 1) * 255), cv2.COLORMAP_TURBO)
    probability_view = cv2.addWeighted(image, 0.45, probability_view, 0.55, 0)
    candidates = draw_points(image, uv, high_intensity, (255, 255, 0), 3)
    contours = draw_points(image, uv, lidar_contour, (255, 0, 255), 3)
    fused = draw_points(image, uv, high_intensity, (0, 165, 255), 2)
    fused = draw_points(fused, uv, accepted, (0, 255, 0), 3)
    panels = [
        caption(probability_view, "1 TEED edge probability"),
        caption(candidates, "2 High-intensity stacked ground candidates"),
        caption(contours, "3 BEV intensity contours"),
        caption(fused, "4 Fusion: orange=all candidates, green=TEED-supported"),
    ]
    height, width = image.shape[:2]
    sheet = np.hstack([cv2.resize(panel, (width // 2, height // 2)) for panel in panels])
    cv2.imwrite(str(out / f"{args.camera}_teed_probability.jpg"), panels[0])
    cv2.imwrite(str(out / f"{args.camera}_intensity_candidates.jpg"), panels[1])
    cv2.imwrite(str(out / f"{args.camera}_intensity_contours.jpg"), panels[2])
    cv2.imwrite(str(out / f"{args.camera}_fusion.jpg"), panels[3])
    cv2.imwrite(str(out / f"{args.camera}_comparison.jpg"), sheet)
    report = {
        "scene": scene["name"], "camera": args.camera, "reference_sample": reference_sample["token"],
        "sweeps": args.sweeps, "device": str(device), "dynamic_removed_per_sweep": removed,
        "static_points": int(len(points)), "projected_points": int(visible.sum()),
        "ground_points": int(ground.sum()), "intensity_percentile": args.intensity_percentile,
        "intensity_cutoff": intensity_cutoff, "high_intensity_candidates": int(high_intensity.sum()),
        "bev_resolution_m": args.bev_resolution, "bev_intensity_contour_candidates": int(lidar_contour.sum()),
        "teed_threshold": teed_threshold,
        "teed_threshold_source": "absolute" if args.teed_threshold is not None else f"p{args.teed_percentile:g}",
        "match_radius_px": radius,
        "teed_supported_candidates": int(accepted.sum()),
        "teed_support_ratio": float(accepted.sum() / max(1, high_intensity.sum())),
        "candidate_teed_probability_median": float(np.median(point_probability[high_intensity])),
        "candidate_teed_probability_p90": float(np.percentile(point_probability[high_intensity], 90)),
        "teed_edge_pixels": int(edge_mask.sum()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
