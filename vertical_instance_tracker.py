"""Track stable narrow vertical LiDAR structures across sliding sweep windows."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from stacked_lidar_demo import dynamic_mask, sensor_to_global
from teed_ground_inspection import collect_sweeps
from teed_vertical_roll_refinement import vertical_column_mask


def samples(nusc, scene, start, stride):
    token, index = scene["first_sample_token"], 0
    while token:
        sample = nusc.get("sample", token)
        if index >= start and (index - start) % stride == 0:
            yield index, sample
        token, index = sample["next"], index + 1


def observations(nusc, sample, sweeps, cell, min_height, min_points):
    lidar_ref = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose_ref = nusc.get("ego_pose", lidar_ref["ego_pose_token"])
    global_parts = []
    for sd, annotation_sample in collect_sweeps(nusc, sample, sweeps):
        calibrated = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        raw = np.fromfile(Path(nusc.dataroot) / sd["filename"], dtype=np.float32).reshape(-1, 5)
        points = sensor_to_global(raw[:, :3], calibrated, pose)
        global_parts.append(points[~dynamic_mask(nusc, annotation_sample, points)])
    global_points = np.concatenate(global_parts)
    ego = (global_points - np.asarray(pose_ref["translation"])) @ Quaternion(pose_ref["rotation"]).rotation_matrix
    valid = vertical_column_mask(ego, cell, min_height, min_points)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return []
    cells = np.floor(ego[ids, :2] / cell).astype(np.int32)
    minimum = cells.min(axis=0)
    grid = np.zeros((cells[:, 1].max() - minimum[1] + 1, cells[:, 0].max() - minimum[0] + 1), np.uint8)
    grid[cells[:, 1] - minimum[1], cells[:, 0] - minimum[0]] = 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    result = []
    for label in range(1, count):
        width, height, area = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT], stats[label, cv2.CC_STAT_AREA]
        if area > 16 or max(width, height) * cell > 2.0:
            continue
        point_labels = labels[cells[:, 1] - minimum[1], cells[:, 0] - minimum[0]]
        points = global_points[ids[point_labels == label]]
        if len(points) >= min_points:
            result.append({"xy": np.median(points[:, :2], axis=0), "z": np.median(points[:, 2]), "points": int(len(points)), "area_m2": float(area * cell * cell)})
    return result


def main():
    parser = argparse.ArgumentParser(description="Track stable vertical LiDAR instances")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--start-offset", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--association-m", type=float, default=1.0)
    parser.add_argument("--min-observations", type=int, default=4)
    parser.add_argument("--out", default="/workspace/results/vertical_instance_tracks")
    args = parser.parse_args()
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    tracks, windows = [], []
    for offset, sample in samples(nusc, scene, args.start_offset, args.stride):
        detected = observations(nusc, sample, args.sweeps, 0.5, 1.2, 5)
        used = set()
        for observation in detected:
            candidates = [(np.linalg.norm(track["xy"] - observation["xy"]), i) for i, track in enumerate(tracks) if i not in used]
            distance, index = min(candidates, default=(np.inf, None))
            if distance <= args.association_m:
                track = tracks[index]
                track["xy"] = 0.75 * track["xy"] + 0.25 * observation["xy"]
                track["z"] = 0.75 * track["z"] + 0.25 * observation["z"]
                track["hits"] += 1; track["last_offset"] = offset; used.add(index)
            else:
                tracks.append({"xy": observation["xy"], "z": observation["z"], "hits": 1, "first_offset": offset, "last_offset": offset})
        windows.append({"offset": offset, "detections": len(detected)})
    stable = [{**track, "xy": track["xy"].tolist(), "z": float(track["z"])} for track in tracks if track["hits"] >= args.min_observations]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {"scene": scene["name"], "sweeps": args.sweeps, "windows": windows,
              "raw_tracks": len(tracks), "stable_tracks": len(stable), "tracks": stable}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
