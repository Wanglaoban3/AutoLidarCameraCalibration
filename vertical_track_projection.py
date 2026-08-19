"""Project temporally stable 3D vertical tracks onto a nuScenes camera image."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from nuscenes_edge_demo import global_to_sensor, project


def sample_at_offset(nusc, scene, offset):
    token = scene["first_sample_token"]
    for _ in range(offset):
        sample = nusc.get("sample", token)
        if not sample["next"]:
            raise ValueError("reference offset exceeds scene")
        token = sample["next"]
    return nusc.get("sample", token)


def main():
    parser = argparse.ArgumentParser(description="Project stable vertical LiDAR tracks")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--tracks-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--reference-offset", type=int, default=9)
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--min-hits", type=int, default=8)
    parser.add_argument("--half-height-m", type=float, default=1.5)
    args = parser.parse_args()
    tracks = json.loads(Path(args.tracks_json).read_text())["tracks"]
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    sample = sample_at_offset(nusc, scene, args.reference_offset)
    sd = nusc.get("sample_data", sample["data"][args.camera])
    camera = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    image = cv2.imread(str(Path(args.dataroot) / sd["filename"]))
    if image is None:
        raise FileNotFoundError(sd["filename"])
    K = np.asarray(camera["camera_intrinsic"], dtype=np.float64)
    canvas, shown = image.copy(), 0
    for track in tracks:
        if track["hits"] < args.min_hits:
            continue
        xy, z = np.asarray(track["xy"]), float(track["z"])
        global_line = np.array([[xy[0], xy[1], z - args.half_height_m], [xy[0], xy[1], z + args.half_height_m]])
        points = global_to_sensor(global_line, camera, pose)
        uv, valid = project(points, np.eye(4), K, image.shape)
        if valid.all():
            a, b = uv.astype(int)
            cv2.line(canvas, tuple(a), tuple(b), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(track["hits"]), tuple(a), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            shown += 1
    cv2.rectangle(canvas, (0, 0), (min(1000, canvas.shape[1]), 42), (0, 0, 0), -1)
    cv2.putText(canvas, f"Stable vertical tracks: hits >= {args.min_hits}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(json.dumps({"shown_tracks": shown, "output": args.out}, indent=2))


if __name__ == "__main__":
    main()
