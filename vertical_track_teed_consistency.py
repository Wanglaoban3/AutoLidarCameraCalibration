"""Score stable 3D vertical tracks against cached TEED vertical edges."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from nuscenes_edge_demo import global_to_sensor, project
from teed_edge_cache import TEEDCache
from teed_model import TEED


def sample_at_offset(nusc, scene, offset):
    token = scene["first_sample_token"]
    for _ in range(offset):
        sample = nusc.get("sample", token)
        token = nusc.get("sample", token)["next"]
    return nusc.get("sample", token)


def support(image, probability, track, camera, pose, percentile, radius):
    threshold = np.percentile(probability, percentile)
    gx = cv2.Sobel(probability, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(probability, cv2.CV_32F, 0, 1, ksize=3)
    edge = (probability >= threshold) & (np.abs(gx) >= 0.75 * np.hypot(gx, gy))
    distance = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    xy, z = np.asarray(track["xy"]), float(track["z"])
    heights = np.linspace(z - 1.5, z + 1.5, 25)
    global_line = np.c_[np.full(len(heights), xy[0]), np.full(len(heights), xy[1]), heights]
    points = global_to_sensor(global_line, camera, pose)
    K = np.asarray(camera["camera_intrinsic"], dtype=np.float64)
    uv, valid = project(points, np.eye(4), K, image.shape)
    if valid.sum() < 5:
        return 0.0
    pixel = np.rint(uv[valid]).astype(int)
    pixel[:, 0] = np.clip(pixel[:, 0], 0, image.shape[1] - 1); pixel[:, 1] = np.clip(pixel[:, 1], 0, image.shape[0] - 1)
    return float(np.mean(distance[pixel[:, 1], pixel[:, 0]] <= radius))


def main():
    parser = argparse.ArgumentParser(description="Validate vertical tracks using cached TEED edges")
    parser.add_argument("--dataroot", required=True); parser.add_argument("--tracks-json", required=True)
    parser.add_argument("--cache-dir", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--scene", type=int, default=0); parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--min-image-support", type=float, default=0.35); parser.add_argument("--min-supported-windows", type=int, default=4)
    args = parser.parse_args()
    data = json.loads(Path(args.tracks_json).read_text()); tracks, offsets = data["tracks"], [item["offset"] for item in data["windows"]]
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False); scene = nusc.scene[args.scene]
    device = torch.device("cpu"); model = TEED().to(device); model.load_state_dict(torch.load("/workspace/models/teed_biped_epoch5.pth", map_location=device, weights_only=True), strict=True); model.eval()
    cache = TEEDCache(args.cache_dir, model, device)
    scores = [[] for _ in tracks]
    for offset in offsets:
        sample = sample_at_offset(nusc, scene, offset); sd = nusc.get("sample_data", sample["data"][args.camera])
        camera, pose = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]), nusc.get("ego_pose", sd["ego_pose_token"])
        image = cv2.imread(str(Path(args.dataroot) / sd["filename"])); probability = cache.probability(image, sd["token"])
        for index, track in enumerate(tracks): scores[index].append(support(image, probability, track, camera, pose, 95, 3.0))
    result = []
    for index, (track, values) in enumerate(zip(tracks, scores)):
        hits = int(np.sum(np.asarray(values) >= args.min_image_support))
        result.append({"id": index, "track": track, "image_support": values, "supported_windows": hits,
                       "accepted": hits >= args.min_supported_windows})
    report = {"scene": scene["name"], "camera": args.camera, "tracks": result,
              "accepted_tracks": int(sum(item["accepted"] for item in result)),
              "cache": {"hits": cache.hits, "inferred": cache.misses}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
