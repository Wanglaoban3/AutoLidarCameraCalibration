"""Disk cache for TEED probability maps keyed by nuScenes sample-data token."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from nuscenes_edge_demo import CAMERAS
from teed_ground_inspection import teed_probability
from teed_model import TEED


class TEEDCache:
    def __init__(self, directory, model, device, force=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model, self.device, self.force = model, device, force
        self.hits, self.misses = 0, 0

    def probability(self, image, sample_data_token):
        path = self.directory / f"{sample_data_token}.npy"
        if path.is_file() and not self.force:
            value = np.load(path)
            if value.shape == image.shape[:2] and np.isfinite(value).all():
                self.hits += 1
                return value.astype(np.float32, copy=False)
        value = np.clip(teed_probability(image, self.model, self.device), 0.0, 1.0).astype(np.float32)
        np.save(path, value)
        self.misses += 1
        return value


def scene_samples(nusc, scene):
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        yield sample
        token = sample["next"]


def main():
    parser = argparse.ArgumentParser(description="Precompute reusable TEED maps for one nuScenes scene")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--weights", default="/workspace/models/teed_biped_epoch5.pth")
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    model = TEED().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True), strict=True)
    model.eval()
    cache = TEEDCache(args.cache_dir, model, device, force=args.force)
    scene = nusc.scene[args.scene]
    count = 0
    for sample in scene_samples(nusc, scene):
        for camera_name in args.cameras:
            sd = nusc.get("sample_data", sample["data"][camera_name])
            image = cv2.imread(str(Path(args.dataroot) / sd["filename"]))
            if image is None:
                raise FileNotFoundError(sd["filename"])
            cache.probability(image, sd["token"])
            count += 1
    report = {"scene": scene["name"], "images": count, "cache_dir": str(args.cache_dir),
              "cache_hits": cache.hits, "inferred": cache.misses, "device": str(device)}
    (Path(args.cache_dir) / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
