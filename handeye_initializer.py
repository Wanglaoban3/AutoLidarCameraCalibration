"""LiDAR-odometry / GNSS-IMU hand-eye initializer for large extrinsic errors.

This evaluator uses nuScenes poses as an oracle LiDAR-odometry stream. In a
vehicle replace lidar_relative_poses() with KISS-ICP, FAST-LIO or the production
LiDAR odometry output; GNSS/IMU remains the body-pose input.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.optimize import least_squares

from nuscenes_edge_demo import se3, se3_vector


def matrix(record):
    T = np.eye(4)
    T[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    T[:3, 3] = record["translation"]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--out", default="/workspace/results/handeye")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--noise-rpy-deg", type=float, nargs=3, default=[3.0, -3.0, 4.0])
    ap.add_argument("--noise-translation-m", type=float, nargs=3, default=[0.06, -0.04, 0.08])
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

    body_poses, lidar_poses = [], []
    X_true = None
    for sample in samples:
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        body_pose = matrix(nusc.get("ego_pose", sd["ego_pose_token"]))
        X = matrix(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]))
        body_poses.append(body_pose)
        lidar_poses.append(body_pose @ X)
        X_true = X

    noise = se3(np.r_[np.deg2rad(args.noise_rpy_deg), args.noise_translation_m])
    X_manual = noise @ X_true
    pairs = []
    for gap in (1, 3, 6):
        for i in range(len(samples) - gap):
            B = np.linalg.inv(body_poses[i]) @ body_poses[i + gap]
            # This is the interface a real LiDAR odometry backend supplies.
            A = np.linalg.inv(lidar_poses[i]) @ lidar_poses[i + gap]
            pairs.append((A, B))

    def residual(delta):
        X = se3(delta) @ X_manual
        values = []
        for A, B in pairs:
            # B * X = X * A, expressed as a six-dimensional SE(3) residual.
            values.append(se3_vector(np.linalg.inv(B @ X) @ (X @ A)))
        return np.concatenate(values)

    result = least_squares(
        residual, np.zeros(6), loss="huber", f_scale=0.01,
        bounds=(-np.array([0.20, 0.20, 0.20, 0.30, 0.30, 0.30]),
                np.array([0.20, 0.20, 0.20, 0.30, 0.30, 0.30])), max_nfev=300,
    )
    expected = se3_vector(X_true @ np.linalg.inv(X_manual))
    report = {
        "mode": "oracle evaluation: replace nuScenes LiDAR poses with LiDAR odometry in production",
        "motion_pairs": len(pairs), "manual_body_noise": se3_vector(noise).tolist(),
        "expected_body_correction": expected.tolist(), "estimated_body_correction": result.x.tolist(),
        "rmse_se3": float(np.sqrt(np.mean(residual(result.x) ** 2))),
        "success": bool(result.success),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
