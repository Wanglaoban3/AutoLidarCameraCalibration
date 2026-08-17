"""Simulation of large-angle calibration using real LiDAR scans + ICP + GNSS/IMU.

nuScenes ego poses emulate GNSS/IMU. LiDAR relative motion is estimated directly
from the raw .bin scans by ICP; dataset LiDAR poses are used only for evaluation.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from nuscenes_edge_demo import se3, se3_vector


def matrix(record):
    T = np.eye(4)
    T[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    T[:3, 3] = record["translation"]
    return T


def transform(points, T):
    return (T[:3, :3] @ points.T + T[:3, 3:4]).T


def downsample(points, voxel=0.45, limit=7000):
    keep = (np.linalg.norm(points[:, :2], axis=1) < 55.0) & (points[:, 2] > -3.0) & (points[:, 2] < 3.0)
    points = points[keep]
    keys = np.floor(points / voxel).astype(np.int32)
    _, ids = np.unique(keys, axis=0, return_index=True)
    points = points[np.sort(ids)]
    return points[::max(1, len(points) // limit)]


def rigid_fit(source, target):
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, target_center - R @ source_center
    return T


def icp(source, target, initial, iterations=20):
    tree, T = cKDTree(target), initial.copy()
    for _ in range(iterations):
        moved = transform(source, T)
        distance, ids = tree.query(moved, k=1, workers=-1)
        inlier = distance < 1.2
        if inlier.sum() < 80:
            break
        update = rigid_fit(moved[inlier], target[ids[inlier]])
        T = update @ T
        if np.linalg.norm(se3_vector(update)) < 1e-4:
            break
    moved = transform(source, T)
    distance, _ = tree.query(moved, k=1, workers=-1)
    return T, float(np.sqrt(np.mean(np.minimum(distance, 1.2) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--out", default="/workspace/results/icp_handeye")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--frames", type=int, default=12)
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

    body, scans, X_true = [], [], None
    for sample in samples:
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        body.append(matrix(nusc.get("ego_pose", sd["ego_pose_token"])))
        X_true = matrix(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]))
        raw = np.fromfile(Path(args.dataroot) / sd["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]
        scans.append(downsample(raw))

    noise = se3(np.r_[np.deg2rad(args.noise_rpy_deg), args.noise_translation_m])
    X_manual = noise @ X_true
    step_motions, icp_rmse = [], []
    # Register scan j into scan i. GNSS/IMU plus manual extrinsic provides the
    # initial guess; ICP itself only consumes raw point clouds.
    for i in range(len(scans) - 1):
        B = np.linalg.inv(body[i]) @ body[i + 1]
        A_seed = np.linalg.inv(X_manual) @ B @ X_manual
        A, rmse = icp(scans[i + 1], scans[i], A_seed)
        step_motions.append(A)
        icp_rmse.append(rmse)

    # Compose locally registered steps into longer baselines. This increases
    # observability of yaw and translation while preserving raw-scan ICP as the
    # only LiDAR motion source.
    pairs = []
    for gap in (1, 3, 6):
        for i in range(len(scans) - gap):
            A = np.eye(4)
            for k in range(i, i + gap):
                A = A @ step_motions[k]
            B = np.linalg.inv(body[i]) @ body[i + gap]
            pairs.append((A, B))

    def residual(delta):
        X = se3(delta) @ X_manual
        return np.concatenate([se3_vector(np.linalg.inv(B @ X) @ (X @ A)) for A, B in pairs])

    result = least_squares(
        residual, np.zeros(6), loss="huber", f_scale=0.03,
        bounds=(-np.array([0.20, 0.20, 0.20, 0.30, 0.30, 0.30]),
                np.array([0.20, 0.20, 0.20, 0.30, 0.30, 0.30])), max_nfev=300,
    )
    expected = se3_vector(X_true @ np.linalg.inv(X_manual))
    report = {
        "mode": "raw nuScenes LiDAR ICP + nuScenes ego poses emulating GNSS/IMU",
        "frames": len(scans), "motion_pairs": len(pairs), "motion_gaps": [1, 3, 6], "manual_body_noise": se3_vector(noise).tolist(),
        "expected_body_correction": expected.tolist(), "estimated_body_correction": result.x.tolist(),
        "handeye_rmse_se3": float(np.sqrt(np.mean(residual(result.x) ** 2))),
        "icp_rmse_median_m": float(np.median(icp_rmse)), "icp_rmse_per_pair_m": icp_rmse,
        "success": bool(result.success),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
