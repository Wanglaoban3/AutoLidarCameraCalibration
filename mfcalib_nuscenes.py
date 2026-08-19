#!/usr/bin/env python3
"""Run the ROS-free MFCalib implementation on a nuScenes-mini segment.

nuScenes calibrated_sensor/ego_pose records are used only to create a
motion-compensated evaluation segment and to score the final estimate. The
reported input mode is therefore an oracle-trajectory benchmark, not a
production odometry pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

import mfcalib_python


def transform_record(record):
    T = np.eye(4)
    T[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    T[:3, 3] = np.asarray(record["translation"], dtype=float)
    return T


def se3_vector(T):
    """Match mfcalib_python's Z-Y-X extrinsic parameter convention."""
    from scipy.spatial.transform import Rotation
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_euler("ZYX"), T[:3, 3]]


def load_segment(nusc, scene, start, frames):
    samples = []
    token = scene["first_sample_token"]
    while token and len(samples) < start + frames:
        sample = nusc.get("sample", token)
        if len(samples) >= start:
            samples.append(sample)
        token = sample["next"]
    if len(samples) != frames:
        raise ValueError(f"scene has only {len(samples)} samples from start={start}, requested {frames}")
    return samples


def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    samples = load_segment(nusc, scene, args.start, args.frames)
    ref = samples[-1]
    camera_sd = nusc.get("sample_data", ref["data"][args.camera])
    camera_cs = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
    lidar_sd_ref = nusc.get("sample_data", ref["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd_ref["calibrated_sensor_token"])
    camera_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])
    lidar_pose_ref = nusc.get("ego_pose", lidar_sd_ref["ego_pose_token"])
    T_ego_cam = transform_record(camera_cs)
    T_ego_lidar = transform_record(lidar_cs)
    T_global_cam = transform_record(camera_pose)
    T_global_ego_ref = transform_record(lidar_pose_ref)
    # The camera and LiDAR key-data timestamps differ in nuScenes.  The true
    # transform must therefore include their two ego poses.
    T_true = np.linalg.inv(T_global_cam @ T_ego_cam) @ (T_global_ego_ref @ T_ego_lidar)

    if args.noise_mode == "uniform":
        rng = np.random.default_rng(args.seed)
        rpy_limit = np.abs(np.asarray(args.noise_rpy_deg, dtype=float))
        translation_limit = np.abs(np.asarray(args.noise_translation_m, dtype=float))
        noise_rpy = np.deg2rad(rng.uniform(-rpy_limit, rpy_limit))
        noise_t = rng.uniform(-translation_limit, translation_limit)
    else:
        noise_rpy = np.deg2rad(np.asarray(args.noise_rpy_deg, dtype=float))
        noise_t = np.asarray(args.noise_translation_m, dtype=float)
    noise = mfcalib_python.se3(np.r_[noise_rpy, noise_t])
    T_noisy = np.linalg.inv(T_global_cam @ T_ego_cam) @ (T_global_ego_ref @ noise @ T_ego_lidar)

    # Build a static segment in the reference LiDAR frame. This uses the
    # nuScenes trajectory as an oracle motion source for reproducible scoring.
    T_ref_global_lidar = T_global_ego_ref @ T_ego_lidar
    stacked = []
    for sample in samples:
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        T_global_lidar = transform_record(pose) @ T_ego_lidar
        T_ref_lidar = np.linalg.inv(T_ref_global_lidar) @ T_global_lidar
        raw = np.fromfile(Path(args.dataroot) / sd["filename"], dtype=np.float32).reshape(-1, 5)
        points = raw[:, :3]
        intensity = raw[:, 3:4]
        homogeneous = np.c_[points, np.ones(len(points))]
        moved = (T_ref_lidar @ homogeneous.T).T[:, :3]
        stacked.append(np.c_[moved, intensity])
    cloud = np.concatenate(stacked, axis=0)
    if len(cloud) > args.max_points:
        ids = np.linspace(0, len(cloud) - 1, args.max_points).astype(int)
        cloud = cloud[ids]
    points_path = out / "segment_points.npy"
    np.save(points_path, cloud)

    image_path = Path(args.dataroot) / camera_sd["filename"]
    config_path = out / "mfcalib_config.yaml"
    config = {
        "camera": {
            "camera_matrix": np.asarray(camera_cs["camera_intrinsic"], dtype=float).reshape(-1).tolist(),
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "extrinsic": T_noisy.tolist(),
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    run_args = argparse.Namespace(
        image=image_path, points=points_path, config=config_path, camera_config=None,
        out=out / "mfcalib", min_component=args.min_component,
        lidar_voxel=args.lidar_voxel, depth_jump=args.depth_jump,
        max_lidar_edges=args.max_lidar_edges, angular_resolution=args.angular_resolution,
        voxel_size=args.voxel_size, ransac_threshold=args.ransac_threshold,
        plane_min_points=args.plane_min_points, max_planes=args.max_planes,
        match_threshold=args.match_threshold, thresholds=args.thresholds,
        max_nfev=args.max_nfev, max_rotation_update_deg=args.max_rotation_update_deg,
        max_translation_update_m=args.max_translation_update_m,
        min_stage_improvement_px=args.min_stage_improvement_px,
    )
    mfcalib_python.run(run_args)
    report = json.loads((out / "mfcalib" / "report.json").read_text())
    T_est = np.asarray(report["extrinsic_lidar_to_camera"], dtype=float)
    initial_error = se3_vector(np.linalg.inv(T_true) @ T_noisy)
    final_error = se3_vector(np.linalg.inv(T_true) @ T_est)
    report.update({
        "mode": "nuScenes-mini segment + oracle trajectory motion compensation",
        "scene": scene["name"], "scene_index": args.scene, "start": args.start,
        "frames": args.frames, "camera": args.camera, "reference_sample": ref["token"],
        "noise_seed": args.seed, "noise_rpy_deg": args.noise_rpy_deg,
        "noise_translation_m": args.noise_translation_m,
        "noise_mode": args.noise_mode, "sampled_noise_se3": np.r_[noise_rpy, noise_t].tolist(),
        "initial_error_se3": initial_error.tolist(),
        "final_error_se3": final_error.tolist(),
        "initial_error_rpy_deg": np.rad2deg(initial_error[:3]).tolist(),
        "final_error_rpy_deg": np.rad2deg(final_error[:3]).tolist(),
        "ground_truth_extrinsic_lidar_to_camera": T_true.tolist(),
        "noisy_initial_extrinsic_lidar_to_camera": T_noisy.tolist(),
        "trajectory_source": "nuScenes ego_pose oracle; replace with LiDAR odometry in vehicle",
    })
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True); p.add_argument("--scene", type=int, default=0)
    p.add_argument("--start", type=int, default=0); p.add_argument("--frames", type=int, default=8)
    p.add_argument("--camera", default="CAM_FRONT"); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--noise-mode", choices=("fixed", "uniform"), default="uniform")
    p.add_argument("--noise-rpy-deg", type=float, nargs=3, default=[3.0, -3.0, 4.0])
    p.add_argument("--noise-translation-m", type=float, nargs=3, default=[0.06, -0.04, 0.08])
    p.add_argument("--max-points", type=int, default=30000)
    p.add_argument("--min-component", type=int, default=40); p.add_argument("--lidar-voxel", type=float, default=0.08)
    p.add_argument("--depth-jump", type=float, default=0.35); p.add_argument("--max-lidar-edges", type=int, default=5000)
    p.add_argument("--angular-resolution", type=float, default=0.003); p.add_argument("--voxel-size", type=float, default=1.0)
    p.add_argument("--ransac-threshold", type=float, default=0.02); p.add_argument("--plane-min-points", type=int, default=30)
    p.add_argument("--max-planes", type=int, default=8); p.add_argument("--match-threshold", type=float, default=20.0)
    p.add_argument("--thresholds", type=float, nargs="+", default=[20, 12, 8, 5, 3]); p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument("--max-rotation-update-deg", type=float, default=5.0); p.add_argument("--max-translation-update-m", type=float, default=0.30); p.add_argument("--min-stage-improvement-px", type=float, default=0.05)
    main(p.parse_args())
