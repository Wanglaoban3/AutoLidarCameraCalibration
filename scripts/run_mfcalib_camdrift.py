# -*- coding: utf-8 -*-
"""Camera-drift mirror of the reference MFCalib benchmark: the LIDAR
extrinsic stays factory-correct, the CAMERA mounting is polluted, and we
ask whether the reference repo's mfcalib_python recovers the lidar->camera
extrinsic. Same segment building / scoring as mfcalib_nuscenes.py (via
the compat devkit shim); only the noise application is mirrored to the
camera side: T_ec_cam' = noise @ T_ec_cam.

Usage:
  python scripts/run_mfcalib_camdrift.py --scene 0 --seed 7
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = Path(r'H:\projects\AutoLidarCameraCalibration')
sys.path.insert(0, str(ROOT / 'compat'))
sys.path.insert(0, str(REF))

from nuscenes.nuscenes import NuScenes          # noqa: E402  (compat shim)
from pyquaternion import Quaternion             # noqa: E402  (compat shim)
import mfcalib_python                           # noqa: E402


def transform_record(record):
    T = np.eye(4)
    T[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    T[:3, 3] = np.asarray(record['translation'], dtype=float)
    return T


def se3_vector(T):
    from scipy.spatial.transform import Rotation
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_euler('ZYX'), T[:3, 3]]


def load_segment(nusc, scene, start, frames):
    samples = []
    token = scene['first_sample_token']
    while token and len(samples) < start + frames:
        sample = nusc.get('sample', token)
        if len(samples) >= start:
            samples.append(sample)
        token = sample['next']
    if len(samples) != frames:
        raise ValueError('not enough samples')
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default=r'H:\datasets\nuscenes-mini')
    ap.add_argument('--scene', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--camera', default='CAM_FRONT')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--noise-rpy-deg', type=float, nargs=3,
                    default=[1.0, 1.0, 1.0])
    ap.add_argument('--noise-translation-m', type=float, nargs=3,
                    default=[0.05, 0.05, 0.05])
    ap.add_argument('--max-points', type=int, default=30000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = Path(args.out or (ROOT / 'outputs' / 'mfcalib'
                            / f'scene{args.scene}_camdrift_seed{args.seed}'))
    out.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot,
                    verbose=False)
    scene = nusc.scene[args.scene]
    samples = load_segment(nusc, scene, args.start, args.frames)
    ref = samples[-1]
    camera_sd = nusc.get('sample_data', ref['data'][args.camera])
    camera_cs = nusc.get('calibrated_sensor',
                         camera_sd['calibrated_sensor_token'])
    lidar_sd_ref = nusc.get('sample_data', ref['data']['LIDAR_TOP'])
    lidar_cs = nusc.get('calibrated_sensor',
                        lidar_sd_ref['calibrated_sensor_token'])
    camera_pose = nusc.get('ego_pose', camera_sd['ego_pose_token'])
    lidar_pose_ref = nusc.get('ego_pose', lidar_sd_ref['ego_pose_token'])
    T_ego_cam = transform_record(camera_cs)
    T_ego_lidar = transform_record(lidar_cs)
    T_global_cam = transform_record(camera_pose)
    T_global_ego_ref = transform_record(lidar_pose_ref)
    T_true = (np.linalg.inv(T_global_cam @ T_ego_cam)
              @ (T_global_ego_ref @ T_ego_lidar))

    rng = np.random.default_rng(args.seed)
    rpy_limit = np.abs(np.asarray(args.noise_rpy_deg, float))
    t_limit = np.abs(np.asarray(args.noise_translation_m, float))
    noise_rpy = np.deg2rad(rng.uniform(-rpy_limit, rpy_limit))
    noise_t = rng.uniform(-t_limit, t_limit)
    noise = mfcalib_python.se3(np.r_[noise_rpy, noise_t])
    # MIRRORED injection: the CAMERA mounting drifted, the lidar record is
    # factory-correct
    T_noisy = (np.linalg.inv(T_global_cam @ noise @ T_ego_cam)
               @ (T_global_ego_ref @ T_ego_lidar))

    T_ref_global_lidar = T_global_ego_ref @ T_ego_lidar
    stacked = []
    for sample in samples:
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        T_ref_lidar = (np.linalg.inv(T_ref_global_lidar)
                       @ (transform_record(pose) @ T_ego_lidar))
        raw = np.fromfile(Path(args.dataroot) / sd['filename'],
                          dtype=np.float32).reshape(-1, 5)
        hom = np.c_[raw[:, :3], np.ones(len(raw))]
        moved = (T_ref_lidar @ hom.T).T[:, :3]
        stacked.append(np.c_[moved, raw[:, 3:4]])
    cloud = np.concatenate(stacked, axis=0)
    if len(cloud) > args.max_points:
        ids = np.linspace(0, len(cloud) - 1, args.max_points).astype(int)
        cloud = cloud[ids]
    points_path = out / 'segment_points.npy'
    np.save(points_path, cloud)

    import yaml
    config_path = out / 'mfcalib_config.yaml'
    config = {
        'camera': {
            'camera_matrix': np.asarray(camera_cs['camera_intrinsic'],
                                        float).reshape(-1).tolist(),
            'dist_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        'extrinsic': T_noisy.tolist(),
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    image_path = Path(args.dataroot) / camera_sd['filename']
    run_args = argparse.Namespace(
        image=image_path, points=points_path, config=config_path,
        camera_config=None, out=out / 'mfcalib', min_component=40,
        lidar_voxel=0.08, depth_jump=0.35, max_lidar_edges=5000,
        angular_resolution=0.003, voxel_size=1.0, ransac_threshold=0.02,
        plane_min_points=30, max_planes=8, match_threshold=20.0,
        thresholds=[20, 12, 8, 5, 3], max_nfev=80,
        max_rotation_update_deg=5.0, max_translation_update_m=0.30,
        min_stage_improvement_px=0.05)
    mfcalib_python.run(run_args)
    report = json.loads((out / 'mfcalib' / 'report.json').read_text())
    T_est = np.asarray(report['extrinsic_lidar_to_camera'], float)
    initial_error = se3_vector(np.linalg.inv(T_true) @ T_noisy)
    final_error = se3_vector(np.linalg.inv(T_true) @ T_est)
    report.update({
        'mode': 'CAMERA-side drift (lidar factory-correct)',
        'scene': scene['name'], 'scene_index': args.scene,
        'camera': args.camera, 'noise_seed': args.seed,
        'initial_error_rpy_deg': np.rad2deg(initial_error[:3]).tolist(),
        'final_error_rpy_deg': np.rad2deg(final_error[:3]).tolist(),
        'accepted_stages': report['accepted_stages'],
    })
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        'scene', 'camera', 'initial_error_rpy_deg', 'final_error_rpy_deg',
        'accepted_stages', 'median_pixel_error_before',
        'median_pixel_error_after')}, indent=1))


if __name__ == '__main__':
    main()
