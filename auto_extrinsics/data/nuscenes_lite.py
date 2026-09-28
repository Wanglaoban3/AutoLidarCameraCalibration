# -*- coding: utf-8 -*-
"""Minimal nuScenes (v1.0-mini layout) metadata + data access.

Conventions follow the existing workspace scripts (bev_prevnext.T_of):
calibrated_sensor rotation/translation map SENSOR -> EGO, ego_pose maps
EGO -> GLOBAL. Key-frame sample_data for CAM_FRONT and LIDAR_TOP of one
log are exposed as aligned "frames" (one nuScenes sample each).

Env vars:
  AUTOEX_NUSCENES_ROOT  dataset root (default H:\datasets\nuscenes-mini)
  AUTOEX_NUSCENES_TAR   full mini tar for on-demand sweep extraction
"""
import json
import os
import tarfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DYNAMIC_CATEGORIES = (
    'human.pedestrian', 'vehicle.car', 'vehicle.truck', 'vehicle.bus',
    'vehicle.trailer', 'vehicle.motorcycle', 'vehicle.bicycle',
    'vehicle.construction', 'movable_object',
)


def quat_to_mat(q):
    """Raw JSON stores [w, x, y, z] lists (devkit dicts also accepted)."""
    if isinstance(q, dict):
        q = [q['x'], q['y'], q['z'], q['w']]
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def T_of(rec):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(rec['rotation'])
    T[:3, 3] = np.array(rec['translation'], float)
    return T


class NuScenesLite:
    def __init__(self, root=None, version='v1.0-mini'):
        self.root = Path(root or os.environ.get(
            'AUTOEX_NUSCENES_ROOT', r'H:\datasets\nuscenes-mini'))
        self.meta = self.root / version
        self.tar_path = Path(os.environ.get(
            'AUTOEX_NUSCENES_TAR', r'H:\datasets\v1.0-mini\v1.0-mini.tar'))
        self._tables = {}

    # ---- metadata tables ---------------------------------------------------
    def table(self, name):
        if name not in self._tables:
            with open(self.meta / f'{name}.json', encoding='utf-8') as f:
                self._tables[name] = json.load(f)
        return self._tables[name]

    def _by_token(self, name):
        return {r['token']: r for r in self.table(name)}

    # ---- frame listing -----------------------------------------------------
    CAM_CHANNELS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')

    def frames_of_log_multi(self, log_prefix, channels=CAM_CHANNELS,
                            lidar='LIDAR_TOP', scene_name=None):
        """Key-frame records with ALL requested channels + lidar per
        sample: dict(sample=..., CAM_*=sample_data..., LIDAR_TOP=...).
        scene_name filters to one scene's clip -- v1.0-mini has 10 scenes
        but only 8 logs (n015-2018-11-21 alone holds three scenes), so
        the 10-clip harness enumerates SCENES, not logs."""
        log_of = {l['token']: l['logfile'] for l in self.table('log')}
        scene_by_log = {log_of[s['log_token']]: s['token']
                        for s in self.table('scene')}
        if log_prefix not in scene_by_log:
            raise KeyError(f'log {log_prefix!r} not found')
        scene_token = scene_by_log[log_prefix]
        samples = self._by_token('sample')
        sd_of = self._by_token('sample_data')
        sensor_of = {s['token']: s['channel'] for s in self.table('sensor')}
        ch_of = {c['token']: sensor_of[c['sensor_token']]
                 for c in self.table('calibrated_sensor')}
        want = set(channels) | {lidar}
        per_sample = {}
        for sd in sd_of.values():
            if not sd['is_key_frame']:
                continue
            ch = ch_of[sd['calibrated_sensor_token']]
            if ch in want:
                per_sample.setdefault(sd['sample_token'], {})[ch] = sd
        out = []
        scenes = self._by_token('scene')
        for tok, recs in per_sample.items():
            samp = samples[tok]
            if samp['scene_token'] != scene_token:
                continue
            if scene_name is not None \
                    and scenes[samp['scene_token']]['name'] != scene_name:
                continue
            if lidar in recs and all(c in recs for c in channels):
                recs['sample'] = samp
                out.append(recs)
        out.sort(key=lambda r: r['sample']['timestamp'])
        return out

    def frames_of_log(self, log_prefix, channel_a='CAM_FRONT',
                      channel_b='LIDAR_TOP'):
        """Key-frame pairs (one per nuScenes sample) for one log, sorted by
        time. Raw schema has no sample['data'] map (that is a devkit
        derived field), so key sample_data is grouped by sample_token."""
        log_of = {l['token']: l['logfile'] for l in self.table('log')}
        scene_by_log = {log_of[s['log_token']]: s['token']
                        for s in self.table('scene')}
        if log_prefix not in scene_by_log:
            raise KeyError(f'log {log_prefix!r} not found')
        scene_token = scene_by_log[log_prefix]
        samples = self._by_token('sample')
        sd_of = self._by_token('sample_data')
        sensor_of = {s['token']: s['channel'] for s in self.table('sensor')}
        ch_of = {c['token']: sensor_of[c['sensor_token']]
                 for c in self.table('calibrated_sensor')}
        per_sample = {}
        for sd in sd_of.values():
            if not sd['is_key_frame']:
                continue
            ch = ch_of[sd['calibrated_sensor_token']]
            if ch in (channel_a, channel_b):
                per_sample.setdefault(sd['sample_token'], {})[ch] = sd
        out = []
        for tok, recs in per_sample.items():
            samp = samples[tok]
            if samp['scene_token'] != scene_token:
                continue
            if channel_a in recs and channel_b in recs:
                out.append(dict(sample=samp, CAM_FRONT=recs[channel_a],
                                LIDAR_TOP=recs[channel_b]))
        out.sort(key=lambda r: r['sample']['timestamp'])
        return out

    # ---- calibration / poses ----------------------------------------------
    def calib(self, sd_record, channel):
        cs = self._by_token('calibrated_sensor')[
            sd_record['calibrated_sensor_token']]
        cal = dict(R_cs=quat_to_mat(cs['rotation']),
                   t_cs=np.array(cs['translation'], float))
        if channel.startswith('CAM'):
            cal['K'] = np.array(cs['camera_intrinsic'])
        return cal

    def ego_pose(self, sd_record):
        return T_of(self._by_token('ego_pose')[sd_record['ego_pose_token']])

    # ---- annotations --------------------------------------------------------
    def dynamic_boxes_global(self, sample):
        """Dynamic-object annotations as box->global 4x4 poses + extents.
        Raw schema chain: annotation -> instance -> category_token ->
        category.name (devkit derives category_name)."""
        names = {c['token']: c['name'] for c in self.table('category')}
        inst_cat = {i['token']: i['category_token']
                    for i in self.table('instance')}
        out = []
        for ann in self.table('sample_annotation'):
            if ann['sample_token'] != sample['token']:
                continue
            if not names[inst_cat[ann['instance_token']]].startswith(
                    DYNAMIC_CATEGORIES):
                continue
            T_bg = np.eye(4)
            T_bg[:3, :3] = quat_to_mat(ann['rotation'])
            T_bg[:3, 3] = np.array(ann['translation'], float)
            out.append(dict(T_bg=T_bg, size=np.array(ann['size'], float)))
        return out

    # ---- raw data -----------------------------------------------------------
    def load_image(self, sd_record):
        import cv2
        p = self.root / sd_record['filename']
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        return img

    def load_sweep(self, sd_record, with_intensity=False):
        p = self.root / sd_record['filename']
        if not p.exists() and self.tar_path.exists():
            with tarfile.open(self.tar_path) as tf:
                tf.extract(sd_record['filename'], self.root)
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 5)
        return pts[:, :4 if with_intensity else 3].astype(np.float64)

    def sweep_history(self, lidar_sd, n=10):
        """n most recent LIDAR_TOP sweeps ending at lidar_sd (inclusive),
        chronological order. Sweeps are 50 ms apart and each carries its OWN
        ego_pose record (verified on v1.0-mini: distinct tokens, ~10 cm ego
        motion between consecutive sweeps), so per-sweep bridging removes
        the inter-sweep motion; the within-sweep smear at city speed is
        <1 px and is ignored (same simplification as stacking references;
        a production system would bridge with LiDAR odometry instead)."""
        sd_of = self._by_token('sample_data')
        cs = lidar_sd['calibrated_sensor_token']
        chain, tok = [], lidar_sd['token']
        while tok and len(chain) < n:
            sd = sd_of[tok]
            if sd['calibrated_sensor_token'] == cs:
                chain.append(sd)
            tok = sd['prev']
        chain.reverse()
        return chain
