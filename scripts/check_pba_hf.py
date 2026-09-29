# -*- coding: utf-8 -*-
"""Smoke checks for the PBA-HF changes (accessor, self-test, adapter)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.spatial.transform import Rotation

from auto_extrinsics.data.nuscenes_lite import NuScenesLite
from auto_extrinsics.fine import pba

# 1. residual algebra still exact after the refactor
r_true, r_zero = pba.self_test()
print(f'self_test: |r(true)| max {r_true:.2e} (want <1e-6), '
      f'|r(0)| mean {r_zero:.3f} (want >0.01)')

# 2. FixedLidarPBA recovers a pure-camera error from the same evidence
rng = np.random.default_rng(1)
d_ec = np.radians([-0.5, 0.2, -0.4])
n_true = np.array([0.0, -1.0, 0.05]); n_true /= np.linalg.norm(n_true)
d_true = -8.0
pts = rng.normal(size=(400, 3)) * [10, 1, 5] + [20, 0, 2]
pts += (d_true - n_true @ pts.T)[:, None] * n_true
Ec = Rotation.from_rotvec(d_ec).as_matrix()
u_est = (pts - np.array([1.6, 0, 1.5])) @ Ec.T + np.array([1.6, 0, 1.5])
c = pts.mean(0)
_, _, vh = np.linalg.svd(pts - c, full_matrices=False)
n_hat = vh[-1]; d_hat = -float(n_hat @ c)
pb = pba.PBAResidual(u_est, np.zeros(400, np.int32), np.ones(400),
                     [(n_hat, d_hat)], np.eye(4), np.array([0, 0, 1.8]),
                     np.array([1.6, 0, 1.5]),
                     np.repeat(np.eye(4)[None], 400, axis=0),
                     np.eye(3), np.eye(3))
fl = pba.FixedLidarPBA(pb)
res = fl.solve(np.zeros(3), np.arange(400))
print(f'FixedLidarPBA: x_ec recovered (deg) '
      f'{np.array2string(np.degrees(res.x), precision=2)} '
      f'(want ~[-50, 20, -40] sign-inverted => '
      f'{np.array2string(np.degrees(-res.x), precision=2)})')

# 3. camera_window accessor: counts + ego poses + is_key flags
ds = NuScenesLite()
recs = ds.frames_of_log('n008-2018-08-01-15-16-36-0400')
rec = recs[5]
win = ds.camera_window(rec['CAM_FRONT'], 6, 6)
toks = [w['token'] for w in win]
print(f'camera_window: {len(win)} frames, key flags '
      f'{[w["is_key_frame"] for w in win]}')
dts = np.diff([w['timestamp'] for w in win]) * 1e-6
print(f'  dt ms: {np.array2string(dts, precision=1)}')
eg = [ds.ego_pose(w) for w in win]
step = np.linalg.norm(eg[1][:3, 3] - eg[0][:3, 3])
print(f'  per-step ego motion ~{step:.2f} m, unique ego tokens '
      f'{len({w["ego_pose_token"] for w in win})}/{len(win)}')
