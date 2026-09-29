# -*- coding: utf-8 -*-
"""Debug 12: are non-key CAMERA ego poses a TIME-SHIFTED trajectory?
Fit p_nk(t) ~= p_sweep(t + delta) over dense 20 Hz sweep poses."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from auto_extrinsics.data.nuscenes_lite import NuScenesLite

ds = NuScenesLite()
LOG = 'n008-2018-08-01-15-16-36-0400'
recs = ds.frames_of_log_multi(LOG, channels=('CAM_FRONT',))

# dense reference trajectory: 20 Hz sweep poses across the log
ref_t, ref_p = [], []
for r in recs:
    for s in ds.sweep_history(r['LIDAR_TOP'], 12):
        e = ds.ego_pose(s)
        ref_t.append(s['timestamp'])
        ref_p.append(e[:3, 3])
ref_t = np.array(ref_t, np.int64)
ref_p = np.asarray(ref_p)
order = np.argsort(ref_t)
ref_t, ref_p = ref_t[order], ref_p[order]
print(f'reference sweep trajectory: {len(ref_t)} poses, '
      f'{(ref_t[-1] - ref_t[0]) * 1e-6:.1f} s')


def traj_at(t):
    i = np.searchsorted(ref_t, t)
    i = np.clip(i, 1, len(ref_t) - 1)
    t0, t1 = ref_t[i - 1], ref_t[i]
    w = (t - t0) / max(t1 - t0, 1)
    return ref_p[i - 1] * (1 - w) + ref_p[i] * w


# non-key camera records over a stretch of the log
sds = ds.camera_window(recs[20]['CAM_FRONT'], 40, 40)
nk = [s for s in sds if not s['is_key_frame']]
print(f'{len(nk)} non-key camera records around keyframe 20')
best = None
for delta_us in range(-6000000, 6000001, 250000):
    errs = []
    for s in nk:
        p = ds.ego_pose(s)[:3, 3]
        errs.append(np.linalg.norm(p - traj_at(s['timestamp']
                                               + delta_us)))
    m = float(np.mean(errs))
    if best is None or m < best[1]:
        best = (delta_us * 1e-6, m)
    if abs(delta_us) % 1000000 == 0 or m < 0.5:
        print(f'  delta {delta_us * 1e-6:+.2f} s: mean |dp| {m:.3f} m')
print(f'BEST: delta {best[0]:+.2f} s, mean |dp| {best[1]:.3f} m')

# same fit around a different keyframe for delta constancy
sds2 = ds.camera_window(recs[5]['CAM_FRONT'], 40, 40)
nk2 = [s for s in sds2 if not s['is_key_frame']]
errs = []
for delta_us in range(-6000000, 6000001, 250000):
    m = float(np.mean([np.linalg.norm(ds.ego_pose(s)[:3, 3]
                                      - traj_at(s['timestamp']
                                                + delta_us))
                       for s in nk2]))
    errs.append((delta_us * 1e-6, m))
b2 = min(errs, key=lambda t: t[1])
print(f'keyframe-5 stretch BEST: delta {b2[0]:+.2f} s, '
      f'mean |dp| {b2[1]:.3f} m')
