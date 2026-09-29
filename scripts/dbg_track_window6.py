# -*- coding: utf-8 -*-
"""Debug 6: chain-length vs anchor-triangulation structure quality.
Truncate the KLT chains at N steps and check the 3-anchor DLT
(k-1, k, k+1) against the center LiDAR depth map."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import cv2
import numpy as np

from auto_extrinsics.data.nuscenes_lite import NuScenesLite
from auto_extrinsics.fine import geometry as geo
from auto_extrinsics.fine import pba
from run_pba import build_frames

ds = NuScenesLite()
LOG = 'n008-2018-08-01-15-16-36-0400'
CI = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAXSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
recs = ds.frames_of_log_multi(LOG, channels=('CAM_FRONT',))
centers = recs[::4][:10]
cal_c = ds.calib(centers[0]['CAM_FRONT'], 'CAM_FRONT')
cal_l = ds.calib(centers[0]['LIDAR_TOP'], 'LIDAR_TOP')

c_ci = build_frames(ds, [centers[CI]])
fg_c = c_ci[0]
fg_c.R_ec0 = cal_c['R_cs'].copy()
fg_c.R_le0 = cal_l['R_cs'].copy()
dm, cell = pba.build_depth_map(fg_c)
gh, gw = dm.shape

sds = ds.camera_window(centers[CI]['CAM_FRONT'], 13, 13)
win, key_mask = [], []
for sd in sds:
    cc = ds.calib(sd, 'CAM_FRONT')
    fg = geo.FrameGeom(name=sd['token'][:8], K=cc['K'], R_ec=cc['R_cs'],
                       t_ec=cc['t_cs'], ego_c=ds.ego_pose(sd),
                       R_le=np.eye(3), t_le=np.zeros(3),
                       ego_l=ds.ego_pose(sd), img_shape=(900, 1600))
    fg.img = ds.load_image(sd)
    fg.R_ec0 = cal_c['R_cs'].copy()
    win.append(fg)
    key_mask.append(bool(sd['is_key_frame']))
ridx = next(i for i, s in enumerate(sds)
            if s['token'] == centers[CI]['CAM_FRONT']['token'])

(kp0, _), _ = pba.detect_features(win[ridx].img)
uv0 = np.array([k.pt for k in kp0], np.float32)
grays = [cv2.cvtColor(f.img, cv2.COLOR_BGR2GRAY) for f in win]
lk = dict(winSize=(21, 21), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30,
                    0.01))
obs = {i: [(ridx, uv0[i])] for i in range(len(uv0))}
for sgn in (1, -1):
    chain = uv0.reshape(-1, 1, 2).copy()
    al = np.ones(len(uv0), bool)
    prev = ridx
    for step in range(1, MAXSTEPS + 1):
        j = ridx + sgn * step
        if not (0 <= j < len(win)):
            break
        p2, st, _ = cv2.calcOpticalFlowPyrLK(grays[prev], grays[j],
                                             chain, None, **lk)
        p1b, stb, _ = cv2.calcOpticalFlowPyrLK(grays[j], grays[prev],
                                               p2, None, **lk)
        fb = np.linalg.norm((p1b - chain).reshape(-1, 2), axis=1)
        ok = al & (st.ravel() == 1) & (stb.ravel() == 1) & (fb < 1.0)
        for i in np.nonzero(ok)[0]:
            obs[i].append((j, p2[i, 0].copy()))
        chain, al, prev = p2, ok, j

P, C = pba._proj_centers(win, ridx)
n3 = med_es, maxes, angs, zerr, n_pass = [], [], [], [], 0
n_reproj = n_ang = n_z = 0
for i, views in obs.items():
    anchors = [(j, uv) for j, uv in views if key_mask[j]]
    if len(anchors) < 2:
        continue
    X = pba._dlt([P[j] for j, _ in anchors], [uv for _, uv in anchors])
    if X is None or not np.isfinite(X).all():
        continue
    rr = X / max(np.linalg.norm(X), 1e-9)
    es, ang = [], 0.0
    for j, uv in anchors:
        p = P[j] @ np.r_[X, 1.0]
        if p[2] <= 0:
            es = None
            break
        es.append(np.linalg.norm(p[:2] / p[2] - uv))
        dj = (X - C[j]) / max(float(np.linalg.norm(X - C[j])), 1e-9)
        ang = max(ang, np.degrees(np.arccos(np.clip(
            float(rr @ dj), -1, 1))))
    if es is None:
        continue
    med_es.append(np.median(es))
    maxes.append(max(es))
    angs.append(ang)
    if np.median(es) <= 1.5 and max(es) <= 3.0:
        n_reproj += 1
        if ang < pba.MIN_TRI_ANGLE_DEG:
            continue
        n_ang += 1
        if not (pba.DEPTH_MIN < X[2] < pba.DEPTH_MAX):
            continue
        n_z += 1
        vi = int(np.clip(uv0[i][1] / cell, 0, gh - 1))
        ui = int(np.clip(uv0[i][0] / cell, 0, gw - 1))
        z_l = dm[vi, ui]
        if np.isfinite(z_l):
            zerr.append(X[2] - z_l)
            if abs(X[2] - z_l) <= max(0.30 * z_l, 2.0):
                n_pass += 1


def pct(v):
    v = np.asarray(v)
    if len(v) == 0:
        return 'empty'
    return np.array2string(np.percentile(v, [10, 50, 90]), precision=2)


print(f'center {CI}, chain <= {MAXSTEPS} steps: {len(obs)} pts, '
      f'2+ anchors -> DLT')
print(f'median reproj px {pct(med_es)}, max es px {pct(maxes)}, '
      f'max angle deg {pct(angs)}')
print(f'gates: reproj {n_reproj}, angle {n_ang}, z-range {n_z}, '
      f'depth-consistent {n_pass}')
print(f'z - z_lidar m {pct(zerr)}')
