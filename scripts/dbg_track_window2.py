# -*- coding: utf-8 -*-
"""Debug 2: synthetic DLT sanity + per-pair Sampson of LK chains."""
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
recs = ds.frames_of_log_multi(LOG, channels=('CAM_FRONT',))
centers = recs[::3][:10]
cal_c = ds.calib(centers[0]['CAM_FRONT'], 'CAM_FRONT')

# ---- window (images + poses) -------------------------------------------
sds = ds.camera_window(centers[5]['CAM_FRONT'], 6, 6)
win = []
for sd in sds:
    cc = ds.calib(sd, 'CAM_FRONT')
    img = ds.load_image(sd)
    fg = geo.FrameGeom(name=sd['token'][:8], K=cc['K'], R_ec=cc['R_cs'],
                       t_ec=cc['t_cs'], ego_c=ds.ego_pose(sd),
                       R_le=np.eye(3), t_le=np.zeros(3),
                       ego_l=ds.ego_pose(sd), img_shape=img.shape[:2])
    fg.img = img
    fg.R_ec0 = cal_c['R_cs'].copy()
    fg.R_le0 = cal_l['R_cs'] if (cal_l := ds.calib(
        centers[5]['LIDAR_TOP'], 'LIDAR_TOP')) is not None else None
    win.append(fg)
ridx = sds.index(centers[5]['CAM_FRONT'])
K = win[ridx].K

# ---- 1. synthetic DLT through the SAME code path ------------------------
P, C = pba._proj_centers(win, ridx)
rng = np.random.default_rng(0)
X_true = np.array([3.0, -8.0, 30.0])          # ref-cam coords, wall-ish
uvs = []
for j in sorted(P):
    p = P[j] @ np.r_[X_true, 1.0]
    uv = p[:2] / p[2] + rng.normal(scale=0.3, size=2)
    uvs.append((j, uv))
X_est = pba._dlt([P[j] for j, _ in uvs], [uv for _, uv in uvs])
es = []
for j, uv in uvs:
    p = P[j] @ np.r_[X_est, 1.0]
    es.append(np.linalg.norm(p[:2] / p[2] - uv))
print(f'[synthetic] X_true {X_true.round(2)} -> est '
      f'{X_est.round(3) if X_est is not None else None}, '
      f'reproj med {np.median(es):.3f} px')

# ---- 2. real LK chains: per-pair Sampson under the oracle E -------------
(kp0, _), _ = pba.detect_features(win[ridx].img)
uv0 = np.array([k.pt for k in kp0], np.float32)
grays = [cv2.cvtColor(f.img, cv2.COLOR_BGR2GRAY) for f in win]
lk = dict(winSize=(21, 21), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30,
                    0.01))
obs = {i: {ridx: uv0[i]} for i in range(len(uv0))}
for sgn in (1, -1):
    chain = uv0.reshape(-1, 1, 2).copy()
    al = np.ones(len(uv0), bool)
    prev = ridx
    for step in range(1, len(win)):
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
            obs[i][j] = p2[i, 0].copy()
        chain, al, prev = p2, ok, j

T_ec = np.eye(4)
T_ec[:3, :3], T_ec[:3, 3] = win[ridx].R_ec0, win[ridx].t_ec
T_wc_ref = win[ridx].T_ge_c @ T_ec
print('[sampson] per pair: n, median Sampson (px^2), p90')
for step in (-6, -3, -1, 1, 3, 6):
    j = ridx + step
    if j < 0 or j >= len(win):
        continue
    T_j_ref = np.linalg.inv(T_wc_ref) @ (win[j].T_ge_c @ T_ec)
    R, t = T_j_ref[:3, :3], T_j_ref[:3, 3]
    E = np.cross(t, np.eye(3)) @ R
    idx = [i for i in obs if j in obs[i]]
    if len(idx) < 20:
        continue
    uvr = np.array([obs[i][ridx] for i in idx])
    uvi = np.array([obs[i][j] for i in idx])
    x1 = (np.linalg.inv(K) @ np.c_[uvr, np.ones(len(uvr))].T).T[:, :3]
    x2 = (np.linalg.inv(K) @ np.c_[uvi, np.ones(len(uvi))].T).T[:, :3]
    E1 = x1 @ E.T
    E2 = x2 @ E
    num = (E1 * x2).sum(1)
    den = (E1 ** 2).sum(1) + (E2 ** 2).sum(1)
    samp = np.abs(num) / np.maximum(den, 1e-12)
    # pixel units: multiply by (f/2) approx -> report normalized + px est
    print(f'  step {step:+d}: n {len(idx)}, sampson med '
          f'{np.median(samp):.3e} (norm), px~'
          f'{np.median(samp) * K[0, 0] ** 2 / 2:.2f}')
