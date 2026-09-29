# -*- coding: utf-8 -*-
"""Debug 3: is the 12 Hz ego chain consistent with the images?
Estimate F from LK correspondences; compare recovered relative pose to
the oracle ego_pose chain. Also print exact frame dts."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from auto_extrinsics.data.nuscenes_lite import NuScenesLite
from auto_extrinsics.fine import geometry as geo
from auto_extrinsics.fine import pba

ds = NuScenesLite()
LOG = 'n008-2018-08-01-15-16-36-0400'
CI = int(sys.argv[1]) if len(sys.argv) > 1 else 5
recs = ds.frames_of_log_multi(LOG, channels=('CAM_FRONT',))
centers = recs[::3][:10]
cal_c = ds.calib(centers[0]['CAM_FRONT'], 'CAM_FRONT')

sds = ds.camera_window(centers[CI]['CAM_FRONT'], 6, 6)
dts = np.diff([s['timestamp'] for s in sds])
print('exact dts (ms):', dts / 1000.0)

win = []
for sd in sds:
    cc = ds.calib(sd, 'CAM_FRONT')
    fg = geo.FrameGeom(name=sd['token'][:8], K=cc['K'], R_ec=cc['R_cs'],
                       t_ec=cc['t_cs'], ego_c=ds.ego_pose(sd),
                       R_le=np.eye(3), t_le=np.zeros(3),
                       ego_l=ds.ego_pose(sd),
                       img_shape=(900, 1600))
    fg.img = ds.load_image(sd)
    fg.R_ec0 = cal_c['R_cs'].copy()
    win.append(fg)
ridx = 6
K = win[ridx].K

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

for step in (1, 6):
    j = ridx + step
    idx = [i for i in obs if j in obs[i]]
    uvr = np.array([obs[i][ridx] for i in idx], np.float32)
    uvi = np.array([obs[i][j] for i in idx], np.float32)
    flow = np.linalg.norm(uvi - uvr, axis=1)
    print(f'\nstep {step:+d}: n {len(idx)}, flow px med '
          f'{np.median(flow):.1f}')

    F, mask = cv2.findFundamentalMat(uvr, uvi, cv2.USAC_MAGSAC, 2.0,
                                     0.999, 10000)
    inl = int(mask.sum())
    E_f = K.T @ F
    # oracle
    T_j_ref = np.linalg.inv(T_wc_ref) @ (win[j].T_ge_c @ T_ec)
    R_o, t_o = T_j_ref[:3, :3], T_j_ref[:3, 3]
    E_o = np.cross(t_o, np.eye(3)) @ R_o
    out = cv2.recoverPose(E_f, uvr[mask.ravel() == 1],
                          uvi[mask.ravel() == 1], K)
    R_f, t_f = out[1], out[2]
    dang = np.degrees(np.arccos(np.clip(
        (np.trace(R_f @ R_o.T) - 1) / 2, -1, 1)))
    tdir = np.degrees(np.arccos(np.clip(
        float(t_f.ravel() @ t_o / (np.linalg.norm(t_f)
                                   * np.linalg.norm(t_o))), -1, 1)))
    tf_ratio = np.linalg.norm(t_f) / np.linalg.norm(t_o)
    print(f'  F-RANSAC inliers {inl}/{len(idx)}; '
          f'recovered-vs-oracle: dR {dang:.2f} deg, '
          f't-dir {tdir:.2f} deg, |t_f|/|t_o| {tf_ratio:.2f}')
    # inlier Sampson under ORACLE E, in px
    x1 = (np.linalg.inv(K) @ np.c_[uvr, np.ones(len(uvr))].T).T[:, :3]
    x2 = (np.linalg.inv(K) @ np.c_[uvi, np.ones(len(uvi))].T).T[:, :3]
    E1 = x1 @ E_o.T
    E2 = x2 @ E_o
    num = (E1 * x2).sum(1)
    den = (E1 ** 2).sum(1) + (E2 ** 2).sum(1)
    d_norm = np.abs(num) / np.sqrt(np.maximum(den, 1e-12))
    d_px = d_norm * np.sqrt(K[0, 0] ** 2 + K[1, 1] ** 2) / np.sqrt(2)
    print(f'  point-to-epipolar-line px under ORACLE E: med '
          f'{np.median(d_px):.1f}, p10 {np.percentile(d_px, 10):.1f}, '
          f'p90 {np.percentile(d_px, 90):.1f}')
