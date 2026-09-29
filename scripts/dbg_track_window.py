# -*- coding: utf-8 -*-
"""Debug one PBA-HF window: where do the KLT tracks die?"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from auto_extrinsics.data.nuscenes_lite import NuScenesLite
from auto_extrinsics.fine import geometry as geo
from auto_extrinsics.fine import pba
from run_pba import build_frames  # noqa: E402  (scripts/ on path)

sys.path.insert(0, os.path.join(ROOT, 'scripts'))

ds = NuScenesLite()
LOG = 'n008-2018-08-01-15-16-36-0400'
CI = int(sys.argv[1]) if len(sys.argv) > 1 else 5
recs = ds.frames_of_log_multi(LOG, channels=('CAM_FRONT',))
centers = recs[::3][:10]
cal_c = ds.calib(centers[0]['CAM_FRONT'], 'CAM_FRONT')
cal_l = ds.calib(centers[0]['LIDAR_TOP'], 'LIDAR_TOP')

# center frame CI with its stacked cloud (GT base)
c5 = build_frames(ds, [centers[CI]])
fg5 = c5[0]
fg5.R_ec0 = cal_c['R_cs'].copy()
fg5.R_le0 = cal_l['R_cs'].copy()

rng = np.random.default_rng(7)
planes, per_frame, cloud = pba.extract_facades(c5, 0, rng)
print(f'{len(planes)} planes from single center')

# window around center 5
sds = ds.camera_window(centers[CI]['CAM_FRONT'], 6, 6)
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
    fg.R_le0 = cal_l['R_cs'].copy()
    win.append(fg)
ridx = next(i for i, s in enumerate(sds)
            if s['token'] == centers[CI]['CAM_FRONT']['token'])
print(f'window {len(win)}, ridx {ridx}')

B = np.linalg.inv(win[ridx].T_ge_c) @ win[ridx].T_ge_c  # identity here
planes_k = [(n, d, inl) for n, d, inl in per_frame]
m = pba.facade_mask(win[ridx], planes_k)
print(f'mask coverage {m.mean() * 100:.1f}%')
feat_raw = pba.detect_features(win[ridx].img)
feat_m = pba.detect_features(win[ridx].img, mask=m)
print(f'SIFT raw {len(feat_raw[0][0])}, masked {len(feat_m[0][0])}')

(kp0, _), _ = feat_m
uv0 = np.array([k.pt for k in kp0], np.float32)
print(f'keypoints {len(uv0)}')
if len(uv0) == 0:
    sys.exit(0)

grays = [cv2.cvtColor(f.img, cv2.COLOR_BGR2GRAY) for f in win]
lk = dict(winSize=(21, 21), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30,
                    0.01))
obs = {i: [(ridx, uv0[i])] for i in range(len(uv0))}
for sgn in (1, -1):
    chain = uv0.reshape(-1, 1, 2).copy()
    al = np.ones(len(uv0), bool)
    prev = ridx
    surv = []
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
        surv.append(int(ok.sum()))
        for i in np.nonzero(ok)[0]:
            obs[i].append((j, p2[i, 0].copy()))
        chain, al, prev = p2, ok, j
    print(f'sgn {sgn:+d} survival per step: {surv}')

P, C = pba._proj_centers(win, ridx)
vs_counts = {}
n_dlt_ok = n_reproj = n_angle = n_depth = n_zrange = 0
dm, cell = pba.build_depth_map(fg5)
zdiffs, angmaxs, reps = [], [], []
for i, views in obs.items():
    vs_counts[len(views)] = vs_counts.get(len(views), 0) + 1
    if len(views) < pba.KLT_MIN_VIEWS:
        continue
    X = pba._dlt([P[j] for j, _ in views], [uv for _, uv in views])
    if X is None or not np.isfinite(X).all():
        continue
    n_dlt_ok += 1
    es, angs = [], []
    rr = X / max(np.linalg.norm(X), 1e-9)
    bad = False
    for j, uv in views:
        p = P[j] @ np.r_[X, 1.0]
        if p[2] <= 0:
            bad = True
            break
        es.append(np.linalg.norm(p[:2] / p[2] - uv))
        dj = (X - C[j]) / max(np.linalg.norm(X - C[j]), 1e-9)
        angs.append(np.degrees(np.arccos(np.clip(float(rr @ dj), -1, 1))))
    if bad or np.isnan(es).any():
        continue
    reps.append(np.median(es))
    angmaxs.append(max(angs))
    if np.median(es) > pba.KLT_REPROJ_MAX_PX or max(es) > 3.0:
        continue
    n_reproj += 1
    if max(angs) < pba.MIN_TRI_ANGLE_DEG:
        continue
    n_angle += 1
    if not (pba.DEPTH_MIN < X[2] < pba.DEPTH_MAX):
        continue
    n_zrange += 1
    gh, gw = dm.shape
    vi = int(np.clip(uv0[i][1] / cell, 0, gh - 1))
    ui = int(np.clip(uv0[i][0] / cell, 0, gw - 1))
    z_l = dm[vi, ui]
    if np.isfinite(z_l) and abs(X[2] - z_l) <= max(0.3 * z_l, 2.0):
        n_depth += 1
        zdiffs.append(X[2] - z_l)

print(f'view-count histogram: '
      f'{dict(sorted(vs_counts.items()))}')
print(f'DLT ok {n_dlt_ok}, reproj-pass {n_reproj}, angle-pass '
      f'{n_angle}, z-range-pass {n_zrange}, depth-pass {n_depth}')
if reps:
    print(f'median reproj px: {np.percentile(reps, [10, 50, 90])}')
if angmaxs:
    print(f'max tri angle deg: {np.percentile(angmaxs, [10, 50, 90])}')
if zdiffs:
    print(f'z - z_lidar m: {np.percentile(zdiffs, [10, 50, 90])}')
