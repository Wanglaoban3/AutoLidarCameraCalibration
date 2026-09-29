# -*- coding: utf-8 -*-
"""Debug 7: per-step NCC template matching instead of LK chaining.
Census identical to dbg6: 3-anchor DLT vs the center LiDAR depth map."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import cv2
import numpy as np

from auto_extrinsics.data.nuscenes_lite import NuScenesLite
from auto_extrinsics.fine import geometry as geo
from auto_extrinsics.fine import pba
from run_pba import build_frames

PATCH = 15      # template half-size -> 31x31
SEARCH = 24     # search radius px (per-step motion is 4-26 px)
NCC_MIN = 0.80  # peak response floor
NCC_MARGIN = 1.10  # peak / second-peak ambiguity guard

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
print(f'anchors at {[i for i, k in enumerate(key_mask) if k]}, '
      f'ridx {ridx}, chain <= {MAXSTEPS} steps')

(kp0, _), _ = pba.detect_features(win[ridx].img)
uv0 = np.array([k.pt for k in kp0], np.float32)
grays = [cv2.cvtColor(f.img, cv2.COLOR_BGR2GRAY) for f in win]
H, W = grays[0].shape


def crop(img, cx, cy, half):
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half + 1 > W or y0 + 2 * half + 1 > H:
        return None
    return img[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1]


def ncc_step(img_a, img_b, uv_a):
    """Global NCC template match for one point, one step. Returns
    (uv_b, response) or (None, 0)."""
    tpl = crop(img_a, uv_a[0], uv_a[1], PATCH)
    if tpl is None:
        return None, 0.0
    cx, cy = int(round(uv_a[0])), int(round(uv_a[1]))
    x0, y0 = cx - SEARCH, cy - SEARCH
    x1, y1 = cx + SEARCH, cy + SEARCH
    if x0 < 0 or y0 < 0 or x1 + 1 > W or y1 + 1 > H:
        return None, 0.0
    roi = img_b[y0:y1 + 1, x0:x1 + 1]
    if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]:
        return None, 0.0
    r = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(r)
    # ambiguity: best vs second best outside a small neighborhood
    r2 = r.copy()
    cv2.rectangle(r2, (max(ml[0] - 3, 0), max(ml[1] - 3, 0)),
                  (min(ml[0] + 3, r2.shape[1] - 1),
                   min(ml[1] + 3, r2.shape[0] - 1)), -1, -1)
    _, mx2, _, _ = cv2.minMaxLoc(r2)
    if mx < NCC_MIN or mx < NCC_MARGIN * max(mx2, -1):
        return None, float(mx)
    # integer peak -> subpixel parabola in x and y
    px, py = ml
    dx = dy = 0.0
    if 0 < px < r.shape[1] - 1:
        denom = (r[py, px - 1] - 2 * r[py, px] + r[py, px + 1])
        if abs(denom) > 1e-9:
            dx = 0.5 * (r[py, px - 1] - r[py, px + 1]) / denom
    if 0 < py < r.shape[0] - 1:
        denom = (r[py - 1, px] - 2 * r[py, px] + r[py + 1, px])
        if abs(denom) > 1e-9:
            dy = 0.5 * (r[py - 1, px] - r[py + 1, px]) / denom
    ux = x0 + px + PATCH + dx
    uy = y0 + py + PATCH + dy
    return np.array([ux, uy], np.float32), float(mx)


t0 = time.time()
obs = {i: [(ridx, uv0[i])] for i in range(len(uv0))}
for sgn in (1, -1):
    pos = uv0.copy()
    al = np.ones(len(uv0), bool)
    prev = ridx
    for step in range(1, MAXSTEPS + 1):
        j = ridx + sgn * step
        if not (0 <= j < len(win)):
            break
        for i in np.nonzero(al)[0]:
            uv_b, resp = ncc_step(grays[prev], grays[j], pos[i])
            if uv_b is None:
                al[i] = False
                continue
            obs[i].append((j, uv_b))
            pos[i] = uv_b
        prev = j
print(f'chaining {time.time() - t0:.1f}s, survivors '
      f'{sum(1 for v in obs.values() if len(v) > 1)}')

P, C = pba._proj_centers(win, ridx)
med_es, maxes, angs, zerr = [], [], [], []
n_reproj = n_ang = n_z = n_pass = 0
n3 = 0
for i, views in obs.items():
    anchors = [(j, uv) for j, uv in views if key_mask[j]]
    if len(anchors) < 3:
        continue
    n3 += 1
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


print(f'3-anchor pts {n3}')
print(f'median reproj px {pct(med_es)}, max es px {pct(maxes)}, '
      f'max angle deg {pct(angs)}')
print(f'gates: reproj {n_reproj}, angle {n_ang}, z-range {n_z}, '
      f'depth-consistent {n_pass}')
print(f'z - z_lidar m {pct(zerr)}')
