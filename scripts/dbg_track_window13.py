# -*- coding: utf-8 -*-
"""Debug 13: WHERE do the adjudicated tracks sit? Image + top-down."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import cv2
import numpy as np

from auto_extrinsics.data.nuscenes_lite import NuScenesLite
from auto_extrinsics.fine import pba
from run_pba import build_frames

ds = NuScenesLite()
LOG = 'n008-2018-08-01-15-16-36-0400'
recs = ds.frames_of_log_multi(LOG, channels=('CAM_FRONT',))
cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')

kfs = build_frames(ds, recs)
for fg in kfs:
    fg.R_ec0 = cal_c['R_cs'].copy()
    fg.R_le0 = cal_l['R_cs'].copy()

u_refs = []

centers = [i for i in range(len(kfs)) if i % 2 == 0][:6]
center_fgs = [kfs[i] for i in centers]
rng = np.random.default_rng(7)
planes, per_frame, cloud = pba.extract_facades(center_fgs, 3, rng)
T_ge_ref = center_fgs[3].T_ge_c
print(f'{len(planes)} planes')

feats_cache = {}
all_uv, all_d, all_z, all_zl = [], [], [], []
vis_img = None
for i in centers[:4]:
    offsets = []
    p_i = kfs[i].T_ge_c[:3, 3]
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(kfs) and np.linalg.norm(
                kfs[j].T_ge_c[:3, 3] - p_i) >= 1.2:
            offsets.append(off)
    B = np.linalg.inv(kfs[i].T_ge_c) @ T_ge_ref
    planes_k = [(n, d, inl @ B[:3, :3].T + B[:3, 3])
                for n, d, inl in per_frame]
    m = pba.facade_mask(kfs[i], planes_k)
    print(f'kf{i}: mask {m.mean() * 100:.1f}%, offsets {offsets}')
    feat = pba.detect_features(kfs[i].img, mask=m)
    feats_ref = [None] * len(kfs)
    feats_ref[i] = feat
    dm = pba.build_depth_map(kfs[i])[0]
    tr = pba.build_tracks_pairs(kfs, i, feats_ref,
                                lambda j: feats_cache.setdefault(
                                    j, pba.detect_features(kfs[j].img)),
                                dm, offsets)
    N = np.array([p[0] for p in planes])
    D = np.array([p[1] for p in planes])
    Bb = np.linalg.inv(T_ge_ref) @ kfs[i].T_ge_c
    for t in tr:
        u_ref = Bb[:3, :3] @ t['u'] + Bb[:3, 3]
        d = float(np.abs(N @ u_ref + D).min())
        all_uv.append(t['uv_ref'])
        all_d.append(d)
        all_z.append(t['X_ref'][2])
        gh, gw = dm.shape
        vi = int(np.clip(t['uv_ref'][1] / 8, 0, gh - 1))
        ui = int(np.clip(t['uv_ref'][0] / 8, 0, gw - 1))
        all_zl.append(dm[vi, ui])
        u_refs.append(u_ref)
    if vis_img is None:
        vis_img = kfs[i].img.copy()
    for t, d_ in zip(tr, all_d[-len(tr):]):
        col = (0, 200, 0) if d_ < 0.5 else \
            ((0, 165, 255) if d_ < 2.0 else (0, 0, 255))
        cv2.circle(vis_img, (int(t['uv_ref'][0]),
                             int(t['uv_ref'][1])), 3, col, -1)

uv = np.asarray(all_uv)
d = np.asarray(all_d)
z = np.asarray(all_z)
zl = np.asarray(all_zl)
print(f'tracks {len(d)}: plane-dist [10/50/90] '
      f'{np.percentile(d, [10, 50, 90]).round(2)}')
print(f'z_tri [10/50/90] {np.percentile(z, [10, 50, 90]).round(1)}, '
      f'z_lidar {np.percentile(zl[np.isfinite(zl)], [10, 50, 90]).round(1)}')
print(f'frac d<0.5m: {np.mean(d < 0.5):.3f}, d<2m: {np.mean(d < 2):.3f}')
cv2.imwrite('outputs/m1/dbg13_tracks_img.png', vis_img)
print('saved outputs/m1/dbg13_tracks_img.png')

# top-down: scene-ref ego, x forward, y left
top = np.full((700, 800, 3), 255, np.uint8)
for r in range(len(planes)):
    n, dd, _ = planes[r]
    # draw the plane line: points p with n[0]x + n[1]y + d = 0
    pts = []
    for xx in (-20, 80):
        if abs(n[1]) > 1e-6:
            yy = -(n[0] * xx + dd) / n[1]
            if -25 <= yy <= 25:
                pts.append((xx, yy))
    if len(pts) == 2:
        p1 = (int((pts[0][0] + 20) * 10), int((25 - pts[0][1]) * 14))
        p2 = (int((pts[1][0] + 20) * 10), int((25 - pts[1][1]) * 14))
        cv2.line(top, p1, p2, (200, 200, 200), 1)
for u_ref, dd in zip(u_refs, all_d):
    px = int((u_ref[0] + 20) * 10)
    py = int((25 - u_ref[1]) * 14)
    if 0 <= px < 800 and 0 <= py < 700:
        col = (0, 180, 0) if dd < 0.5 else \
            ((0, 165, 255) if dd < 2.0 else (0, 0, 255))
        cv2.circle(top, (px, py), 2, col, -1)
cv2.imwrite('outputs/m1/dbg13_topdown.png', top)
print('saved outputs/m1/dbg13_topdown.png')

