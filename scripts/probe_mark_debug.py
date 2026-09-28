# -*- coding: utf-8 -*-
"""Debug the marking-point selection: where do the selected points sit in
the reference-ego frame, and what does the BEV occupancy/rim look like?"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auto_extrinsics.data.nuscenes_lite import NuScenesLite      # noqa: E402
from auto_extrinsics.fine import geometry as geo                 # noqa: E402
from auto_extrinsics.fine import evidence as ev                  # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'
OUT = os.path.join(ROOT, 'outputs', 'm1')
os.makedirs(OUT, exist_ok=True)

ds = NuScenesLite()
recs = ds.frames_of_log(LOG)[0:1]
rec = recs[0]
cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
cal_c = ds.calib(cam_sd, 'CAM_FRONT')
cal_l = ds.calib(lid_sd, 'LIDAR_TOP')
img = ds.load_image(cam_sd)
fg = geo.FrameGeom(name='f00', K=cal_c['K'], R_ec=cal_c['R_cs'],
                   t_ec=cal_c['t_cs'], ego_c=ds.ego_pose(cam_sd),
                   R_le=cal_l['R_cs'], t_le=cal_l['t_cs'],
                   ego_l=ds.ego_pose(lid_sd), img_shape=img.shape[:2])
sweeps = []
for sd in ds.sweep_history(lid_sd, 10):
    arr = ds.load_sweep(sd, with_intensity=True)
    p = arr[arr[:, 0] > -10.0]
    sweeps.append(dict(p_l=p[:, :3], intensity=p[:, 3],
                       T_eg=ds.ego_pose(sd)))
fg.set_stacked(sweeps)

p_e = fg.stacked_ego_ref()
print(f'stacked {len(p_e)} pts')
print(f'p_e z stats: min {p_e[:,2].min():.2f} p5 {np.percentile(p_e[:,2],5):.2f} '
      f'med {np.percentile(p_e[:,2],50):.2f} p95 {np.percentile(p_e[:,2],95):.2f} '
      f'max {p_e[:,2].max():.2f}')

band = ((p_e[:, 0] > -15.0) & (p_e[:, 0] < 70.0)
        & (np.abs(p_e[:, 1]) < 25.0)
        & (p_e[:, 2] > -1.5) & (p_e[:, 2] < 0.5))
print(f'band pts: {band.sum()}')
C = p_e[band]
c = np.median(C, axis=0)
_, _, vh = np.linalg.svd(C - c, full_matrices=False)
n = vh[-1]
if n[2] < 0:
    n = -n
print(f'ground plane: center {np.round(c,2)} normal {np.round(n,3)}')
ground = band & (np.abs((p_e - c) @ n) < 0.10)
it = fg.mark_intensity()
print(f'ground pts: {ground.sum()}, intensity p50/p90 of ground: '
      f'{np.percentile(it[ground], 50):.2f}/{np.percentile(it[ground], 90):.2f}')
cutoff = float(np.percentile(it[ground], 90))
high = ground & (it >= cutoff)
print(f'high-intensity ground: {high.sum()}')

ids = ev.marking_points(fg, it, max_points=100000)
print(f'selected markings (no cap): {len(ids)}')
sel = p_e[ids]
print('selected z hist:',
      np.histogram(sel[:, 2], bins=[-2, -1, -0.5, 0, 0.5, 1, 2, 5, 100])[0])
print('selected xy: x med', np.median(sel[:, 0]), 'y med', np.median(sel[:, 1]))

# draw ALL high-intensity ground points in the image (blue), selected (red)
uv, z, ok = fg.project_stacked(fg.R_ec0, fg.R_le0)
vis = img.copy()
h, w = fg.shape
uhi = uv[high & ok]
for u_, v_ in uhi.astype(int):
    if 0 <= u_ < w and 0 <= v_ < h:
        cv2.circle(vis, (u_, v_), 2, (255, 0, 0), -1)
for u_, v_ in uv[ids][ok[ids]].astype(int):
    cv2.circle(vis, (u_, v_), 2, (0, 0, 255), -1)
cv2.imwrite(os.path.join(OUT, 'mark_debug_image.png'), vis)

# BEV of the high-intensity ground points (x forward up, y left right)
bev = np.full((700, 700, 3), 255, np.uint8)

def to_bev(P):
    gx = np.round(P[:, 0] / 0.1).astype(int) + 50
    gy = np.round(P[:, 1] / 0.1).astype(int) + 350
    m = (gx >= 0) & (gx < 700) & (gy >= 0) & (gy < 700)
    return gx[m], gy[m]

gx, gy = to_bev(p_e[high])
bev[700 - 1 - gy, gx] = (255, 100, 0)
gx2, gy2 = to_bev(sel)
bev[700 - 1 - gy2, gx2] = (0, 0, 255)
cv2.circle(bev, (50, 350), 4, (0, 0, 0), -1)
cv2.putText(bev, 'BEV x fwd / y left; orange=high-int ground, red=selected',
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
cv2.imwrite(os.path.join(OUT, 'mark_debug_bev.png'), bev)
print('saved mark_debug_image.png / mark_debug_bev.png')
