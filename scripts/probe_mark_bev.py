# -*- coding: utf-8 -*-
"""Pin down the marking-selection bug: dump the BEV occupancy pipeline
stages and the (range, azimuth) pattern of the selected points."""
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

ds = NuScenesLite()
rec = ds.frames_of_log(LOG)[0]
cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
cal_c = ds.calib(cam_sd, 'CAM_FRONT')
cal_l = ds.calib(lid_sd, 'LIDAR_TOP')
fg = geo.FrameGeom(name='f00', K=cal_c['K'], R_ec=cal_c['R_cs'],
                   t_ec=cal_c['t_cs'], ego_c=ds.ego_pose(cam_sd),
                   R_le=cal_l['R_cs'], t_le=cal_l['t_cs'],
                   ego_l=ds.ego_pose(lid_sd), img_shape=(900, 1600))
sweeps = []
for sd in ds.sweep_history(lid_sd, 10):
    arr = ds.load_sweep(sd, with_intensity=True)
    p = arr[arr[:, 0] > -10.0]
    sweeps.append(dict(p_l=p[:, :3], intensity=p[:, 3],
                       T_eg=ds.ego_pose(sd)))
fg.set_stacked(sweeps)

p_e = fg.stacked_ego_ref()
it = fg.mark_intensity()
band = ((p_e[:, 0] > -15.0) & (p_e[:, 0] < 70.0)
        & (np.abs(p_e[:, 1]) < 25.0)
        & (p_e[:, 2] > -1.5) & (p_e[:, 2] < 0.5))
C = p_e[band]
c = np.median(C, axis=0)
_, _, vh = np.linalg.svd(C - c, full_matrices=False)
n = vh[-1]
if n[2] < 0:
    n = -n
ground = band & (np.abs((p_e - c) @ n) < 0.10)
res, x0_, y0_ = 0.10, -5.0, -40.0
# ego-vehicle footprint: bumpers/doors are z<0.5, high-intensity paint, and
# surround the sensor (not annotated as dynamic) -- they would otherwise
# dominate the intensity selection as constant-range rings
ego_m = ((p_e[:, 0] > -4.5) & (p_e[:, 0] < 1.5)
         & (np.abs(p_e[:, 1]) < 1.3))
print(f'ego-footprint ground pts removed: {int((ground & ego_m).sum())}')
ground = ground & ~ego_m
for pct in (90.0, 95.0, 98.0):
    cutoff = float(np.percentile(it[ground], pct))
    high = ground & (it >= cutoff)
    print(f'-- pct {pct}: cutoff {cutoff:.0f}, high pts {int(high.sum())}')
    gx_ = ((p_e[:, 0] - x0_) / res).astype(np.int32)
    gy_ = ((p_e[:, 1] - y0_) / res).astype(np.int32)
    ig = high & (gx_ >= 0) & (gx_ < 750) & (gy_ >= 0) & (gy_ < 800)
    occ = np.zeros((800, 750), np.uint8)
    occ[gy_[ig], gx_[ig]] = 255
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    kept = [l for l in range(1, count)
            if stats[l, cv2.CC_STAT_AREA] >= 8]
    tot = sum(int((labels == l).sum()) for l in kept)
    far = sum(1 for l in kept
              if x0_ + stats[l, cv2.CC_STAT_LEFT] * res > 8.0)
    print(f'   comps kept {len(kept)}, cells {tot}, comps with x>8m: {far}')
    for l in sorted(kept, key=lambda l: -stats[l, cv2.CC_STAT_AREA])[:5]:
        print(f'   comp x [{x0_ + stats[l, cv2.CC_STAT_LEFT] * res:.1f},'
              f'{x0_ + (stats[l, cv2.CC_STAT_LEFT] + stats[l, cv2.CC_STAT_WIDTH]) * res:.1f}] '
              f'y [{y0_ + stats[l, cv2.CC_STAT_TOP] * res:.1f},'
              f'{y0_ + (stats[l, cv2.CC_STAT_TOP] + stats[l, cv2.CC_STAT_HEIGHT]) * res:.1f}] '
              f'area {stats[l, cv2.CC_STAT_AREA]}')
cutoff = float(np.percentile(it[ground], 95))
high = ground & (it >= cutoff)

# far-field paint: intensity of ground returns 10-45 m ahead
far = ground & (p_e[:, 0] > 10.0) & (p_e[:, 0] < 45.0)
print(f'far ground pts {int(far.sum())}, intensity p50/p90/p99: '
      f'{np.percentile(it[far], 50):.0f}/{np.percentile(it[far], 90):.0f}/'
      f'{np.percentile(it[far], 99):.0f}')
hist2, e2 = np.histogram(it[far], bins=[0, 8, 12, 16, 20, 25, 30, 35, 40,
                                        50, 70, 256])
print('far intensity hist:', dict(zip(e2[:-1].astype(int), hist2)))
for thr, k_close in ((30, 3), (35, 3), (35, 5), (40, 5)):
    hi = ground & (it >= thr)
    gx2 = ((p_e[:, 0] - x0_) / res).astype(np.int32)
    gy2 = ((p_e[:, 1] - y0_) / res).astype(np.int32)
    ig2 = hi & (gx2 >= 0) & (gx2 < 750) & (gy2 >= 0) & (gy2 < 800)
    occ2 = np.zeros((800, 750), np.uint8)
    occ2[gy2[ig2], gx2[ig2]] = 255
    cl2 = cv2.morphologyEx(occ2, cv2.MORPH_CLOSE,
                           np.ones((k_close, k_close), np.uint8))
    cnt2, lab2, st2, _ = cv2.connectedComponentsWithStats(cl2, 8)
    keep2 = [l for l in range(1, cnt2) if st2[l, cv2.CC_STAT_AREA] >= 8]
    farc = [l for l in keep2
            if x0_ + st2[l, cv2.CC_STAT_LEFT] * res > 10.0]
    print(f'thr {thr} close {k_close}: high {int(hi.sum())}, comps '
          f'{len(keep2)} (far x>10m: {len(farc)}), cells '
          f'{sum(int((lab2 == l).sum()) for l in keep2)}')
    for l in sorted(farc, key=lambda l: -st2[l, cv2.CC_STAT_AREA])[:4]:
        print(f'   far comp x [{x0_ + st2[l, cv2.CC_STAT_LEFT] * res:.1f},'
              f'{x0_ + (st2[l, cv2.CC_STAT_LEFT] + st2[l, cv2.CC_STAT_WIDTH]) * res:.1f}] '
              f'y [{y0_ + st2[l, cv2.CC_STAT_TOP] * res:.1f},'
              f'{y0_ + (st2[l, cv2.CC_STAT_TOP] + st2[l, cv2.CC_STAT_HEIGHT]) * res:.1f}] '
              f'area {st2[l, cv2.CC_STAT_AREA]}')


# intensity histogram of ground returns
hist, edges = np.histogram(it[ground], bins=[0, 2, 5, 8, 12, 16, 20, 25, 30,
                                             40, 60, 90, 130, 200, 256])
print('ground intensity hist:')
for k in range(len(hist)):
    print(f'  [{edges[k]:5.0f},{edges[k+1]:5.0f}): {hist[k]}')

# BEV stages (x forward -> cols, y left -> rows), 0.1 m, front 70x80 m
res, x0, y0 = 0.10, -5.0, -40.0
gx = ((p_e[:, 0] - x0) / res).astype(np.int32)
gy = ((p_e[:, 1] - y0) / res).astype(np.int32)
in_grid = high & (gx >= 0) & (gx < 750) & (gy >= 0) & (gy < 800)
occ = np.zeros((800, 750), np.uint8)
occ[gy[in_grid], gx[in_grid]] = 255
print(f'in_grid pts {int(in_grid.sum())}, occ nonzero cells '
      f'{int((occ > 0).sum())}')
closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
markings = np.zeros_like(closed)
big = 0
for label in range(1, count):
    if stats[label, cv2.CC_STAT_AREA] >= 8:
        markings[labels == label] = 255
        big += 1
print(f'components: {count - 1}, kept(area>=8): {big}, '
      f'marking cells {int((markings > 0).sum())}')
rim = cv2.morphologyEx(markings, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
print(f'rim cells {int((rim > 0).sum())}')

# who are the marked components? top-10 by area with their bbox in metres
order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
for label in order[:10]:
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    w_ = stats[label, cv2.CC_STAT_WIDTH]
    h_ = stats[label, cv2.CC_STAT_HEIGHT]
    print(f'  comp {label}: area {stats[label, cv2.CC_STAT_AREA]}, '
          f'x [{x0 + x * res:.1f},{x0 + (x + w_) * res:.1f}] m, '
          f'y [{y0 + y * res:.1f},{y0 + (y + h_) * res:.1f}] m')

canvas = np.zeros((800, 750, 3), np.uint8)
canvas[..., 0] = occ
canvas[..., 1] = markings
canvas[..., 2] = rim
canvas = cv2.resize(canvas, (750, 800))
cv2.imwrite(os.path.join(OUT, 'mark_debug_stages.png'), canvas)

# range/azimuth of selected points
ids = np.flatnonzero(in_grid)[rim[gy[in_grid], gx[in_grid]] > 0]
S = p_e[ids]
r = np.linalg.norm(S[:, :2], axis=1)
az = np.degrees(np.arctan2(S[:, 1], S[:, 0]))
print(f'selected {len(ids)}: range p10/50/90 '
      f'{np.percentile(r, 10):.1f}/{np.percentile(r, 50):.1f}/'
      f'{np.percentile(r, 90):.1f} m, azimuth p10/50/90 '
      f'{np.percentile(az, 10):.1f}/{np.percentile(az, 50):.1f}/'
      f'{np.percentile(az, 90):.1f} deg')
