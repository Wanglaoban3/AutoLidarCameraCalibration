# -*- coding: utf-8 -*-
"""Visual sanity: GT-projection alignment. Blue = GT-projected sweep points
(subsampled), drawn over the image; crops saved at 2x zoom for inspection."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite     # noqa: E402
from auto_extrinsics.fine import geometry as geo                # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'
ds = NuScenesLite()
recs = ds.frames_of_log(LOG)
rec = recs[0]
cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
cal_c, cal_l = ds.calib(cam_sd, 'CAM_FRONT'), ds.calib(lid_sd, 'LIDAR_TOP')
img = ds.load_image(cam_sd)
fg = geo.FrameGeom('f0', cal_c['K'], cal_c['R_cs'], cal_c['t_cs'],
                   ds.ego_pose(cam_sd), cal_l['R_cs'], cal_l['t_cs'],
                   ds.ego_pose(lid_sd), img.shape[:2])
p_l = ds.load_sweep(lid_sd)
T_lc = fg.T_lc(cal_c['R_cs'], cal_l['R_cs'])
uv, z, ok = fg.project(p_l, T_lc)
vis = img.copy()
sel = np.nonzero(ok)[0]
sel = sel[np.random.default_rng(0).choice(len(sel), min(6000, len(sel)),
                                          replace=False)]
for i in sel:
    c = (0, 255, 255) if z[i] < 12 else ((0, 165, 255) if z[i] < 30
                                         else (255, 0, 0))
    cv2.circle(vis, (int(round(uv[i, 0])), int(round(uv[i, 1]))), 1, c, -1)
out = ROOT / 'outputs' / 'm1' / 'gt_overlay_full.png'
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), vis)
# zoom crops: left building edge, center road/cars, right facade
h, w = img.shape[:2]
for name, (cx, cy) in dict(left=(300, 350), center=(800, 560),
                           right=(1350, 380)).items():
    x0, y0 = max(cx - 200, 0), max(cy - 160, 0)
    crop = vis[y0:y0 + 320, x0:x0 + 400]
    crop = cv2.resize(crop, (800, 640), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out.parent / f'gt_overlay_{name}.png'), crop)
print('saved', out)
