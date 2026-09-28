# -*- coding: utf-8 -*-
"""Debug why depth-edge samples come out empty."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite     # noqa: E402
from auto_extrinsics.fine import geometry as geo                # noqa: E402
from scipy.spatial import cKDTree

LOG = 'n008-2018-08-01-15-16-36-0400'
ds = NuScenesLite()
recs = ds.frames_of_log(LOG)
rec = recs[0]
cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
cal_c, cal_l = ds.calib(cam_sd, 'CAM_FRONT'), ds.calib(lid_sd, 'LIDAR_TOP')
img = ds.load_image(cam_sd)
fg = geo.FrameGeom('f', cal_c['K'], cal_c['R_cs'], cal_c['t_cs'],
                   ds.ego_pose(cam_sd), cal_l['R_cs'], cal_l['t_cs'],
                   ds.ego_pose(lid_sd), img.shape[:2])
boxes = ds.dynamic_boxes_global(rec['sample'])
print('dyn boxes:', len(boxes))
mask = geo.build_dynamic_mask(fg, boxes)
print('mask px:', None if mask is None else int(mask.sum()))
p_l = ds.load_sweep(lid_sd)
T_lc = fg.T_lc(cal_c['R_cs'], cal_l['R_cs'])
uv, z, ok = fg.project(p_l, T_lc)
print('ok:', int(ok.sum()))
# no-mask comparison
saved = fg.dyn_mask
fg.dyn_mask = None
uv2, z2, ok2 = fg.project(p_l, T_lc)
print('ok without mask:', int(ok2.sum()))
fg.dyn_mask = saved

valid = np.nonzero(ok)[0]
pv, zv, uvv = p_l[valid], z[valid], uv[valid]
print('z range:', zv.min().round(1), zv.max().round(1), 'median',
      np.median(zv).round(1))
tree = cKDTree(pv)
radius = np.maximum(0.5, 0.06 * zv)
nbrs = tree.query_ball_point(pv, radius)
cnt = np.array([len(n) for n in nbrs])
print('nbr count: median', np.median(cnt), 'p10', np.percentile(cnt, 10),
      'n<4:', int((cnt < 4).sum()), '/', len(cnt))
jump = np.zeros(len(zv))
for i, nb in enumerate(nbrs):
    if len(nb):
        dz = zv[np.asarray(nb)] - zv[i]
        jump[i] = dz.max()
thr = 0.5 + 0.06 * zv
print('jump>thr:', int((jump > thr).sum()))
print('jump percentiles:', np.percentile(jump, [50, 75, 90, 99]).round(2))
