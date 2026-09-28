# -*- coding: utf-8 -*-
"""Why are marking points mostly 'invalid' under project_stacked?"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auto_extrinsics.data.nuscenes_lite import NuScenesLite      # noqa: E402
from auto_extrinsics.fine import geometry as geo                 # noqa: E402
from auto_extrinsics.fine import evidence as ev                  # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'
ds = NuScenesLite()
recs = ds.frames_of_log(LOG)[::1][:12]
teed = ev.load_teed(
    __import__('torch').device('cuda'
                               if __import__('torch').cuda.is_available()
                               else 'cpu'))
rec = recs[6]
cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
cal_c = ds.calib(cam_sd, 'CAM_FRONT')
cal_l = ds.calib(lid_sd, 'LIDAR_TOP')
img = ds.load_image(cam_sd)
fg = geo.FrameGeom(name='f06', K=cal_c['K'], R_ec=cal_c['R_cs'],
                   t_ec=cal_c['t_cs'], ego_c=ds.ego_pose(cam_sd),
                   R_le=cal_l['R_cs'], t_le=cal_l['t_cs'],
                   ego_l=ds.ego_pose(lid_sd), img_shape=img.shape[:2])
fg.img = img
fg.teed_prob = ev.teed_prob(teed, img, __import__('torch').device(
    'cuda' if __import__('torch').cuda.is_available() else 'cpu'))
fg.gm = ev.gradient_magnitude(img)
fg.dyn_boxes = ds.dynamic_boxes_global(rec['sample'])
fg.dyn_mask = geo.build_dynamic_mask(fg, fg.dyn_boxes)
fg.dt_map = ev.build_dt_map(fg)
sweeps = []
for sd in ds.sweep_history(lid_sd, 10):
    arr = ds.load_sweep(sd, with_intensity=True)
    p = arr[arr[:, 0] > -10.0]
    sweeps.append(dict(p_l=p[:, :3], intensity=p[:, 3],
                       T_eg=ds.ego_pose(sd)))
fg.set_stacked(sweeps)
fg.R_ec0 = cal_c['R_cs'].copy()
fg.R_le0 = cal_l['R_cs'].copy()
fg.mark_ids = ev.marking_points(fg, fg.mark_intensity(), 400)
mk = fg.mark_ids
print(f'mark_ids: {len(mk)}')
p_e = fg.stacked_ego_ref()[mk]
print(f'ego xyz ranges: x [{p_e[:, 0].min():.1f},{p_e[:, 0].max():.1f}] '
      f'y [{p_e[:, 1].min():.1f},{p_e[:, 1].max():.1f}] '
      f'z [{p_e[:, 2].min():.2f},{p_e[:, 2].max():.2f}]')
uv, z, ok = fg.project_stacked(fg.R_ec0, fg.R_le0, idx=mk)
print(f'ok fraction: {ok.sum()}/{len(mk)}')
h, w = fg.shape
inb = ((uv[:, 0] > 0.5) & (uv[:, 0] < w - 1.5) & (uv[:, 1] > 0.5)
       & (uv[:, 1] < h - 1.5))
print(f'in-bounds: {inb.sum()}; depth ok: {((z > 0.5) & (z < 200)).sum()}')
dyn_hit = np.zeros(len(mk), bool)
ui = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
vi = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
dyn_hit = fg.dyn_mask[vi, ui] != 0
print(f'dyn-masked: {dyn_hit.sum()}')
d = ev.bilinear(fg.dt_map, uv[ok, 0], uv[ok, 1])
print(f'DT of ok pts: med {np.nanmedian(d):.2f} px')
