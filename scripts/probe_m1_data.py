# -*- coding: utf-8 -*-
"""Smoke probe: data layer + projection chain + depth-edge extraction with
GT poses (no TEED, no solver). Expect most sweep points in-FOV for CAM_FRONT
(32-line, ~70 deg HFOV) and a healthy depth-edge sample count."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite     # noqa: E402
from auto_extrinsics.fine import geometry as geo                # noqa: E402
from auto_extrinsics.fine import evidence as ev                 # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'
rng = np.random.default_rng(0)
ds = NuScenesLite()
recs = ds.frames_of_log(LOG)
print(f'frames in log: {len(recs)}')
step = max(1, len(recs) // 12)
recs = recs[::step][:12]
for k, rec in enumerate(recs[:4]):
    cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
    cal_c, cal_l = ds.calib(cam_sd, 'CAM_FRONT'), ds.calib(lid_sd, 'LIDAR_TOP')
    img = ds.load_image(cam_sd)
    fg = geo.FrameGeom('f', cal_c['K'], cal_c['R_cs'], cal_c['t_cs'],
                       ds.ego_pose(cam_sd), cal_l['R_cs'], cal_l['t_cs'],
                       ds.ego_pose(lid_sd), img.shape[:2])
    fg.dyn_mask = geo.build_dynamic_mask(fg, ds.dynamic_boxes_global(rec['sample']))
    p_l = ds.load_sweep(lid_sd)
    T_lc = fg.T_lc(cal_c['R_cs'], cal_l['R_cs'])
    uv, z, ok = fg.project(p_l, T_lc)
    s = ev.depth_edge_samples(fg, p_l, T_lc, 2500, rng)
    t = rec['sample']['timestamp']
    dt_ms = (cam_sd['timestamp'] - lid_sd['timestamp']) / 1e3
    print(f'frame {k}: t={t} cam-lidar dt={dt_ms:+.1f}ms sweep={len(p_l)} '
          f'in_fov={int(ok.sum())} ({100*ok.mean():.0f}%) '
          f'edge_samples={len(s["p_idx"])} dyn_mask='
          f'{"yes" if fg.dyn_mask is not None else "no"}')
print('PROBE_DONE')
