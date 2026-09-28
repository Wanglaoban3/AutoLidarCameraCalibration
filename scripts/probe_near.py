# -*- coding: utf-8 -*-
"""Per-frame near-side sample DT quality across the whole M1 log."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auto_extrinsics.data.nuscenes_lite import NuScenesLite      # noqa: E402
from auto_extrinsics.fine import geometry as geo                 # noqa: E402
from auto_extrinsics.fine import evidence as ev                  # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teed = ev.load_teed(device)
ds = NuScenesLite()
recs = ds.frames_of_log(LOG)
rng = np.random.default_rng(0)
alld = []
for k, rec in enumerate(recs):
    cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
    cal_c = ds.calib(cam_sd, 'CAM_FRONT')
    cal_l = ds.calib(lid_sd, 'LIDAR_TOP')
    img = ds.load_image(cam_sd)
    fg = geo.FrameGeom(name=f'f{k:02d}', K=cal_c['K'], R_ec=cal_c['R_cs'],
                       t_ec=cal_c['t_cs'], ego_c=ds.ego_pose(cam_sd),
                       R_le=cal_l['R_cs'], t_le=cal_l['t_cs'],
                       ego_l=ds.ego_pose(lid_sd), img_shape=img.shape[:2])
    fg.img = img
    fg.teed_prob = ev.teed_prob(teed, img, device)
    fg.dt_map = ev.build_dt_map(fg)
    sweeps = []
    for sd in ds.sweep_history(lid_sd, 10):
        arr = ds.load_sweep(sd, with_intensity=True)
        p = arr[arr[:, 0] > -10.0]
        sweeps.append(dict(p_l=p[:, :3], intensity=p[:, 3],
                           T_eg=ds.ego_pose(sd)))
    fg.set_stacked(sweeps)
    s = ev.depth_edge_samples_dense(fg, cal_c['R_cs'], cal_l['R_cs'],
                                    max_samples=2500, rng=rng, near_side=True)
    if len(s['p_idx']) == 0:
        print(f'frame {k}: no samples')
        continue
    uv, ok = ev.project_stacked_samples(fg, cal_c['R_cs'], cal_l['R_cs'], s)
    d = ev.bilinear(fg.dt_map, uv[ok, 0], uv[ok, 1])
    d = d[np.isfinite(d)]
    alld.append(d)
    print(f'frame {k}: n={len(s["p_idx"])} ok={int(ok.sum())} '
          f'DT med {np.median(d):.2f} p90 {np.percentile(d, 90):.2f} '
          f'frac<5px {np.mean(d < 5):.2f}')
a = np.concatenate(alld)
print(f'pooled: med {np.median(a):.2f} p90 {np.percentile(a, 90):.2f} '
      f'frac<5px {np.mean(a < 5):.2f} frac<15px {np.mean(a < 15):.2f}')
