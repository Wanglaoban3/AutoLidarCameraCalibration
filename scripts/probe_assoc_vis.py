# -*- coding: utf-8 -*-
"""Visualize pass-A associations: sample uv -> target uv arrows on the image,
plus the offset distribution along normals."""
import sys
from pathlib import Path

import cv2
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
rec = recs[0]
cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
cal_c, cal_l = ds.calib(cam_sd, 'CAM_FRONT'), ds.calib(lid_sd, 'LIDAR_TOP')
img = ds.load_image(cam_sd)
coarse_dir = ROOT / 'extrinsic_recovery' / 'results'
R_ec0 = np.load(coarse_dir / 'camera_R_ec_recovered.npy')
R_le0 = np.load(coarse_dir / 'lidar_R_le_recovered.npy')
fg = geo.FrameGeom('f0', cal_c['K'], R_ec0, cal_c['t_cs'],
                   ds.ego_pose(cam_sd), R_le0, cal_l['t_cs'],
                   ds.ego_pose(lid_sd), img.shape[:2])
fg.teed_prob = ev.teed_prob(None, img, 'cpu') if False else None
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teed = ev.load_teed(device)
fg.teed_prob = ev.teed_prob(teed, img, device)
fg.gm = ev.gradient_magnitude(img)
fg.dyn_mask = geo.build_dynamic_mask(
    fg, ds.dynamic_boxes_global(rec['sample']))
fg.p_l = ds.load_sweep(lid_sd)

s = ev.depth_edge_samples(fg, fg.p_l, fg.T_lc(R_ec0, R_le0), 2500, rng)
uv, ok = ev.project_samples(fg, fg.p_l, fg.T_lc(R_ec0, R_le0), s)
half_win = 14.4
targets, w = ev.locate_ridges(fg, uv, s['normals'], half_win)
good = w > 0
t_off = ev.offset_to_targets(uv, s['normals'], targets)
print(f'associated {int(good.sum())}/{len(uv)}')
print('t* stats: mean %.2f median %.2f std %.2f' % (
    t_off[good].mean(), np.median(t_off[good]), t_off[good].std()))
hist, edges = np.histogram(t_off[good], bins=16, range=(-15, 15))
print('t* hist [-15..15]:', hist.tolist())

vis = img.copy()
sub = np.nonzero(good)[0]
for i in sub[:: max(1, len(sub) // 120)]:
    p = (int(round(uv[i, 0])), int(round(uv[i, 1])))
    q = (int(round(targets[i, 0])), int(round(targets[i, 1])))
    cv2.circle(vis, p, 3, (255, 0, 0), -1)          # blue: sample
    cv2.arrowedLine(vis, p, q, (0, 0, 255), 1, tipLength=0.3)
    cv2.circle(vis, q, 2, (0, 255, 0), -1)          # green: target
out = ROOT / 'outputs' / 'm1' / 'assoc_debug.png'
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), vis)
print('saved', out)
