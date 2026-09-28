# -*- coding: utf-8 -*-
"""6-surround-camera probe: how much marking evidence does each camera
see? The reference recipe (AutoLidarCameraCalibration) refines with all
6 cameras; our single-front-camera port starved (5% in-frame). Numbers
here decide whether the faithful multi-camera port can work.

Per keyframe: marking rim points on the FULL ego disk (front_only=False),
then per camera the in-frame fraction at GT and -- for the first two
keyframes -- the TEED-DT quality of the in-frame points."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402
from auto_extrinsics.fine import geometry as geo             # noqa: E402
from auto_extrinsics.fine import evidence as ev              # noqa: E402
import run_m1                                                # noqa: E402

N_KEY = 6
TEED_KEYS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teed = ev.load_teed(device)
ds = NuScenesLite()
recs = ds.frames_of_log_multi(run_m1.LOG)
step = max(1, len(recs) // N_KEY)
recs = recs[::step][:N_KEY]
print(f'{len(recs)} keyframes x {len(ds.CAM_CHANNELS)} cameras')

per_cam_n = {c: 0 for c in ds.CAM_CHANNELS}
per_cam_ok = {c: 0 for c in ds.CAM_CHANNELS}
per_cam_dt = {c: [] for c in ds.CAM_CHANNELS}
n_marks_total = 0

for k, rec in enumerate(recs):
    lid_sd = rec['LIDAR_TOP']
    sweeps = []
    for sd in ds.sweep_history(lid_sd, run_m1.SWEEPS):
        arr = ds.load_sweep(sd, with_intensity=True)
        p = arr[arr[:, 0] > -10.0]
        sweeps.append(dict(p_l=p[:, :3], intensity=p[:, 3],
                           T_eg=ds.ego_pose(sd)))
    fgs = {}
    for ch in ds.CAM_CHANNELS:
        cam_sd = rec[ch]
        cal = ds.calib(cam_sd, ch)
        img = ds.load_image(cam_sd)
        fg = geo.FrameGeom(
            name=f'f{k:02d}_{ch}', K=cal['K'], R_ec=cal['R_cs'],
            t_ec=cal['t_cs'], ego_c=ds.ego_pose(cam_sd),
            R_le=ds.calib(lid_sd, 'LIDAR_TOP')['R_cs'],
            t_le=ds.calib(lid_sd, 'LIDAR_TOP')['t_cs'],
            ego_l=ds.ego_pose(lid_sd), img_shape=img.shape[:2])
        fg.img = img
        fg.dyn_boxes = ds.dynamic_boxes_global(rec['sample'])
        fg.dyn_mask = geo.build_dynamic_mask(fg, fg.dyn_boxes)
        if k < TEED_KEYS:
            fg.teed_prob = ev.teed_prob(teed, img, device)
            fg.dt_map = ev.build_dt_map(fg)
        fgs[ch] = fg
    front = fgs['CAM_FRONT']
    front.set_stacked(sweeps)
    for ch, fg in fgs.items():
        if ch != 'CAM_FRONT':
            # all FrameGeom share the same stacked cloud
            fg.stack_p = front.stack_p
            fg.stack_sid = front.stack_sid
            fg.stack_G = front.stack_G
            fg.stack_intensity = front.stack_intensity
    marks = ev.marking_points(front, front.stack_intensity,
                              front_only=False)
    n_marks_total += len(marks)
    R_ec_f = fgs['CAM_FRONT'].R_ec0
    R_le_f = fgs['CAM_FRONT'].R_le0
    line = f'f{k:02d}: marks {len(marks):4d} |'
    for ch, fg in fgs.items():
        uv, z, ok = fg.project_stacked(R_ec_f, R_le_f, idx=marks)
        per_cam_n[ch] += len(marks)
        per_cam_ok[ch] += int(ok.sum())
        msg = f' {ch[4:]:11s} {int(ok.sum()):4d}/{len(marks):4d}'
        if k < TEED_KEYS and ok.any():
            d = ev.bilinear(fg.dt_map, uv[ok, 0], uv[ok, 1])
            d = d[np.isfinite(d)]
            per_cam_dt[ch].append(d)
            msg += (f' dt {np.median(d):5.1f}px <5px {np.mean(d < 5):.2f}')
        line += msg + ' |'
    print(line)

print('pooled in-frame fractions at GT:')
for ch in ds.CAM_CHANNELS:
    frac = per_cam_ok[ch] / max(per_cam_n[ch], 1)
    dts = np.concatenate(per_cam_dt[ch]) if per_cam_dt[ch] \
        else np.array([np.nan])
    print(f'  {ch:16s} {frac:6.2f}  '
          f'dt med {np.nanmedian(dts):5.1f}px <5px '
          f'{np.nanmean(dts < 5):.2f}')
print(f'total marking rim points: {n_marks_total} '
      f'({n_marks_total / max(len(recs), 1):.0f}/keyframe)')
