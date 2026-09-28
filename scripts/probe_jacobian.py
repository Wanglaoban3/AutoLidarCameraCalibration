# -*- coding: utf-8 -*-
"""Measure residual sensitivity to pose: is the numeric Jacobian really ~0?"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite     # noqa: E402
from auto_extrinsics.fine import geometry as geo                # noqa: E402
from auto_extrinsics.fine import evidence as ev                 # noqa: E402
from auto_extrinsics.fine.solver import apply_deltas            # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'
rng = np.random.default_rng(0)
ds = NuScenesLite()
recs = ds.frames_of_log(LOG)
recs = recs[::max(1, len(recs) // 12)][:12]

coarse_dir = ROOT / 'extrinsic_recovery' / 'results'
R_ec0 = np.load(coarse_dir / 'camera_R_ec_recovered.npy')
R_le0 = np.load(coarse_dir / 'lidar_R_le_recovered.npy')

frames = []
for k, rec in enumerate(recs[:4]):        # 4 frames enough for diagnosis
    cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
    cal_c, cal_l = ds.calib(cam_sd, 'CAM_FRONT'), ds.calib(lid_sd, 'LIDAR_TOP')
    img = ds.load_image(cam_sd)
    fg = geo.FrameGeom(f'f{k}', cal_c['K'], R_ec0.copy(), cal_c['t_cs'],
                       ds.ego_pose(cam_sd), R_le0.copy(), cal_l['t_cs'],
                       ds.ego_pose(lid_sd), img.shape[:2])
    fg.teed_prob = ev.teed_prob(None, img, 'cpu') if False else None
    frames.append((fg, cal_c, cal_l, img, rec))

# TEED on GPU once
import torch
from auto_extrinsics.fine.evidence import load_teed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teed = load_teed(device)
for fg, cal_c, cal_l, img, rec in frames:
    fg.teed_prob = ev.teed_prob(teed, img, device)
    fg.gm = ev.gradient_magnitude(img)
    fg.dyn_mask = geo.build_dynamic_mask(
        fg, ds.dynamic_boxes_global(rec['sample']))
    fg.p_l = ds.load_sweep(ds.frames_of_log(LOG)[0]['LIDAR_TOP'])  # placeholder

# redo sweeps properly per frame
for (fg, cal_c, cal_l, img, rec) in frames:
    lid_sd = rec['LIDAR_TOP']
    fg.p_l = ds.load_sweep(lid_sd)

samples = []
for fg, *_ in frames:
    s = ev.depth_edge_samples(fg, fg.p_l, fg.T_lc(fg.R_ec0, fg.R_le0),
                              2500, rng)
    s['frame'] = fg
    samples.append(s)
    nrm = np.linalg.norm(s['normals'], axis=1) if len(s['normals']) else []
    ang = np.degrees(np.arctan2(s['normals'][:, 1], s['normals'][:, 0])) \
        if len(s['normals']) else []
    hist, _ = np.histogram(ang, bins=12, range=(-180, 180))
    print(f'{fg.name}: n={len(s["p_idx"])} |n| ok={np.all(nrm <= 1.0)} '
          f'normal-angle hist(30deg bins)={hist.tolist()}')


def residuals(x, half_win=11.1):
    R_le, R_ec = apply_deltas(R_le0, R_ec0, x)
    rs, ws = [], []
    for s in samples:
        fg = s['frame']
        uv, ok = ev.project_samples(fg, fg.p_l, fg.T_lc(R_ec, R_le), s)
        r, w = ev.grad_residuals(fg, uv, s['normals'], half_win=half_win)
        bad = ~np.isfinite(r) | ~ok
        rs.append(np.where(bad, 0.0, r))
        ws.append(np.where(bad, 0.0, w))
    return np.concatenate(rs), np.concatenate(ws)


for pitch_deg in (-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0):
    x = np.zeros(6)
    x[4] = np.radians(pitch_deg)      # cam pitch
    r, w = residuals(x)
    m = w > 0
    cost = float(np.sum(w[m] * r[m] ** 2))
    print(f'cam pitch {pitch_deg:+.1f}deg: cost {cost:10.1f} '
          f'valid {int(m.sum())} mean|r| {np.abs(r[m]).mean() if m.any() else float("nan"):.3f}')

# numeric J columns at x=0
x0 = np.zeros(6)
r0, w0 = residuals(x0)
m0 = w0 > 0
fd = 1e-4
for i in range(6):
    xp = x0.copy(); xp[i] += fd
    xm = x0.copy(); xm[i] -= fd
    rp, _ = residuals(xp)
    rm, _ = residuals(xm)
    col = (rp[m0] - rm[m0]) / (2 * fd)
    print(f'J col {i}: norm {np.linalg.norm(col):10.1f} '
          f'median|.| {np.median(np.abs(col)) if m0.any() else 0:.4f}')
print('PROBE_DONE')
