# -*- coding: utf-8 -*-
"""Debug contact-line extraction + GT ridge quality over several frames.

For each probed frame: extract contact lines at the base pose (= GT here),
then measure each line's ridge association at GT (median |offset|, locked
count). A healthy stage shows med|res| ~ 0.5-2 px with most samples
locked; lines off-edge at GT are evidence pollution, not pose error.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auto_extrinsics.data.nuscenes_lite import NuScenesLite      # noqa: E402
from auto_extrinsics.fine import geometry as geo                 # noqa: E402
from auto_extrinsics.fine import lines as fl                     # noqa: E402
from auto_extrinsics.fine import evidence as ev                  # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'
OUT = os.path.join(ROOT, 'outputs', 'm1')
FRAMES = [int(a) for a in sys.argv[1:]] or [0, 2, 4, 6, 8, 10]

import torch                                                      # noqa: E402
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teed = ev.load_teed(device)

ds = NuScenesLite()
recs = ds.frames_of_log(LOG)

meds, totals = [], 0
for k in FRAMES:
    rec = recs[k]
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
    fg.gm = ev.gradient_magnitude(img)
    sweeps = []
    for sd in ds.sweep_history(lid_sd, 10):
        arr = ds.load_sweep(sd, with_intensity=True)
        p = arr[arr[:, 0] > -10.0]
        sweeps.append(dict(p_l=p[:, :3], intensity=p[:, 3],
                           T_eg=ds.ego_pose(sd)))
    fg.set_stacked(sweeps)

    rng = np.random.default_rng(7)
    lines = fl.contact_lines(fg, rng, debug=True)
    totals += len(lines)
    print(f'frame {k}: ground n={np.round(fg.ground_plane[0], 3)} '
          f'd={fg.ground_plane[1]:.2f}, lines: {len(lines)}')
    for info in getattr(fg, 'line_debug', []):
        if 'n' in info:
            print(f'  plane n={np.round(info["n"], 3)} '
                  f'd={info.get("d", float("nan")):.1f} '
                  f'inl={info.get("inliers", "-")} '
                  + (f'span={info["span"]:.1f}m vis={info.get("visible", "-")}'
                     if 'span' in info else '')
                  + (f'  [{info["reason"]}]' if info['reason'] else '  [ok]'))
        else:
            print(f'  [{info["reason"]}]')
    vis = img.copy()
    for li, line in enumerate(lines):
        uv, normals, ok = fl.line_samples_px(fg, line, fg.R_ec0, fg.R_le0)
        targets, w = ev.locate_ridges(fg, uv[ok], normals[ok], 6.0)
        good = w > 0
        d = uv[ok][good] - targets[good]
        nm = normals[ok][good]
        res = d[:, 0] * nm[:, 0] + d[:, 1] * nm[:, 1]
        med = float(np.median(np.abs(res))) if good.any() else float('nan')
        meds.append(med)
        print(f'  line {li}: locked {int(good.sum())}/{int(ok.sum())}, '
              f'med|res| {med:.2f}px  (strong frac '
              f'{np.median(w[good]) if good.any() else 0:.2f})')
        idx_ok = np.nonzero(ok)[0]
        colmap = {0: (0, 200, 0), 1: (0, 165, 255), 2: (0, 0, 255)}
        res_full = np.full(len(good), np.nan)
        res_full[good] = res
        for j in range(len(good)):
            i_ = idx_ok[j]
            r_ = res_full[j]
            if np.isnan(r_):
                col = (128, 128, 128)
            else:
                col = colmap[0 if abs(r_) < 1.5
                             else (1 if abs(r_) < 4 else 2)]
            cv2.circle(vis, (int(uv[i_, 0]), int(uv[i_, 1])), 2, col, -1)
    cv2.imwrite(os.path.join(OUT, f'line_debug_f{k}.png'), vis)

print(f'\ntotal lines {totals} over {len(FRAMES)} frames; '
      f'median med|res| at GT: {np.nanmedian(meds):.2f}px')
print('saved line_debug_f*.png')
