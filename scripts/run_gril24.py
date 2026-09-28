# -*- coding: utf-8 -*-
"""GRIL-Calib intensity-edge fine calibration runner (core: auto_extrinsics
.fine.gril24).

Structure mirrors scripts/run_m1.py: 12 frames of one Boston log, one rigid
mounting rotation error per sensor per run, evidence extracted ONCE at the
base pose and frozen, scipy least_squares (huber 2.5, bounds +-1.5 deg) over
x = [rotvec_le, rotvec_ec] with a holdout publish gate.

Modes
  gt           init = GT (zero-noise stability check: gate must REJECT,
               final within 0.10 deg of init per sensor).
  real_coarse  init = actual coarse-pipeline outputs
               (extrinsic_recovery/results/*_recovered.npy; expected init
               geodesics ~0.507 deg cam / ~0.364 deg lidar).

Usage (from repo root):
  H:/miniconda3/envs/yt/python.exe scripts/run_gril24.py --mode gt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402
from auto_extrinsics import noise as nz                      # noqa: E402
from auto_extrinsics.fine import evidence as ev              # noqa: E402
from auto_extrinsics.fine import gril24 as g24               # noqa: E402
from auto_extrinsics.fine.solver import apply_deltas         # noqa: E402
import run_m1                                                # noqa: E402

N_FRAMES = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='real_coarse',
                    choices=['real_coarse', 'gt'])
    ap.add_argument('--frames', type=int, default=N_FRAMES)
    args = ap.parse_args()

    t_start = time.time()
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = NuScenesLite()
    recs = ds.frames_of_log(run_m1.LOG)
    step = max(1, len(recs) // args.frames)
    recs = recs[::step][:args.frames]
    print(f'{len(recs)} frames from {run_m1.LOG}')

    teed = ev.load_teed(device)
    t0 = time.time()
    frames = run_m1.build_frames(ds, recs, teed, device)
    t_build = time.time() - t0

    cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
    cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')
    R_ec_gt, R_le_gt = cal_c['R_cs'], cal_l['R_cs']

    # ---- init (identical to run_m1) -----------------------------------------
    if args.mode == 'real_coarse':
        coarse_dir = ROOT / 'extrinsic_recovery' / 'results'
        rng_c = np.random.default_rng(7)
        er, ep = (np.radians(rng_c.uniform(-2, 2)) for _ in range(2))
        Rotation.from_euler('xyz', [er, ep, 0]).as_matrix()
        rng_l = np.random.default_rng(11)
        er2, ep2 = (np.radians(rng_l.uniform(-2, 2)) for _ in range(2))
        Rotation.from_euler('xyz', [er2, ep2, 0]).as_matrix()
        R_ec0 = np.load(coarse_dir / 'camera_R_ec_recovered.npy')
        R_le0 = np.load(coarse_dir / 'lidar_R_le_recovered.npy')
        print('init: actual coarse outputs (cam seed 7 / lidar seed 11 '
              'replicated)')
    else:
        R_ec0, R_le0 = R_ec_gt.copy(), R_le_gt.copy()
        print('init: GT (zero-noise stability check: gate must reject)')

    for fg in frames:
        fg.R_ec0 = R_ec0.copy()
        fg.R_le0 = R_le0.copy()

    # ---- frozen evidence: intensity-edge points ------------------------------
    t0 = time.time()
    evs = [g24.extract_intensity_edges(fg) for fg in frames]
    t_extract = time.time() - t0
    per_frame = []
    for k, (fg, e) in enumerate(zip(frames, evs)):
        n_ok, frac = g24.inframe_stats(fg, e['ids'], R_ec0, R_le0)
        per_frame.append(dict(frame=fg.name, n_splat=e['n_splat'],
                              n_canny=e['n_canny'], n_edge=e['n_kept'],
                              n_inframe_init=n_ok, frac_inframe_init=frac))
        print(f'frame {k}: splat {e["n_splat"]} canny {e["n_canny"]} -> '
              f'{e["n_kept"]} edge pts, in-frame at init {n_ok} '
              f'({frac * 100:.1f}%)')
    n_edge = sum(p['n_edge'] for p in per_frame)
    n_in = sum(p['n_inframe_init'] for p in per_frame)
    print(f'total edge points {n_edge}, in-frame at init {n_in} '
          f'({(n_in / max(n_edge, 1)) * 100:.1f}%)')

    pairs = [(fg, e['ids']) for fg, e in zip(frames, evs)]
    train = [p for i, p in enumerate(pairs) if i % 2 == 0
             and len(p[1]) > 0]
    hold = [p for i, p in enumerate(pairs) if i % 2 == 1 and len(p[1]) > 0]
    print(f'train {len(train)} / holdout {len(hold)} frames (odd = holdout)')

    # ---- ground normal + objective -------------------------------------------
    n_e0 = g24.ground_normal_ego([fg for fg, _ in train])
    if n_e0 is not None:
        print(f'ground normal (train, ego, frozen): '
              f'{np.array2string(n_e0, precision=4)}')
    focal_px = float(frames[0].K[0, 0])
    bound = np.radians(g24.BOUND_DEG)

    def fn(x):
        return g24.make_objective(train, R_le0, R_ec0, n_e0, focal_px,
                                  bound_rad=bound)(x)

    med0, p900, vf0 = g24.score_pose(hold, R_le0, R_ec0)
    tm0, tp900, tvf0 = g24.score_pose(train, R_le0, R_ec0)
    print(f'gate holdout @init: DT med {med0:.2f} p90 {p900:.2f} valid '
          f'{vf0:.2f} | train med {tm0:.2f} p90 {tp900:.2f}')

    # ---- optimize --------------------------------------------------------------
    t0 = time.time()
    res = least_squares(fn, np.zeros(6), bounds=(-bound, bound),
                        loss='huber', f_scale=2.5, max_nfev=100,
                        diff_step=1e-6)
    t_opt = time.time() - t0
    x = res.x
    R_le_t, R_ec_t = apply_deltas(R_le0, R_ec0, x)
    run_m1.report('trial', R_le_t, R_ec_t, R_le_gt, R_ec_gt)

    med1, p901, vf1 = g24.score_pose(hold, R_le_t, R_ec_t)
    tm1, tp901, tvf1 = g24.score_pose(train, R_le_t, R_ec_t)
    boundary_hit = bool(np.any(np.isclose(x, -bound, atol=1e-5))
                        or np.any(np.isclose(x, bound, atol=1e-5)))
    print(f'gate holdout: DT med {med0:.2f}->{med1:.2f} p90 '
          f'{p900:.2f}->{p901:.2f} valid {vf0:.2f}->{vf1:.2f} | train med '
          f'{tm0:.2f}->{tm1:.2f} p90 {tp900:.2f}->{tp901:.2f} | nfev '
          f'{res.nfev}' + (' -- BOUNDARY HIT' if boundary_hit else ''))

    keep = bool(res.success and med1 < med0 and p901 < p900
                and vf1 >= 0.8 * vf0 and not boundary_hit)
    print('  -- PUBLISH' if keep
          else '  -- REJECT (holdout not improved / evidence collapse '
               '/ boundary hit)')
    if not keep:
        x = np.zeros(6)
    R_le_f, R_ec_f = apply_deltas(R_le0, R_ec0, x)
    run_m1.report('final', R_le_f, R_ec_f, R_le_gt, R_ec_gt)

    # ---- persist ----------------------------------------------------------------
    out_dir = ROOT / 'outputs' / 'm1'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        mode=args.mode, frames=len(frames),
        method='GRIL-Calib intensity-edge vs TEED-DT port (fine/gril24.py)',
        init=dict(lidar=nz.geodesic_deg(R_le0, R_le_gt),
                  cam=nz.geodesic_deg(R_ec0, R_ec_gt)),
        trial=dict(lidar=nz.geodesic_deg(R_le_t, R_le_gt),
                   cam=nz.geodesic_deg(R_ec_t, R_ec_gt),
                   x_deg=np.degrees(x).tolist()),
        final=dict(lidar=nz.geodesic_deg(R_le_f, R_le_gt),
                   cam=nz.geodesic_deg(R_ec_f, R_ec_gt),
                   x_deg=np.degrees(x).tolist()),
        gate=dict(holdout_median=[med0, med1], holdout_p90=[p900, p901],
                  valid_frac=[vf0, vf1], train_median=[tm0, tm1],
                  train_p90=[tp900, tp901],
                  boundary_hit=boundary_hit, published=keep,
                  optimizer_success=bool(res.success), nfev=int(res.nfev)),
        edges=dict(total=n_edge, total_inframe_init=n_in,
                   frac_inframe_init=n_in / max(n_edge, 1),
                   per_frame=per_frame),
        ground_normal_ego=(n_e0.tolist() if n_e0 is not None else None),
        ground_prior_weight_px=g24.GROUND_W * focal_px,
        timing=dict(build_s=t_build, extract_s=t_extract, optimize_s=t_opt,
                    total_s=time.time() - t_start))
    with open(out_dir / 'gril24_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('saved:', out_dir / 'gril24_summary.json')
    print(f'timing: build {t_build:.1f}s extract {t_extract:.1f}s '
          f'optimize {t_opt:.1f}s total {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
