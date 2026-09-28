# -*- coding: utf-8 -*-
"""LT12 experiment: Levinson & Thrun (RSS 2012) correlation objective.

Ported objective (see auto_extrinsics/fine/lt12.py for the exact form):
maximize the pooled correlation, over the projected pool of stacked-cloud
points, between the FROZEN depth-discontinuity indicator (1 for the
near-side depth-jump points of evidence.depth_edge_samples_dense,
extracted ONCE at the base pose) and the Gaussian-blurred Sobel gradient
magnitude sampled at the point projection; the indicator is z-scored
once over the pool, the gradient map once over the whole image
(median/MAD, clip -- robust-stats choice documented in lt12.py), so both
normalizations are pose-independent. Optimizer: 2-stage bounded
Nelder-Mead in a +-1.5 deg box per DoF (6 DoF = rotvec_le + rotvec_ec).
Publish gate (holdout = odd-index frames): holdout correlation improves
AND valid fraction >= 0.8x AND no DoF on its bound.

Modes
  real_coarse     init from the ACTUAL coarse-pipeline outputs
                  (extrinsic_recovery/results/*_recovered.npy), as run_m1.
  gt              init = GT (zero-noise stability check: the gate must
                  reject; final within 0.10 deg of init per sensor).

Usage (from repo root):
  python scripts/run_lt12.py --mode gt
  python scripts/run_lt12.py --mode real_coarse
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402
from auto_extrinsics import noise as nz                      # noqa: E402
from auto_extrinsics.fine import evidence as ev              # noqa: E402
from auto_extrinsics.fine import lt12                        # noqa: E402
import run_m1                                                # noqa: E402

LOG = run_m1.LOG
N_FRAMES = run_m1.N_FRAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='real_coarse',
                    choices=['gt', 'real_coarse'])
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_start = time.time()
    ds = NuScenesLite()
    recs = ds.frames_of_log(LOG)
    step = max(1, len(recs) // N_FRAMES)
    recs = recs[::step][:N_FRAMES]
    print(f'{len(recs)} frames from {LOG}')

    teed = ev.load_teed(device)
    frames = run_m1.build_frames(ds, recs, teed, device)

    cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
    cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')
    R_ec_gt, R_le_gt = cal_c['R_cs'], cal_l['R_cs']

    # ---- init ---------------------------------------------------------------
    if args.mode == 'real_coarse':
        coarse_dir = ROOT / 'extrinsic_recovery' / 'results'
        R_ec0 = np.load(coarse_dir / 'camera_R_ec_recovered.npy')
        R_le0 = np.load(coarse_dir / 'lidar_R_le_recovered.npy')
        print('init: actual coarse outputs')
    else:
        R_ec0, R_le0 = R_ec_gt.copy(), R_le_gt.copy()
        print('init: GT (zero-noise stability check: gate must reject)')
    for fg in frames:
        fg.R_ec0 = R_ec0.copy()
        fg.R_le0 = R_le0.copy()
    run_m1.report('init', R_le0, R_ec0, R_le_gt, R_ec_gt)

    # ---- frozen evidence: edge set + pool, extracted ONCE at the base pose --
    # near_side=True: the FOREGROUND point owns the photometric silhouette
    # (measured in this repo: far-side points sit ~17 px from any strong
    # edge on bare road/wall, which zeroes the correlation at GT and leaves
    # the optimizer nothing but texture overfitting -- observed in the
    # first gt run: train corr 0.004 -> 0.062 at a 0.57 deg drift, holdout
    # worse). lt12.build_pool restricts the random subsample to points
    # in-frame at the base pose.
    rng_e = np.random.default_rng(0)
    pools = []
    n_edges = 0
    n_pool = 0
    for fg in frames:
        fg.gm_z = lt12.edge_map_z(fg)   # pose-independent camera-edge field
        edge = ev.depth_edge_samples_dense(fg, R_ec0, R_le0,
                                           max_samples=lt12.EDGE_CAP,
                                           rng=rng_e, near_side=True)
        idx, az = lt12.build_pool(fg, edge, R_ec0, R_le0, seed=0)
        pools.append((idx, az))
        n_edges += int((az > 0).sum())
        n_pool += len(idx)
    print(f'evidence frozen at base pose: {n_pool} pool pts '
          f'({n_pool / len(frames):.0f}/frame), {n_edges} laser-edge pts '
          f'({100.0 * n_edges / max(n_pool, 1):.1f}%)')

    train = [fg for i, fg in enumerate(frames) if i % 2 == 0]
    hold = [fg for i, fg in enumerate(frames) if i % 2 == 1]
    pools_train = [pools[i] for i in range(len(frames)) if i % 2 == 0]
    pools_hold = [pools[i] for i in range(len(frames)) if i % 2 == 1]

    # ---- optimize on train frames -------------------------------------------
    t0 = time.time()
    fn = lt12.make_objective(train, pools_train)
    x = lt12.optimize(fn)
    t_opt = time.time() - t0
    x_trial = np.asarray(x, float).copy()
    R_le_t, R_ec_t = _apply(R_le0, R_ec0, x)
    run_m1.report('trial', R_le_t, R_ec_t, R_le_gt, R_ec_gt)

    # ---- holdout publish gate ------------------------------------------------
    keep, gate = lt12.publish_gate(hold, pools_hold, R_le0, R_ec0, x)
    c_tr0, vf_tr0 = lt12.pooled(train, R_le0, R_ec0, pools_train)
    c_tr1, vf_tr1 = lt12.pooled(train, R_le_t, R_ec_t, pools_train)
    print(f"gate holdout: corr {gate['corr_init']:.5f} -> "
          f"{gate['corr_final']:.5f} | valid {gate['valid_frac_init']:.3f} "
          f"-> {gate['valid_frac_final']:.3f}"
          f" | on_bound {gate['on_bound']}")
    print(f'train:        corr {c_tr0:.5f} -> {c_tr1:.5f} | valid '
          f'{vf_tr0:.3f} -> {vf_tr1:.3f}')
    print('  -- PUBLISH' if keep
          else '  -- REJECT (holdout not improved / valid collapse / '
               'bound hit)')

    if not keep:
        x = np.zeros(6)
    R_le_f, R_ec_f = _apply(R_le0, R_ec0, x)
    run_m1.report('final', R_le_f, R_ec_f, R_le_gt, R_ec_gt)

    # ---- persist --------------------------------------------------------------
    out_dir = ROOT / 'outputs' / 'm1'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        method='lt12', mode=args.mode, frames=len(frames),
        n_pool=n_pool, n_edge=n_edges,
        init=dict(lidar=nz.geodesic_deg(R_le0, R_le_gt),
                  cam=nz.geodesic_deg(R_ec0, R_ec_gt)),
        trial=dict(lidar=nz.geodesic_deg(R_le_t, R_le_gt),
                   cam=nz.geodesic_deg(R_ec_t, R_ec_gt),
                   x_deg=x.tolist() if keep else
                   lt12_xdeg(R_le_t, R_ec_t, R_le0, R_ec0)),
        final=dict(lidar=nz.geodesic_deg(R_le_f, R_le_gt),
                   cam=nz.geodesic_deg(R_ec_f, R_ec_gt)),
        published=bool(keep), gate=gate,
        train=dict(corr_init=c_tr0, corr_final=c_tr1,
                   valid_init=vf_tr0, valid_final=vf_tr1),
        timing=dict(optimize_s=t_opt, total_s=time.time() - t_start))
    with open(out_dir / 'lt12_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('saved:', out_dir / 'lt12_summary.json')
    print(f'timing: optimize {t_opt:.1f} s, total '
          f'{time.time() - t_start:.1f} s')


def _apply(R_le0, R_ec0, x_deg):
    from auto_extrinsics.fine.solver import apply_deltas
    return apply_deltas(R_le0, R_ec0, np.radians(np.asarray(x_deg, float)))


def lt12_xdeg(R_le, R_ec, R_le0, R_ec0):
    """Recover the applied per-DoF rotvec (deg) from trial poses (for the
    rejected-trial record)."""
    from scipy.spatial.transform import Rotation
    dle = Rotation.from_matrix(R_le @ R_le0.T).as_rotvec()
    dec = Rotation.from_matrix(R_ec @ R_ec0.T).as_rotvec()
    return np.concatenate([dle, dec]).tolist()


if __name__ == '__main__':
    main()
