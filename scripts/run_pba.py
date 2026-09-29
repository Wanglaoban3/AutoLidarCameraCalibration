# -*- coding: utf-8 -*-
"""PBA experiment: PBACalib port (arXiv:2308.12629) -- plane-constrained
multi-frame bundle adjustment over the two mounting rotations.

Protocol (mirrors scripts/run_m1.py):
  real_coarse   init from the ACTUAL coarse-pipeline outputs
                (extrinsic_recovery/results/*_recovered.npy;
                published 0.507 / 0.364 deg geodesic init).
  gt            init = GT. HARD GATE: must end within 0.10 deg/sensor
                of init (the holdout publish gate must reject
                noise-level improvements before real_coarse is trusted).
  probe         evidence census only (features, tracks, planes,
                membership at both bands) -- no solve.

Mechanics and paper->port adaptation map: auto_extrinsics/fine/pba.py.
Association = wide 0.30 m membership round, solve, re-associate at the
paper's 0.15 m band, solve again (frozen association per round, exactly
one pose-independent membership test per round in the shared metric
frame). ORACLE: triangulation walks nuScenes ego_pose (printed caveat).

Usage (from repo root):
  python scripts/run_pba.py --mode gt
  python scripts/run_pba.py --mode real_coarse
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_m1 import report  # noqa: E402  (reuse the repo's report format)

from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402
from auto_extrinsics import noise as nz                      # noqa: E402
from auto_extrinsics.fine import geometry as geo             # noqa: E402
from auto_extrinsics.fine import pba                         # noqa: E402
from auto_extrinsics.fine.solver import apply_deltas         # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'   # scene-0103, Boston row houses
N_FRAMES = 12                            # 2 Hz keyframes
STEP = 2                                 # 1 s keyframe spacing (~3 m ego)
SWEEPS = 10                              # 0.5 s of 20 Hz sweeps per frame
HOLDOUT_FRAC = 0.3
GT = {}                                  # filled in main(): factory calib


def build_frames(ds, recs):
    """Slim version of run_m1.build_frames: no TEED/DT maps (not needed
    here); stacked cloud per frame via the repo's sweep bridge; dyn_mask
    placeholder kept (annotation-based, same discipline as run_m1)."""
    frames = []
    for k, rec in enumerate(recs):
        cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
        cal_c = ds.calib(cam_sd, 'CAM_FRONT')
        cal_l = ds.calib(lid_sd, 'LIDAR_TOP')
        img = ds.load_image(cam_sd)
        fg = geo.FrameGeom(
            name=f'f{k:02d}', K=cal_c['K'],
            R_ec=cal_c['R_cs'], t_ec=cal_c['t_cs'],
            ego_c=ds.ego_pose(cam_sd),
            R_le=cal_l['R_cs'], t_le=cal_l['t_cs'],
            ego_l=ds.ego_pose(lid_sd), img_shape=img.shape[:2])
        fg.img = img
        fg.dyn_boxes = ds.dynamic_boxes_global(rec['sample'])
        fg.dyn_mask = geo.build_dynamic_mask(fg, fg.dyn_boxes)
        sweeps = []
        for sd in ds.sweep_history(lid_sd, SWEEPS):
            arr = ds.load_sweep(sd, with_intensity=True)
            p = arr[arr[:, 0] > -10.0]        # drop the rear hemisphere
            sweeps.append(dict(p_l=p[:, :3], intensity=p[:, 3],
                               T_eg=ds.ego_pose(sd)))
        fg.p_l = sweeps[-1]['p_l']
        fg.set_stacked(sweeps)
        frames.append(fg)
        print(f'frame {k}: stacked {len(fg.stack_p)} pts '
              f'({len(sweeps)} sweeps)')
    return frames


def membership(tracks, cloud, tree, planes, band, T_ge_ref, debug=False):
    U_own = np.asarray([t['u'] for t in tracks], float)
    # map every track into the scene reference ego:
    # B_i = inv(T_ge_ref) @ T_ge_i  (ego(track's ref) -> ego(scene ref))
    T = np.asarray([t['T_ge'] for t in tracks], float)
    B = np.linalg.inv(T_ge_ref)[None] @ T
    U = np.einsum('nij,ni->nj', B[:, :3, :3], U_own) + B[:, :3, 3]
    pk, dist, plan, w = pba.plane_membership(U, cloud, tree, planes, band,
                                             debug=debug)
    n_plan = int((plan > 0).sum())
    n_cons = int((pk >= 0).sum())
    per = [int((pk == k).sum()) for k in range(len(planes))]
    print(f'membership @ band {band:.2f} m: {n_plan} planarity-eligible, '
          f'{n_cons} plane-constrained tracks '
          f'({100.0 * n_cons / max(len(U), 1):.0f}% of {len(U)}); '
          f'per plane {per}')
    return U_own, pk, w, plan, n_plan, n_cons, per


def solve_round(pb, plan, band, x_in, rng_gate, tag, sel_pre=None,
                to_full=None):
    """One frozen-association round: holdout-gated bounded Huber solve.
    to_full maps the solve-space x to the 6-dim apply_deltas state
    (FixedLidarPBA solves 3-dim with the lidar rotvec pinned). Returns
    (x, info) in SOLVE space."""
    con = (plan > 0) if plan is not None else None   # noqa: F841 (unused
    # legacy hook: association is already band-filtered upstream)
    if sel_pre is not None:
        con &= sel_pre
    if band is not None:
        # plane_membership already band-filtered the assignment; the
        # eligible set is everything that passed the covariance test
        pass
    idx = np.nonzero(pb.pk >= 0)[0]
    gate_sel = idx[rng_gate.random(len(idx)) < HOLDOUT_FRAC]
    train_sel = idx[rng_gate.random(len(idx)) >= HOLDOUT_FRAC]
    med0, p900, n0 = pb.gate(x_in, gate_sel)
    t0 = time.time()
    res = pb.solve(x_in, train_sel)
    x_t = res.x
    x_f = to_full(x_t) if to_full is not None else x_t
    R_le_t, R_ec_t = apply_deltas(pb.R_le0, pb.R_ec0, x_f)
    report(f'{tag} trial', R_le_t, R_ec_t, GT['R_le'], GT['R_ec'])
    med1, p901, n1 = pb.gate(x_t, gate_sel)
    boundary = bool(np.any(np.isclose(x_t, -np.radians(pba.BOX_DEG),
                                      atol=1e-6))
                    or np.any(np.isclose(x_t, np.radians(pba.BOX_DEG),
                                         atol=1e-6)))
    publish = bool(med0 is not None and med1 is not None
                   and med1 < pba.PUBLISH_RATIO * med0
                   and p901 < pba.PUBLISH_P90 * p900
                   and (med0 - med1) > pba.PUBLISH_FLOOR
                   and not boundary)
    print(f'{tag}: train {len(train_sel)} / holdout {len(gate_sel)} | '
          f'gate med {med0:.3f}->{med1:.3f} m p90 {p900:.3f}->'
          f'{p901:.3f} m (n={n1})'
          + (' -- PUBLISH' if publish else
             ' -- REJECTED (holdout not improved / floor / boundary)'))
    print(f'  solve {time.time() - t0:.1f}s, cost {res.cost:.3f}, '
          f'status {res.status}')
    info = dict(band_m=band, n_train=int(len(train_sel)),
                n_holdout=int(len(gate_sel)),
                med0=med0, med1=med1, p900=p900, p901=p901,
                publish=publish, boundary=boundary,
                x_le_deg=np.degrees(x_f[:3]).tolist(),
                x_ec_deg=np.degrees(x_f[3:]).tolist(),
                sec=round(time.time() - t0, 1))
    return (x_t if publish else x_in), info


def write_summary(out_path, mode, summary):
    data = {}
    if out_path.exists():
        try:
            with open(out_path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
    data[mode] = summary
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print('saved:', out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='gt',
                    choices=['probe', 'gt', 'real_coarse'])
    ap.add_argument('--frames', type=int, default=N_FRAMES)
    ap.add_argument('--step', type=int, default=STEP,
                    help='keyframe stride (2 = 1 s spacing)')
    args = ap.parse_args()

    t_start = time.time()
    ds = NuScenesLite()
    recs = ds.frames_of_log(LOG)
    recs = recs[::args.step][:args.frames]
    print(f'{len(recs)} keyframes from {LOG} (stride {args.step})')
    print(pba.ORACLE_NOTE)

    cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
    cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')
    GT.update(R_ec=cal_c['R_cs'], R_le=cal_l['R_cs'])

    frames = build_frames(ds, recs)
    ref_idx = len(frames) // 2
    fg_ref = frames[ref_idx]

    # ---- init --------------------------------------------------------------
    if args.mode == 'real_coarse':
        coarse_dir = ROOT / 'extrinsic_recovery' / 'results'
        R_ec0 = np.load(coarse_dir / 'camera_R_ec_recovered.npy')
        R_le0 = np.load(coarse_dir / 'lidar_R_le_recovered.npy')
        print('init: actual coarse outputs')
    else:
        R_ec0, R_le0 = GT['R_ec'].copy(), GT['R_le'].copy()
        print('init: GT' if args.mode == 'gt' else 'init: GT (probe)')
    report('init', R_le0, R_ec0, GT['R_le'], GT['R_ec'])
    for fg in frames:
        fg.R_ec0 = R_ec0.copy()
        fg.R_le0 = R_le0.copy()

    # ---- lidar facades (per frame, bridged to reference ego, merged) --------
    t0 = time.time()
    rng_p = np.random.default_rng(7)
    planes, per_frame_planes, cloud = pba.extract_facades(
        frames, ref_idx, rng_p)
    print(f'{len(planes)} merged constraint planes '
          f'({len(per_frame_planes)} per-frame) on {len(cloud)} pts '
          f'({time.time() - t0:.1f}s):')
    for k, (n, d, c) in enumerate(planes):
        print(f'  plane {k}: n={np.array2string(n, precision=3)} '
              f'd={d:+.1f} m inliers={c}')
    if len(planes) == 0:
        print('no facades -- abort')
        return

    # ---- features + tracks (multi-reference stars) ---------------------------
    T_ge_ref = fg_ref.T_ge_c
    masks, dms, feats_ref, feats_full = [], [], [], []
    for k, fg in enumerate(frames):
        B = np.linalg.inv(fg.T_ge_c) @ T_ge_ref   # ego(ref) -> ego(k)
        planes_k = [(n, d, inl @ B[:3, :3].T + B[:3, 3])
                    for n, d, inl in per_frame_planes]
        m_k = pba.facade_mask(fg, planes_k)
        masks.append(m_k)
        dms.append(pba.build_depth_map(fg)[0])
        feats_ref.append(pba.detect_features(fg.img, mask=m_k))
        feats_full.append(pba.detect_features(fg.img))
    print('facade mask coverage: '
          + ' '.join(f'{m.mean() * 100:.0f}' for m in masks)
          + ' %; mask features '
          + ' '.join(str(len(f[0][0])) for f in feats_ref))
    t0 = time.time()
    tracks = pba.build_tracks_all(frames, masks, dms, feats_ref,
                                  feats_full)
    n_tracks = len(tracks)
    print(f'tracks: {n_tracks} multi-view 3D pts '
          f'({time.time() - t0:.1f}s)')
    if n_tracks == 0:
        print('no triangulated tracks -- abort')
        return

    g = pba.ground_plane_ego(cloud)
    gmask = np.abs((cloud - g[1]) @ g[0]) > pba.GROUND_MARGIN
    tree_f = cKDTree(cloud[gmask])
    U, pk, w, plan, n_plan, n_cons, per = membership(
        tracks, cloud[gmask], tree_f, planes, pba.BAND_WIDE, T_ge_ref,
        debug=True)
    counts = dict(n_tracks=int(n_tracks),
                  median_views=int(np.median([t['n_views']
                                              for t in tracks]))
                  if tracks else 0,
                  n_planarity=n_plan, n_constrained=int(n_cons),
                  per_plane=per)
    if n_cons < 200:
        print(f'STARVED: {n_cons} constrained tracks < 200 -- stopping '
              f'(no solve); reporting starvation numbers.')
        write_summary(ROOT / 'outputs' / 'm1' / 'pba_summary.json',
                      args.mode,
                      dict(mode=args.mode, frames=len(frames),
                           ref_idx=ref_idx, starved=True,
                           planes=[dict(n=n.tolist(), d=float(d),
                                        inliers=int(c))
                                   for n, d, c in planes],
                           n_cloud=int(len(cloud)), **counts,
                           sec=round(time.time() - t_start, 1)))
        return

    pb = pba.PBAResidual(
        U, pk, w, planes, fg_ref.T_ge_c, fg_ref.t_le, fg_ref.t_ec,
        [t['T_ge'] for t in tracks], R_le0, R_ec0)

    if args.mode == 'probe':
        write_summary(ROOT / 'outputs' / 'm1' / 'pba_summary.json',
                      'probe',
                      dict(mode='probe', frames=len(frames),
                           ref_idx=ref_idx,
                           planes=[dict(n=n.tolist(), d=float(d),
                                        inliers=int(c))
                                   for n, d, c in planes],
                           n_cloud=int(len(cloud)),
                           views=[int(t['n_views']) for t in tracks],
                           dof_sigma_deg=None, **counts,
                           sec=round(time.time() - t_start, 1)))
        print(f'probe done in {time.time() - t_start:.0f}s')
        return

    # ---- round 1 (wide band) + round 2 (paper band) --------------------------
    rng_gate = np.random.default_rng(123)
    x = np.zeros(6)
    x, info1 = solve_round(pb, plan, pba.BAND_WIDE, x, rng_gate, 'round1')
    R_le1, R_ec1 = apply_deltas(R_le0, R_ec0, x)
    report('after round1', R_le1, R_ec1, GT['R_le'], GT['R_ec'])

    # re-associate at the paper band (pose-independent test, tighter band;
    # assignment redone ONCE, then frozen for the round). The 0.15 m band
    # presumes the paper's BA-refined structure quality; on two-view-plus
    # mini-sweeps structure noise is ~0.7 m median, so if the paper band
    # starves we keep the round-1 association and skip the re-solve
    # (documented noise-floor adaptation).
    U2, pk2, w2, plan2, n_plan2, n_cons2, per2 = membership(
        tracks, cloud[gmask], tree_f, planes, pba.BAND_PAPER, T_ge_ref)
    if n_cons2 >= 60:
        pb.pk, pb.sw = pk2, w2
        x, info2 = solve_round(pb, plan2, pba.BAND_PAPER, x, rng_gate,
                               'round2')
    else:
        print(f'round2: paper band {pba.BAND_PAPER} m keeps only '
              f'{n_cons2} tracks (< 60) -- kept round-1 association, '
              f'no re-solve')
        info2 = dict(band_m=pba.BAND_PAPER, skipped_starved=True,
                     n_constrained=int(n_cons2))
    R_le_f, R_ec_f = apply_deltas(R_le0, R_ec0, x)
    report('final', R_le_f, R_ec_f, GT['R_le'], GT['R_ec'])

    dof_sig = pb.dof_sigma_deg(x, np.nonzero(pb.pk >= 0)[0])
    if dof_sig is not None:
        print('per-DoF posterior sigma (deg) [le r,p,y | ec r,p,y]: '
              + np.array2string(dof_sig, precision=3))

    # ---- persist ---------------------------------------------------------------
    out_dir = ROOT / 'outputs' / 'm1'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f'pba_{args.mode}_result.npz',
             R_le=R_le_f, R_ec=R_ec_f, x=x)
    drift = dict(lidar=nz.geodesic_deg(R_le_f, R_le0),
                 cam=nz.geodesic_deg(R_ec_f, R_ec0))
    summary = dict(
        mode=args.mode, frames=len(frames), sweeps=SWEEPS,
        ref_idx=ref_idx,
        planes=[dict(n=n.tolist(), d=float(d), inliers=int(c))
                for n, d, c in planes],
        n_cloud=int(len(cloud)),
        views=[int(t['n_views']) for t in tracks],
        rounds=[info1, info2],
        init=dict(lidar=nz.geodesic_deg(R_le0, GT['R_le']),
                  cam=nz.geodesic_deg(R_ec0, GT['R_ec'])),
        final=dict(lidar=nz.geodesic_deg(R_le_f, GT['R_le']),
                   cam=nz.geodesic_deg(R_ec_f, GT['R_ec'])),
        drift_from_init=drift,
        within_0p10_deg=bool(max(drift.values()) < 0.10),
        dof_sigma_deg=(dof_sig.tolist() if dof_sig is not None else None),
        prior_sigma_deg=pba.PRIOR_SIGMA_DEG, box_deg=pba.BOX_DEG,
        oracle=pba.ORACLE_NOTE, **counts,
        sec=round(time.time() - t_start, 1))
    write_summary(out_dir / 'pba_summary.json', args.mode, summary)

    # ---- visualization: constrained tracks on the reference image --------------
    import cv2
    vis = fg_ref.img.copy()
    r_all = np.abs(pb.raw(x)[0])
    uv_all = np.concatenate([t['uv_ref'] for t in tracks], axis=0)
    for j in np.nonzero(pb.pk >= 0)[0]:
        d = r_all[j]
        col = (0, 200, 0) if d < 0.05 else \
            ((0, 165, 255) if d < 0.15 else (0, 0, 255))
        cv2.circle(vis, (int(round(uv_all[j, 0])), int(round(uv_all[j, 1]))),
                   2, col, -1)
    cv2.imwrite(str(out_dir / f'pba_{args.mode}_vis.png'), vis)
    print('saved vis')
    print(f'done in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
