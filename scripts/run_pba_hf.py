# -*- coding: utf-8 -*-
"""PBA-HF: the PBACalib port on measured-correct evidence geometry.

Keeps run_pba.py's facade/membership/solve/gate machinery VERBATIM;
only the track stage changes, driven by what the data measurements
forced (dbg_track_window*, 2026-09-29):

  * v1.0-mini ego metadata is internally EXACT: keyframe (2 Hz), lidar
    sweep (20 Hz) and intermediate camera (10-12 Hz) poses agree to
    1 mm. ego_pose is usable as trajectory GT everywhere.
  * KLT/NCC chains through the 12 Hz camera frames drift ~2 px/step on
    repetitive facades (LK and global-template matching agree; FB
    checks cannot see consensus slide) -> chain-anchored triangulation
    is unusable.
  * Direct keyframe-pair matching is geometrically sound at 0.4-1 s
    baselines (2-3 px oracle-epipolar) and degrades at 2 s (17 px).
  * The original 2 Hz matcher failed in the descriptor RACE: on
    repetitive facades epipolar-perfect radial aliases beat the true
    correspondent. The fix is adjudication, not tighter matching:
    triangulate ALL epipolar-consistent candidates (up to GUIDED_K per
    keypoint, match_guided) and keep only the depth-unambiguous one --
    |z_tri - z_lidar| at the reference pixel (pba.build_tracks_pairs).

Evidence per center keyframe (excitation-gated by ego GT):
  offsets +-1, +-2 keyframes whose ego displacement >= MIN_PAIR_M;
  candidates from every qualifying offset, depth-adjudicated, N-view
  DLT when a keypoint wins at 2+ offsets. LiDAR side unchanged: 10
  sweeps (0.5 s) stacked per frame for facade planes + depth maps.
ORACLE: triangulation walks nuScenes ego_pose (run_pba.ORACLE_NOTE).

Modes (same discipline as run_pba.py):
  probe   evidence census only.  gt     init = GT; HARD GATE: must end
          within 0.10 deg/sensor.  drift  inject U(-mag, mag) mounting
          noise (--which both = lidar+cam; --which cams = lidar pinned
          factory-correct, camera mounts only).

Usage:
  python scripts/run_pba_hf.py --mode probe
  python scripts/run_pba_hf.py --mode gt
  python scripts/run_pba_hf.py --mode drift --which cams --mag 1.0 --seed 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_pba                                        # noqa: E402
from run_pba import (solve_round, membership,         # noqa: E402
                     write_summary, SWEEPS)

from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402
from auto_extrinsics import noise as nz               # noqa: E402
from auto_extrinsics.fine import geometry as geo      # noqa: E402
from auto_extrinsics.fine import pba                  # noqa: E402
from auto_extrinsics.fine.solver import apply_deltas  # noqa: E402

STRIDE = 2        # keyframes between centers (1 s)
WINDOWS = 10
MIN_PAIR_M = 1.2  # per-offset excitation floor (ego GT displacement)
GT = {}           # factory calib, filled in main()


def pair_offsets(kfs, i, min_pair_m):
    """Qualifying offsets: ego displacement (GT) over the pair >= floor."""
    out = []
    p_i = kfs[i].T_ge_c[:3, 3]
    for off in (-2, -1, 1, 2):
        j = i + off
        if not (0 <= j < len(kfs)):
            continue
        d = float(np.linalg.norm(kfs[j].T_ge_c[:3, 3] - p_i))
        if d >= min_pair_m:
            out.append(off)
    return out


def local_membership(tracks, cloud, tree, band, T_ge_ref, debug=False):
    """Run the per-track LOCAL lidar-plane association on the tracks
    bridged into the scene-reference ego (pba.local_plane_membership).
    Returns (U_own, pk, w, planes_loc, n_cons): U_own is the PER-TRACK
    ego evidence PBAResidual expects (it re-bridges via each track's
    T_ge -- passing the ego-bridged points here double-transforms and
    shows up as ~ego-offset residuals); planes_loc[k] constrains the
    track with pk == k (identity indexing for PBAResidual)."""
    U_own = np.asarray([t.get('u_pin', t['u']) for t in tracks], float)
    T = np.asarray([t['T_ge'] for t in tracks], float)
    B = np.linalg.inv(T_ge_ref)[None] @ T
    U_ref = np.einsum('nij,ni->nj', B[:, :3, :3], U_own) + B[:, :3, 3]
    sel, dist, w, normals, ds = pba.local_plane_membership(
        U_ref, cloud, tree, band, debug=debug)
    idx = np.cumsum(sel) - 1
    pk = np.where(sel, idx, -1).astype(np.int32)
    planes_loc = list(zip(normals[sel], ds[sel]))
    n_cons = int(sel.sum())
    print(f'local membership @ band {band:.2f} m: {n_cons} '
          f'plane-constrained tracks '
          f'({100.0 * n_cons / max(len(U_own), 1):.0f}% of {len(U_own)}); '
          f'median dist '
          f'{np.median(dist[sel]) if n_cons else float("nan"):.3f} m')
    return U_own, pk, w, planes_loc, n_cons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=run_pba.LOG)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--mode', default='gt',
                    choices=['probe', 'gt', 'drift'])
    ap.add_argument('--which', default='both', choices=['both', 'cams'],
                    help='cams = lidar rotvec pinned factory-correct '
                         '(camera-mounts-only drift)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mag', type=float, default=1.0)
    ap.add_argument('--windows', type=int, default=WINDOWS,
                    help='max center keyframes')
    ap.add_argument('--frames', type=int, default=None,
                    help='alias of --windows (batch-harness compat)')
    ap.add_argument('--stride', type=int, default=STRIDE)
    args = ap.parse_args()
    n_windows = args.frames or args.windows
    cams_only = args.which == 'cams'

    t_start = time.time()
    ds = NuScenesLite()
    recs = ds.frames_of_log_multi(args.log, channels=('CAM_FRONT',),
                                  scene_name=args.scene)
    cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
    cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')
    GT.update(R_ec=cal_c['R_cs'], R_le=cal_l['R_cs'])
    run_pba.GT.update(R_ec=cal_c['R_cs'], R_le=cal_l['R_cs'])

    # ---- init ---------------------------------------------------------------
    if args.mode == 'drift':
        rng = np.random.default_rng(args.seed)
        Cn_le = (nz.sample_mounting_noise(rng, args.mag, with_yaw=True)
                 if not cams_only else np.eye(3))
        Cn_ec = nz.sample_mounting_noise(rng, args.mag, with_yaw=True)
        R_le0, R_ec0 = Cn_le @ GT['R_le'], Cn_ec @ GT['R_ec']
        print(f'init: drift mode, which={args.which}, mag {args.mag}, '
              f'seed {args.seed}')
    else:
        R_le0, R_ec0 = GT['R_le'].copy(), GT['R_ec'].copy()
        print('init: GT' if args.mode == 'gt' else 'init: GT (probe)')
    run_pba.report('init', R_le0, R_ec0, GT['R_le'], GT['R_ec'])

    # ---- all keyframes (stacked clouds; centers pick references) -------------
    kfs = run_pba.build_frames(ds, recs)
    for fg in kfs:
        fg.R_ec0 = R_ec0.copy()
        fg.R_le0 = R_le0.copy()
    centers = [i for i in range(len(kfs))
               if i % args.stride == 0
               and pair_offsets(kfs, i, MIN_PAIR_M)][:n_windows]
    print(f'{len(centers)}/{len(kfs)} center keyframes from {args.log} '
          f'(scene {args.scene}, stride {args.stride}, offsets with '
          f'ego displacement >= {MIN_PAIR_M} m)')
    print(pba.ORACLE_NOTE)
    if not centers:
        print('no excited centers -- abort')
        return

    # ---- lidar facades (identical to run_pba) -------------------------------
    t0 = time.time()
    rng_p = np.random.default_rng(7)
    center_fgs = [kfs[i] for i in centers]
    ref_idx = len(center_fgs) // 2
    planes, per_frame_planes, cloud = pba.extract_facades(
        center_fgs, ref_idx, rng_p)
    print(f'{len(planes)} merged constraint planes '
          f'({len(per_frame_planes)} per-frame) on {len(cloud)} pts '
          f'({time.time() - t0:.1f}s)')
    for k, (n, d, c) in enumerate(planes):
        print(f'  plane {k}: n={np.array2string(n, precision=3)} '
              f'd={d:+.1f} m inliers={c}')
    if len(planes) == 0:
        print('no facades -- abort')
        return

    # ---- pair tracks with depth adjudication (the stage that changed) --------
    T_ge_ref = center_fgs[ref_idx].T_ge_c
    feats_full_cache = {}

    def feats_full_of(i):
        if i not in feats_full_cache:
            feats_full_cache[i] = pba.detect_features(kfs[i].img)
        return feats_full_cache[i]

    tracks = []
    n_views_all = []
    for c, i in enumerate(centers):
        offsets = pair_offsets(kfs, i, MIN_PAIR_M)
        B = np.linalg.inv(kfs[i].T_ge_c) @ T_ge_ref
        planes_k = [(n, d, inl @ B[:3, :3].T + B[:3, 3])
                    for n, d, inl in per_frame_planes]
        m_k = pba.facade_mask(kfs[i], planes_k)
        feat = pba.detect_features(kfs[i].img, mask=m_k)
        if len(feat[0][0]) < 300:
            # the global-plane mask misses oblique corner facades
            # (dbg13); fall back to unmasked detection -- the depth
            # adjudication, not the mask, filters the aliases
            feat = pba.detect_features(kfs[i].img)
        dm = pba.build_depth_map(kfs[i])[0]
        feats_ref = [None] * len(kfs)
        feats_ref[i] = feat
        t0 = time.time()
        tr = pba.build_tracks_pairs(kfs, i, feats_ref, feats_full_of,
                                    dm, offsets, verbose=True)
        tracks.extend(tr)
        n_views_all.extend(t['n_views'] for t in tr)
        print(f'center kf{i} (offsets {offsets}): {len(tr)} tracks '
              f'({time.time() - t0:.1f}s)')
    n_tracks = len(tracks)
    print(f'tracks: {n_tracks} 3D pts, median views '
          f'{int(np.median(n_views_all)) if n_views_all else 0}')
    if n_tracks == 0:
        print('no triangulated tracks -- abort')
        return

    # ---- membership + solve ---------------------------------------------------
    # Per-track LOCAL lidar planes (pba.local_plane_membership): the
    # merged global plane list misses the oblique intersection-corner
    # facades the tracks actually live on (dbg13: nearest global plane
    # 4-6.6 m for 100% of tracks -> funnel starves on plane coverage).
    # Ground INCLUDED (2026-09-29): vertical facades alone leave camera
    # pitch unobservable (pitch moves points vertically, horizontal
    # normals see none of it) -- the ground plane is the pitch
    # constraint; noisy ground patches simply fail the patch-PCA
    # planarity test instead of being excluded a priori.
    gmask = np.ones(len(cloud), dtype=bool)
    tree_f = cKDTree(cloud[gmask])
    BAND1, BAND2 = 0.80, 0.30   # m; structure noise here is ~0.3-0.5 m
    U, pk, w, planes_loc, n_cons = local_membership(
        tracks, cloud[gmask], tree_f, BAND1, T_ge_ref, debug=True)
    counts = dict(n_tracks=int(n_tracks),
                  median_views=int(np.median(n_views_all)),
                  n_planarity=n_cons, n_constrained=n_cons,
                  per_plane=[])

    def base_summary(starved=False):
        return dict(mode=args.mode, which=args.which,
                    seed=args.seed, mag=args.mag,
                    centers=len(centers), sweeps=SWEEPS,
                    ref_idx=ref_idx,
                    planes=[dict(n=n.tolist(), d=float(d), inliers=int(c))
                            for n, d, c in planes],
                    n_cloud=int(len(cloud)), starved=starved,
                    band1=BAND1, band2=BAND2,
                    **counts)

    if n_cons < 200:
        print(f'STARVED: {n_cons} constrained tracks < 200 -- no solve')
        write_summary(ROOT / 'outputs' / 'm1' / 'pba_hf_summary.json',
                      f'{args.mode}_{args.which}', base_summary(True))
        return

    pb6 = pba.PBAResidual(
        U, pk, w, planes_loc, kfs[centers[ref_idx]].T_ge_c,
        kfs[centers[ref_idx]].t_le, kfs[centers[ref_idx]].t_ec,
        [t['T_ge'] for t in tracks], R_le0, R_ec0)
    if cams_only:
        pb = pba.FixedLidarPBA(pb6)
        to_full = lambda x: np.r_[np.zeros(3), x]          # noqa: E731
        x = np.zeros(3)
    else:
        pb = pb6
        to_full = None
        x = np.zeros(6)

    if args.mode == 'probe':
        write_summary(ROOT / 'outputs' / 'm1' / 'pba_hf_summary.json',
                      'probe', base_summary(False))
        print(f'probe done in {time.time() - t_start:.0f}s')
        return

    if args.mode == 'drift':
        # diagnostic: what would the holdout gate say at the TRUE
        # correction (x = -rotvec(Cn))? Separates 'solver failed to
        # find the basin' from 'gate unreachable at this noise floor'.
        Cn_le_r = (Rotation.from_matrix(Cn_le).as_rotvec()
                   if not cams_only else np.zeros(3))
        Cn_ec_r = Rotation.from_matrix(Cn_ec).as_rotvec()
        xt_full = np.r_[-Cn_le_r, -Cn_ec_r]
        xt = xt_full[3:] if cams_only else xt_full
        idx0 = np.nonzero(pb.pk >= 0)[0]
        rng_d = np.random.default_rng(123)
        hs = idx0[rng_d.random(len(idx0)) < run_pba.HOLDOUT_FRAC]
        m0, p00, _ = pb.gate(x * 0, hs)
        m1, p01, _ = pb.gate(xt, hs)
        print(f'[diag] gate @init med {m0:.3f} p90 {p00:.3f} | '
              f'@true-corr med {m1:.3f} p90 {p01:.3f} (n={len(hs)})')
        # selection-free signal probe: re-associate ALL tracks after
        # rotating the evidence by 0 / -d / +d and count how many land
        # on lidar planes (the x=0-selected holdout is biased against
        # any move by construction)
        T_all = np.asarray([t['T_ge'] for t in tracks], float)
        B = np.linalg.inv(T_ge_ref)[None] @ T_all
        U_ref_all = np.einsum('nij,nj->nj', B[:, :3, :3],
                              np.asarray([t.get('u_pin', t['u'])
                                          for t in tracks])
                              ) + B[:, :3, 3]
        t_ec_r = kfs[centers[ref_idx]].t_ec
        for tag_x, xv in [('0', np.zeros(3)),
                          ('-d', -Cn_ec_r), ('+d', Cn_ec_r)]:
            Ec_x = Rotation.from_rotvec(xv).as_matrix()
            U_rot = (U_ref_all - t_ec_r) @ Ec_x.T + t_ec_r
            sel_x, dist_x, w_x, _, _ = pba.local_plane_membership(
                U_rot, cloud[gmask], tree_f, BAND1)
            print(f'[diag] reassoc @x_ec={tag_x}: n_cons {int(sel_x.sum())}'
                  f' median {np.median(dist_x[sel_x]) if sel_x.any() else -1:.3f} m')

    rng_gate = np.random.default_rng(123)
    # round 1 at the wide band is already done (the association above);
    x, info1 = solve_round(pb, None, BAND1, x, rng_gate,
                           'round1', to_full=to_full)
    R_le1, R_ec1 = apply_deltas(R_le0, R_ec0,
                                to_full(x) if to_full else x)

    U2, pk2, w2, planes_loc2, n_cons2 = local_membership(
        tracks, cloud[gmask], tree_f, BAND2, T_ge_ref)
    if n_cons2 >= 60:
        pb6.set_association(pk2, w2, planes_loc2)   # incl. ego->global
        x, info2 = solve_round(pb, None, BAND2, x, rng_gate,
                               'round2', to_full=to_full)
    else:
        print(f'round2: band {BAND2} m keeps only '
              f'{n_cons2} tracks (< 60) -- kept round-1 association')
        info2 = dict(band_m=BAND2, skipped_starved=True,
                     n_constrained=int(n_cons2))

    x_f = to_full(x) if to_full else x
    R_le_f, R_ec_f = apply_deltas(R_le0, R_ec0, x_f)
    run_pba.report('final', R_le_f, R_ec_f, GT['R_le'], GT['R_ec'])

    dof_sig = pb.dof_sigma_deg(x, np.nonzero(pb.pk >= 0)[0])
    if dof_sig is not None:
        print('per-DoF posterior sigma (deg) '
              + ('[ec r,p,y] ' if cams_only else '[le r,p,y | ec r,p,y] ')
              + np.array2string(dof_sig, precision=3))

    # ---- persist ---------------------------------------------------------------
    out_dir = ROOT / 'outputs' / 'm1'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f'pba_hf_{args.mode}_{args.which}_result.npz',
             R_le=R_le_f, R_ec=R_ec_f, x=x_f)
    drift = dict(lidar=nz.geodesic_deg(R_le_f, R_le0),
                 cam=nz.geodesic_deg(R_ec_f, R_ec0))
    final_err = dict(lidar=nz.geodesic_deg(R_le_f, GT['R_le']),
                     cam=nz.geodesic_deg(R_ec_f, GT['R_ec']))
    resid = np.degrees(nz.residual_rotvec_deg(R_ec_f, GT['R_ec']))
    published = bool(info1['publish']
                     and (info2.get('publish', False)
                          or info2.get('skipped_starved', False)))
    stable = bool(max(drift.values()) < 0.10)
    summary = base_summary(False)
    summary.update(
        rounds=[info1, info2],
        gate=dict(published=published, zero_noise_stable=stable),
        init=dict(lidar=nz.geodesic_deg(R_le0, GT['R_le']),
                  cam=nz.geodesic_deg(R_ec0, GT['R_ec'])),
        final=final_err, drift_from_init=drift,
        final_cam_resid_deg=resid.tolist(),
        within_0p10_deg=stable,
        dof_sigma_deg=(dof_sig.tolist() if dof_sig is not None else None),
        prior_sigma_deg=pba.PRIOR_SIGMA_DEG, box_deg=pba.BOX_DEG,
        oracle=pba.ORACLE_NOTE,
        sec=round(time.time() - t_start, 1))
    stamp = (f'{args.log[:4]}_{args.scene}_{args.mode}_{args.which}'
             if args.scene else f'{args.mode}_{args.which}')
    write_summary(out_dir / 'pba_hf_summary.json',
                  f'{args.mode}_{args.which}', summary)
    (out_dir / f'pba_hf_{stamp}_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print('saved per-run summary:',
          out_dir / f'pba_hf_{stamp}_summary.json')

    # ---- visualization ----------------------------------------------------------
    vis = kfs[centers[ref_idx]].img.copy()
    uv_all = np.asarray([t['uv_ref'] for t in tracks])
    r_all = np.abs(pb6.raw(np.zeros(6))[0])
    for j in np.nonzero(pb.pk >= 0)[0]:
        d = r_all[j]
        col = (0, 200, 0) if d < 0.05 else \
            ((0, 165, 255) if d < 0.15 else (0, 0, 255))
        cv2.circle(vis, (int(round(uv_all[j, 0])),
                         int(round(uv_all[j, 1]))), 2, col, -1)
    cv2.imwrite(str(out_dir / f'pba_hf_{stamp}_vis.png'), vis)
    print('saved vis')
    print(f'done in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
