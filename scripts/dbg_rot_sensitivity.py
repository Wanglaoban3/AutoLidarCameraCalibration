"""Controlled rotation-sensitivity experiment for the plane-association
field (2026-09-29).

Takes the GT-state lidar-pinned evidence (known consistent: planes fit
the same cloud the depths come from), rotates it by a known delta about
the camera origin, and asks:
  1. does the association metric NOTICE a 1 deg evidence rotation?
  2. does the inverse rotation restore it?
If not, the metric -- not the sign convention -- is the bottleneck and
no solver can recover 1 deg drift from this evidence class.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import run_pba            # noqa: E402
import run_pba_hf         # noqa: E402
from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402
from auto_extrinsics.fine import pba                  # noqa: E402


def main():
    ds = NuScenesLite()
    recs = ds.frames_of_log_multi(run_pba.LOG, channels=('CAM_FRONT',))
    cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
    cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')
    run_pba.GT.update(R_ec=cal_c['R_cs'], R_le=cal_l['R_cs'])

    kfs = run_pba.build_frames(ds, recs)
    centers = list(range(0, len(kfs), run_pba_hf.STRIDE))[:run_pba_hf.WINDOWS]
    center_fgs = [kfs[i] for i in centers]
    ref_idx = len(center_fgs) // 2
    rng_p = np.random.default_rng(7)
    planes, per_frame_planes, cloud = pba.extract_facades(
        center_fgs, ref_idx, rng_p)
    cloud = np.vstack([fg.stack_p[-1] for fg in center_fgs])
    T_ge_ref = center_fgs[ref_idx].T_ge_c
    tree = cKDTree(cloud)

    feats_full_cache = {}

    def feats_full_of(i):
        if i not in feats_full_cache:
            feats_full_cache[i] = pba.detect_features(kfs[i].img)
        return feats_full_cache[i]

    tracks = []
    for i in centers:
        offsets = run_pba_hf.pair_offsets(kfs, i, run_pba_hf.MIN_PAIR_M)
        feat = pba.detect_features(kfs[i].img)
        dm = pba.build_depth_map(kfs[i])[0]
        feats_ref = [None] * len(kfs)
        feats_ref[i] = feat
        tracks += pba.build_tracks_pairs(kfs, i, feats_ref, feats_full_of,
                                         dm, offsets, verbose=False)
    print(f'tracks: {len(tracks)}')

    T_all = np.asarray([t['T_ge'] for t in tracks], float)
    B = np.linalg.inv(T_ge_ref)[None] @ T_all
    U_ref = np.einsum('nij,nj->nj', B[:, :3, :3],
                      np.asarray([t.get('u_pin', t['u']) for t in tracks])
                      ) + B[:, :3, 3]
    t_ec = center_fgs[ref_idx].t_ec

    def probe(U, tag):
        sel, dist, w, _, _ = pba.local_plane_membership(U, cloud, tree, 0.80)
        n = int(sel.sum())
        med = float(np.median(dist[sel])) if n else -1.0
        d_cl = np.minimum(dist, 1.5)
        print(f'{tag}: n_cons {n:4d} median {med:.3f} '
              f'mean_clipped {float(d_cl.mean()):.3f}')
        return n, med, float(d_cl.mean())

    probe(U_ref, 'base (GT evidence)')
    # fixed-association mean profile: freeze the base association
    # (planes + membership), then scan +-1 deg per axis and report the
    # mean/Huber distance of the SAME points against the SAME planes --
    # a renewal-free statistic (counts saturate, means do not)
    sel0, dist0, w0, n0_, d0_ = pba.local_plane_membership(
        U_ref, cloud, tree, 0.80)
    pk0 = np.where(sel0, np.cumsum(sel0) - 1, -1).astype(np.int32)
    planes_loc = list(zip(n0_[sel0], d0_[sel0]))
    keep = pk0 >= 0
    Uk = U_ref[keep]
    pk_k = pk0[keep]
    N = np.array([p[0] for p in planes_loc])
    D = np.array([p[1] for p in planes_loc])

    def prof(axis_vec, deg):
        dv = np.radians(deg) * axis_vec
        R = Rotation.from_rotvec(dv).as_matrix()
        U_r = (Uk - t_ec) @ R.T + t_ec
        r = np.abs(np.einsum('nj,nj->n', U_r, N[pk_k]) + D[pk_k])
        fs = 0.30
        hub = np.where(r < fs, 0.5 * r ** 2 / fs, r - 0.5 * fs)
        return float(r.mean()), float(np.median(r)), float(hub.mean())

    axes = dict(roll=np.array([1., 0., 0.]), pitch=np.array([0., 1., 0.]),
                yaw=np.array([0., 0., 1.]))
    for name, ax in axes.items():
        row = []
        for deg in (-1.0, -0.5, 0.0, 0.5, 1.0):
            m, md, h = prof(ax, deg)
            row.append(f'{deg:+.1f}: mean {m:.3f} hub {h:.4f}')
        print(f'[fixed-assoc {name:5s}] ' + ' | '.join(row))
    axes = dict(roll=np.array([1., 0., 0.]), pitch=np.array([0., 1., 0.]),
                yaw=np.array([0., 0., 1.]))
    for name, ax in axes.items():
        for deg in (0.5, 1.0):
            dv = np.radians(deg) * ax
            R = Rotation.from_rotvec(dv).as_matrix()
            U_d = (U_ref - t_ec) @ R.T + t_ec
            Rinv = Rotation.from_rotvec(-dv).as_matrix()
            U_c = (U_d - t_ec) @ Rinv.T + t_ec
            probe(U_d, f'rot +{deg:.1f} {name:5s}')
            probe(U_c, '  restored')


if __name__ == '__main__':
    main()
