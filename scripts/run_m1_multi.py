# -*- coding: utf-8 -*-
"""M1 multi-camera stage -- 6 cameras + lidar, ALL with drift.

State: the lidar rotation is PINNED at its base estimate (the rig anchor;
its residual absolute error is the coarse stage's business and cancels in
lidar->camera projections); the state x = per-camera rotation-vector
deltas, 18 DoF, order ds.CAM_CHANNELS. Per-camera cross-modal evidence
drives each delta; a 0.08/dim prior toward the init encodes "drift is a
small quantity" (this is what absorbs the common-mode component that
cross-modal evidence cannot see).

Noise model (user-corrected): ALL 6 cameras AND the lidar carry drift.
--mode fresh draws an independent mounting noise per sensor (mag deg);
--mode gt is the zero-noise stability check; --mode real_coarse is the
legacy Boston instance (only CAM_FRONT + lidar recovered npy exist, other
cameras stay factory).

Evidence here is the reference recipe's marking-DT (kept as the fixture
that validates the 18-DoF plumbing; its pooled pull was measured flat and
it is NOT expected to pass -- the live candidates are the published
method ports: lt12 / gril24 / tlc). --probe prints the per-camera
per-DoF pooled-cost sweep; every dead family died right there."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402
from auto_extrinsics import noise as nz                      # noqa: E402
from auto_extrinsics.fine import geometry as geo             # noqa: E402
from auto_extrinsics.fine import evidence as ev              # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'
N_FRAMES = 12
SWEEPS = 10
MAX_PER_CAM = 200
BARRIER = 30.0


def build_keyframes(ds, recs, teed, device):
    """One dict per sample: stacked cloud + 6 camera FrameGeom (own
    calib/ego/image/TEED; shared stack arrays) + per-camera frustum-clipped
    marking pools at the FACTORY base pose (assignment once)."""
    kfs = []
    for k, rec in enumerate(recs):
        lid_sd = rec['LIDAR_TOP']
        cal_l = ds.calib(lid_sd, 'LIDAR_TOP')
        sweeps = []
        for sd in ds.sweep_history(lid_sd, SWEEPS):
            arr = ds.load_sweep(sd, with_intensity=True)
            p = arr[arr[:, 0] > -10.0]
            sweeps.append(dict(p_l=p[:, :3], intensity=p[:, 3],
                               T_eg=ds.ego_pose(sd)))
        cams = {}
        for ch in ds.CAM_CHANNELS:
            cam_sd = rec[ch]
            cal = ds.calib(cam_sd, ch)
            img = ds.load_image(cam_sd)
            fg = geo.FrameGeom(
                name=f'f{k:02d}_{ch}', K=cal['K'], R_ec=cal['R_cs'],
                t_ec=cal['t_cs'], ego_c=ds.ego_pose(cam_sd),
                R_le=cal_l['R_cs'], t_le=cal_l['t_cs'],
                ego_l=ds.ego_pose(lid_sd), img_shape=img.shape[:2])
            fg.img = img
            fg.dyn_boxes = ds.dynamic_boxes_global(rec['sample'])
            fg.dyn_mask = geo.build_dynamic_mask(fg, fg.dyn_boxes)
            fg.teed_prob = ev.teed_prob(teed, img, device)
            fg.dt_map = ev.build_dt_map(fg)
            cams[ch] = fg
        front = cams['CAM_FRONT']
        front.set_stacked(sweeps)
        for ch, fg in cams.items():
            if ch != 'CAM_FRONT':
                fg.stack_p = front.stack_p
                fg.stack_sid = front.stack_sid
                fg.stack_G = front.stack_G
                fg.stack_intensity = front.stack_intensity
        marks = ev.marking_points(front, front.stack_intensity,
                                  front_only=False)
        for ch, fg in cams.items():
            uv, z, ok = fg.project_stacked(fg.R_ec0, fg.R_le0, idx=marks)
            idx = marks[ok]
            if len(idx) > MAX_PER_CAM:
                idx = idx[::int(np.ceil(len(idx) / MAX_PER_CAM))].copy()
            fg.mark_ids = idx
        n = sum(len(cams[c].mark_ids) for c in ds.CAM_CHANNELS)
        print(f'keyframe {k}: marking obs {n} '
              + ' '.join(f'{c[4:7]}:{len(cams[c].mark_ids)}'
                         for c in ds.CAM_CHANNELS))
        kfs.append(dict(cams=cams))
    return kfs


def frame_res(fg, R_le_v, R_ec_v):
    """Per-camera marking DT residuals; barrier for out-of-frame."""
    uv, z, ok = fg.project_stacked(R_ec_v, R_le_v, idx=fg.mark_ids)
    r = np.full(len(uv), BARRIER)
    if ok.any():
        d = ev.bilinear(fg.dt_map, uv[ok, 0], uv[ok, 1])
        r[ok] = np.where(np.isfinite(d), d, BARRIER)
    return r


def apply_state(kfs, R_le_v, x_rad):
    """Per-camera trial rotations: Exp(x_ch) left-multiplied on each
    camera's own base; the lidar rotation is shared and pinned."""
    out = {}
    for i, ch in enumerate(NuScenesLite.CAM_CHANNELS):
        R = Rotation.from_rotvec(x_rad[3 * i:3 * i + 3]).as_matrix()
        for kf in kfs:
            fg = kf['cams'][ch]
            out[id(fg)] = R @ fg.R_ec0
    return out


def pooled_cost(kfs, R_le_v, x_rad, split=None):
    """(median, p90, valid ratio) over the FULL residual vector including
    the barrier. split: None, 'train' (even keyframes), 'hold' (odd)."""
    vals, n_valid, n_all = [], 0, 0
    r_ec = apply_state(kfs, R_le_v, x_rad)
    for ki, kf in enumerate(kfs):
        if split == 'train' and ki % 2:
            continue
        if split == 'hold' and ki % 2 == 0:
            continue
        for ch, fg in kf['cams'].items():
            r = frame_res(fg, R_le_v, r_ec[id(fg)])
            vals.append(r)
            n_valid += int((r < BARRIER).sum())
            n_all += len(r)
    v = np.concatenate(vals)
    return (float(np.median(v)), float(np.percentile(v, 90)),
            n_valid / max(n_all, 1))


def report(tag, R_le, r_ec_by_ch, gts):
    print(f'[{tag}] lidar pinned: geodesic '
          f'{nz.geodesic_deg(R_le, gts["le"]):.3f} deg (not estimated)')
    for ch in r_ec_by_ch:
        abs_g = nz.geodesic_deg(r_ec_by_ch[ch], gts['ec'][ch])
        rel_g = nz.geodesic_deg(
            r_ec_by_ch[ch] @ R_le.T, gts['ec'][ch] @ gts['le'].T)
        print(f'[{tag}] {ch:16s} abs {abs_g:.3f}  rel-to-lidar {rel_g:.3f}'
              f' deg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=LOG)
    ap.add_argument('--scene', default=None,
                    help='scene name (v1.0-mini: 10 scenes over 8 logs; '
                         'the harness enumerates scenes)')
    ap.add_argument('--mode', default='fresh',
                    choices=['fresh', 'gt', 'real_coarse'])
    ap.add_argument('--probe', action='store_true',
                    help='per-camera per-DoF pooled-cost sweep, no opt')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--mag', type=float, default=1.0)
    ap.add_argument('--frames', type=int, default=N_FRAMES)
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = NuScenesLite()
    recs = ds.frames_of_log_multi(args.log, scene_name=args.scene)
    step = max(1, len(recs) // args.frames)
    recs = recs[::step][:args.frames]
    print(f'{len(recs)} keyframes from {args.log} '
          f'scene={args.scene or "(all)"}')

    teed = ev.load_teed(device)
    kfs = build_keyframes(ds, recs, teed, device)

    gts = dict(le=ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')['R_cs'],
               ec={ch: ds.calib(recs[0][ch], ch)['R_cs']
                   for ch in ds.CAM_CHANNELS})

    # drift on ALL 7 sensors (user-corrected model); real_coarse is the
    # legacy Boston instance with only front+lidar recovered npy
    for kf in kfs:
        for ch, fg in kf['cams'].items():
            fg.R_le0 = gts['le'].copy()
            fg.R_ec0 = gts['ec'][ch].copy()
    R_le0 = gts['le'].copy()
    if args.mode == 'real_coarse':
        coarse_dir = ROOT / 'extrinsic_recovery' / 'results'
        R_le0 = np.load(coarse_dir / 'lidar_R_le_recovered.npy')
        for kf in kfs:
            kf['cams']['CAM_FRONT'].R_ec0 = np.load(
                coarse_dir / 'camera_R_ec_recovered.npy')
        print('init: legacy real_coarse (front+lidar npy; others factory)')
    elif args.mode == 'fresh':
        rng = np.random.default_rng(args.seed)
        R_le0 = nz.sample_mounting_noise(rng, args.mag, with_yaw=True) \
            @ R_le0
        for kf in kfs:
            for ch, fg in kf['cams'].items():
                fg.R_ec0 = nz.sample_mounting_noise(
                    rng, args.mag, with_yaw=True) @ fg.R_ec0
        print(f'init: fresh drift on 6 cams + lidar (mag {args.mag}, '
              f'seed {args.seed})')
    else:
        print('init: GT (zero-noise stability check)')

    x0 = np.zeros(18)
    m0, p0, vf0 = pooled_cost(kfs, R_le0, x0)
    r_ec0 = {ch: apply_state(kfs, R_le0, x0)[id(kfs[0]['cams'][ch])]
             for ch in ds.CAM_CHANNELS}
    report('init', R_le0, r_ec0, gts)
    print(f'pooled marking DT at init: median {m0:.2f} p90 {p0:.2f} '
          f'valid {vf0:.2f}')

    if args.probe:
        print('per-camera per-DoF pooled-cost sweep (all keyframes):')
        for i, ch in enumerate(ds.CAM_CHANNELS):
            row = []
            for dim in range(3):
                vals = []
                for x in (-0.3, -0.1, 0.1, 0.3):
                    x18 = np.zeros(18)
                    x18[3 * i + dim] = x
                    med, _, _ = pooled_cost(kfs, R_le0, x18)
                    vals.append(f'{x:+.1f}:{med:5.2f}')
                row.append(' '.join(vals))
            print(f'  {ch:16s} r[{row[0]}] p[{row[1]}] y[{row[2]}]')
        return

    # ---- bounded least_squares over the 18 camera deltas -------------------
    from scipy.optimize import least_squares
    train = [kf for i, kf in enumerate(kfs) if i % 2 == 0]
    step18 = np.full(18, 1.0)
    lo, up = -step18, step18

    def cost(x_deg):
        x_rad = np.radians(x_deg)
        r_ec = apply_state(train, R_le0, x_rad)
        rs = []
        for kf in train:
            for ch, fg in kf['cams'].items():
                rs.append(frame_res(fg, R_le0, r_ec[id(fg)]))
        rs.append(0.08 * x_deg / step18)
        return np.concatenate(rs)

    res = least_squares(cost, x0, bounds=(lo, up),
                        loss='huber', f_scale=2.5, max_nfev=60)
    x = res.x
    m1, p1, vf1 = pooled_cost([kf for i, kf in enumerate(kfs) if i % 2],
                              R_le0, np.radians(x), split='hold')
    m0h, p0h, vf0h = pooled_cost(
        [kf for i, kf in enumerate(kfs) if i % 2], R_le0, x0, split='hold')
    boundary_hit = bool(np.any(np.isclose(x, lo, atol=1e-5))
                        or np.any(np.isclose(x, up, atol=1e-5)))
    print(f'gate holdout: DT med {m0h:.2f}->{m1:.2f} p90 {p0h:.2f}->{p1:.2f} '
          f'valid {vf0h:.2f}->{vf1:.2f}'
          + (' -- BOUNDARY HIT' if boundary_hit else ''))
    keep = bool(res.success and m1 < m0h and p1 < p0h
                and vf1 >= 0.8 * vf0h and not boundary_hit)
    print('  -- PUBLISH' if keep else '  -- REJECT')
    if not keep:
        x = x0
    r_ec_f = apply_state(kfs, R_le0, np.radians(x))
    r_ec_f = {ch: r_ec_f[id(kfs[0]['cams'][ch])] for ch in ds.CAM_CHANNELS}
    report('final', R_le0, r_ec_f, gts)

    out_dir = ROOT / 'outputs' / 'm1'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.log[:4]}_{args.scene or "all"}_{args.mode}'
    np.savez(out_dir / f'{tag}_multi_result.npz',
             R_le=R_le0, x=x, r_ec=np.stack([r_ec_f[c]
                                             for c in ds.CAM_CHANNELS]))
    abs_f = {c: nz.geodesic_deg(r_ec_f[c], gts['ec'][c])
             for c in ds.CAM_CHANNELS}
    rel_f = {c: nz.geodesic_deg(r_ec_f[c] @ R_le0.T,
                                gts['ec'][c] @ gts['le'].T)
             for c in ds.CAM_CHANNELS}
    abs_i = {c: nz.geodesic_deg(r_ec0[c], gts['ec'][c])
             for c in ds.CAM_CHANNELS}
    rel_i = {c: nz.geodesic_deg(r_ec0[c] @ R_le0.T,
                                gts['ec'][c] @ gts['le'].T)
             for c in ds.CAM_CHANNELS}
    summary = dict(
        log=args.log, scene=args.scene, mode=args.mode, seed=args.seed,
        mag=args.mag, frames=len(kfs), stage='marking-dt-18dof',
        init=dict(lidar=nz.geodesic_deg(R_le0, gts['le']),
                  cam_abs=abs_i, cam_rel=rel_i,
                  mean_abs=float(np.mean(list(abs_i.values()))),
                  mean_rel=float(np.mean(list(rel_i.values())))),
        final=dict(lidar=nz.geodesic_deg(R_le0, gts['le']),
                   cam_abs=abs_f, cam_rel=rel_f,
                   mean_abs=float(np.mean(list(abs_f.values()))),
                   mean_rel=float(np.mean(list(rel_f.values())))),
        gate=dict(median=(m0h, m1), p90=(p0h, p1), valid=(vf0h, vf1),
                  published=keep, x_deg=x.tolist()))
    with open(out_dir / f'{tag}_multi_summary.json', 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('saved:', out_dir / f'{tag}_multi_summary.json')


if __name__ == '__main__':
    main()
