# -*- coding: utf-8 -*-
"""M1 experiment: cross-modal fine refinement of the mounting rotations.

Evidence architecture (revised twice; see docs/specs for the full story):

  PRIMARY  depth-edge samples from the DENSE stacked cloud (~0.5 s of
           sweeps bridged per-sweep through their own ego poses to the
           camera-time reference frame), one-sided normal-search to TEED-
           gated gradient ridges, ICP-style frozen association per pass,
           window annealed wide -> narrow, joint (R_le, R_ec) LM with MAP
           prior and shrinking trust regions.
           Zero-noise GT check: stable (drift < 0.02 deg, ridge offsets
           ~1 px median at a 4 px window).
  DIAGNOSTIC road-marking rim points (AutoLidarCameraCalibration recipe)
           against the TEED distance field -- reported, not optimized: on
           v1.0-mini the marking-intensity selection degenerates to
           near-field curb/lane-edge structure (paint >10 m is NOT
           separable in HDL-32E intensity) and as an objective its DT
           landscape has no basin (real_coarse diverged 0.4 -> 2.2 deg
           with the median DT flat). The reference repo uses it only as a
           light polish under tight clamps for the same reason.

Modes
  real_coarse     init from the ACTUAL coarse-pipeline outputs
                  (extrinsic_recovery/results/*_recovered.npy); reproduces
                  their injected noise (cam seed 7 / lidar seed 11) so the
                  start residuals match the published 0.507 / 0.364 deg.
  sim_coarse_yaw  fresh 3-DoF injection (incl. yaw); init = gravity-coarse
                  model (tilt residual +-0.5 deg, yaw untouched) -- validates
                  yaw observability, which the coarse stage cannot fix.
  fresh           init = noisy extrinsics themselves (basin demo).
  gt              init = GT (zero-noise stability check: the pose must stay).

Usage (from repo root):
  python scripts/run_m1.py --mode real_coarse
"""
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
from auto_extrinsics.fine.solver import LMSolver, apply_deltas  # noqa: E402
from auto_extrinsics.fine import lines as fl                    # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'   # Boston row houses (cam+lidar log)
N_FRAMES = 12
SWEEPS = 10                              # 0.5 s of 20 Hz sweeps per frame
MAX_SAMPLES = 2500                       # depth-edge samples per frame


def build_frames(ds, frame_recs, teed_model, device):
    frames = []
    for k, rec in enumerate(frame_recs):
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
        fg.teed_prob = ev.teed_prob(teed_model, img, device)
        fg.gm = ev.gradient_magnitude(img)
        fg.dyn_boxes = ds.dynamic_boxes_global(rec['sample'])
        fg.dyn_mask = geo.build_dynamic_mask(fg, fg.dyn_boxes)
        fg.dt_map = ev.build_dt_map(fg)

        sweeps = []
        for sd in ds.sweep_history(lid_sd, SWEEPS):
            arr = ds.load_sweep(sd, with_intensity=True)
            p = arr[arr[:, 0] > -10.0]        # drop the rear hemisphere
            sweeps.append(dict(p_l=p[:, :3], intensity=p[:, 3],
                               T_eg=ds.ego_pose(sd)))
        fg.p_l = sweeps[-1]['p_l']            # key sweep (fallback/debug)
        fg.set_stacked(sweeps)
        frames.append(fg)
        print(f'frame {k}: stacked {len(fg.stack_p)} pts '
              f'({len(sweeps)} sweeps)')
    return frames


def prepare_markings(frames, max_points=400):
    """Marking rim selection needs the BASE lidar rotation, which main()
    fixes only after build_frames -- so this runs as a second step."""
    for fg in frames:
        fg.mark_ids = ev.marking_points(fg, fg.mark_intensity(),
                                        max_points=max_points)


def extract_samples(frames, R_ec, R_le, rng, max_samples=MAX_SAMPLES,
                    near_side=False):
    """Depth-edge pool on the stacked cloud, extracted at the given (base)
    pose -- the pool is FIXED evidence; only ridge targets re-lock."""
    all_s = []
    for fg in frames:
        s = ev.depth_edge_samples_dense(fg, R_ec, R_le,
                                        max_samples=max_samples, rng=rng,
                                        near_side=near_side)
        s['frame'] = fg
        all_s.append(s)
    return all_s


def make_line_fn(frames, assoc):
    """PRIMARY objective: frozen-ridge residuals of projected 3D contact
    lines (fit lines only). Association frozen within a pass, redone
    between passes; residual = signed px offset along the frozen normal
    to the frozen target."""

    def fn(x):
        R_le, R_ec = apply_deltas(frames[0].R_le0, frames[0].R_ec0, x)
        rs, ws = [], []
        for per in assoc:
            for e in per:
                if e is None or e['gate']:
                    continue
                r, w = fl.frozen_residuals(e, R_ec, R_le)
                rs.append(np.where(np.isfinite(r), r, 0.0))
                ws.append(np.where(np.isfinite(w), w, 0.0))
        return np.concatenate(rs), np.concatenate(ws)
    return fn


def line_gate(frames, lines_per_frame, R_ec, R_le, half_win):
    """Holdout publish gate (AutoLidarCameraCalibration discipline), on
    contact lines re-located fresh at the fixed pose:
      med   median |ridge offset| over holdout lines;
      bias  norm of the MEAN SIGNED offset vector (the median alone is
            insensitive to coherent re-lock slide -- both observed).
    Returns (med, bias, n)."""
    acc_r, acc_v = [], []
    for fg, lines in zip(frames, lines_per_frame):
        for line in lines:
            if not line['gate']:
                continue
            uv, normals, ok = fl.line_samples_px(fg, line, R_ec, R_le)
            if normals is None or ok.sum() < max(10, fl.LINE_MIN_GOOD
                                                 * len(uv)):
                continue
            targets, w = ev.locate_ridges(fg, uv[ok], normals[ok], half_win)
            m = w > 0
            if m.sum() < max(8, fl.LINE_MIN_GOOD * int(ok.sum())):
                continue
            d = uv[ok][m] - targets[m]
            nm = normals[ok][m]
            r = d[:, 0] * nm[:, 0] + d[:, 1] * nm[:, 1]
            acc_r.append(np.abs(r))
            acc_v.append(r[:, None] * nm)
    if not acc_r:
        return None, None, 0
    v = np.concatenate(acc_r)
    if len(v) < 20:
        return None, None, int(len(v))
    vv = np.concatenate(acc_v)
    bias = float(np.hypot(vv[:, 0].mean(), vv[:, 1].mean()))
    return float(np.median(v)), bias, int(len(v))


def make_level_fn(frames, samples, assoc, half_win):
    """ICP-style level: ridge targets are FROZEN for the level (fixed
    association), residual = current projection -> target distance along
    the normal -- smooth in the pose, so LM can actually move. Samples
    flagged 'gate' (holdout) are excluded from the fit."""

    def fn(x):
        R_le, R_ec = apply_deltas(frames[0].R_le0, frames[0].R_ec0, x)
        rs, ws = [], []
        for s, a in zip(samples, assoc):
            fg = s['frame']
            uv, ok = ev.project_stacked_samples(fg, R_ec, R_le, s)
            r = ev.offset_to_targets(uv, s['normals'], a['targets'])
            bad = ~np.isfinite(r) | ~ok | s['gate']
            rs.append(np.where(bad, 0.0, r))
            ws.append(np.where(bad, 0.0, a['weights']))
        return np.concatenate(rs), np.concatenate(ws)
    return fn


def gate_median(frames, samples, R_ec, R_le, half_win):
    """Holdout publish gate (AutoLidarCameraCalibration discipline, at
    pass granularity), two components:
      med   median |offset| over holdout samples, associated fresh;
      bias  norm of the MEAN SIGNED residual vector.
    The median alone is insensitive to coherent re-lock slide (locks
    re-bias at any pose, medians stay flat while the rig drifts -- both
    observed); the signed-mean vector exposes the coherent component."""
    acc_r, acc_vx, acc_vy = [], [], []
    for s in samples:
        fg = s['frame']
        uv, ok = ev.project_stacked_samples(fg, R_ec, R_le, s)
        targets, w = ev.locate_ridges(fg, uv, s['normals'], half_win,
                                      t_max=s['t_max'])
        r = ev.offset_to_targets(uv, s['normals'], targets)
        m = (w > 0) & ok & s['gate'] & np.isfinite(r)
        if m.any():
            acc_r.append(np.abs(r[m]))
            rv = r[m][:, None] * s['normals'][m]
            acc_vx.append(rv[:, 0])
            acc_vy.append(rv[:, 1])
    if not acc_r:
        return None, None, 0
    v = np.concatenate(acc_r)
    if len(v) < 20:
        return None, None, int(len(v))
    med = float(np.median(v))
    bias = float(np.hypot(np.mean(np.concatenate(acc_vx)),
                          np.mean(np.concatenate(acc_vy))))
    return med, bias, int(len(v))


def associate(frames, samples, R_ec, R_le, half_win):
    assoc, n_ok = [], 0
    for s in samples:
        fg = s['frame']
        uv, ok = ev.project_stacked_samples(fg, R_ec, R_le, s)
        targets, weights = ev.locate_ridges(fg, uv, s['normals'], half_win,
                                            t_max=s['t_max'])
        assoc.append(dict(targets=targets, weights=weights))
        n_ok += int((weights > 0).sum())
    return assoc, n_ok


def px_rmse(frames, samples, R_ec, R_le, half_win=ev.GRAD_HALF_WIN):
    vals = []
    for s in samples:
        fg = s['frame']
        uv, ok = ev.project_stacked_samples(fg, R_ec, R_le, s)
        r, w = ev.grad_residuals(fg, uv, s['normals'], half_win=half_win)
        good = (w > 0) & ok
        vals.append(np.abs(r[good]))
    v = np.concatenate(vals) if vals else np.array([np.nan])
    return float(np.mean(v)), float(np.median(v)), int(len(v))


def marking_px(frames, R_ec, R_le):
    """(mean, median, n) DT distances of the marking rim points."""
    vals = []
    for fg in frames:
        if getattr(fg, 'mark_ids', None) is None or len(fg.mark_ids) == 0:
            continue
        uv, z, ok = fg.project_stacked(R_ec, R_le, idx=fg.mark_ids)
        d = ev.bilinear(fg.dt_map, uv[:, 0], uv[:, 1])
        vals.append(np.abs(d[ok & np.isfinite(d)]))
    v = np.concatenate(vals) if vals else np.array([np.nan])
    return float(np.mean(v)), float(np.median(v)), int(len(v))


def report(tag, R_le, R_ec, R_le_gt, R_ec_gt):
    for name, R, Rg in (('lidar', R_le, R_le_gt), ('cam', R_ec, R_ec_gt)):
        rv = nz.residual_rotvec_deg(R, Rg)
        print(f'[{tag}] {name}: geodesic {np.linalg.norm(rv):.3f} deg | '
              f'per-DoF (r,p,y) {np.array2string(rv, precision=3)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='real_coarse',
                    choices=['real_coarse', 'sim_coarse_yaw', 'fresh', 'gt'])
    ap.add_argument('--stage', default='dt', choices=['dt', 'lines'],
                    help='dt: bounded TEED-distance-field alignment '
                         '(AutoLidarCameraCalibration port, primary); '
                         'lines: 3D contact-line + ridge stage (kept for '
                         'comparison, failed GT stability)')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--mag', type=float, default=1.0)
    ap.add_argument('--frames', type=int, default=N_FRAMES)
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = NuScenesLite()
    recs = ds.frames_of_log(LOG)
    step = max(1, len(recs) // args.frames)
    recs = recs[::step][:args.frames]
    print(f'{len(recs)} frames from {LOG}')

    teed = ev.load_teed(device)
    frames = build_frames(ds, recs, teed, device)

    cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
    cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')
    R_ec_gt, R_le_gt = cal_c['R_cs'], cal_l['R_cs']

    # ---- noise + init ------------------------------------------------------
    if args.mode == 'real_coarse':
        coarse_dir = ROOT / 'extrinsic_recovery' / 'results'
        rng_c = np.random.default_rng(7)
        er, ep = (np.radians(rng_c.uniform(-2, 2)) for _ in range(2))
        Cn_c = Rotation.from_euler('xyz', [er, ep, 0]).as_matrix()
        rng_l = np.random.default_rng(11)
        er2, ep2 = (np.radians(rng_l.uniform(-2, 2)) for _ in range(2))
        Cn_l = Rotation.from_euler('xyz', [er2, ep2, 0]).as_matrix()
        R_ec0 = np.load(coarse_dir / 'camera_R_ec_recovered.npy')
        R_le0 = np.load(coarse_dir / 'lidar_R_le_recovered.npy')
        print('init: actual coarse outputs (cam seed 7 / lidar seed 11 '
              'replicated)')
    else:
        rng = np.random.default_rng(args.seed)
        Cn_c = nz.sample_mounting_noise(rng, args.mag, with_yaw=True)
        Cn_l = nz.sample_mounting_noise(rng, args.mag, with_yaw=True)
        R_ec_noisy = Cn_c @ R_ec_gt
        R_le_noisy = Cn_l @ R_le_gt
        if args.mode == 'fresh':
            R_ec0, R_le0 = R_ec_noisy, R_le_noisy
            print(f'init: noisy extrinsics (mag {args.mag} deg, seed '
                  f'{args.seed})')
        elif args.mode == 'gt':
            R_ec0, R_le0 = R_ec_gt.copy(), R_le_gt.copy()
            print('init: GT (zero-noise stability check: pose must stay)')
        else:
            psi_c = Rotation.from_matrix(Cn_c).as_euler('xyz')[2]
            psi_l = Rotation.from_matrix(Cn_l).as_euler('xyz')[2]
            R_ec0 = nz.sim_coarse_init(R_ec_gt, rng, psi_c)
            R_le0 = nz.sim_coarse_init(R_le_gt, rng, psi_l)
            print(f'init: simulated gravity-coarse (tilt resid 0.5 deg, '
                  f'yaw untouched), injection mag {args.mag} deg seed '
                  f'{args.seed}')

    report('init', R_le0, R_ec0, R_le_gt, R_ec_gt)

    # one rigid mounting error per sensor per run: every frame shares the
    # same base rotations
    for fg in frames:
        fg.R_ec0 = R_ec0.copy()
        fg.R_le0 = R_le0.copy()
    prepare_markings(frames)
    n_marks = sum(len(fg.mark_ids) for fg in frames)
    m0_mean, m0_med, _ = marking_px(frames, R_ec0, R_le0)
    print(f'marking rim points: {n_marks} (diagnostic only); DT median '
          f'{m0_med:.2f} px')

    rng_s = np.random.default_rng(0)
    rng_gate = np.random.default_rng(123)
    x = np.zeros(6)

    # ---- fixed evidence pool (extract ONCE at the base pose) ------------------
    # pose-independent evidence: depth-edge samples from the dense stacked
    # cloud, 30% holdout split. Train/holdout publish discipline from
    # AutoLidarCameraCalibration: only poses that improve BOTH gate metrics
    # on samples never fitted get accepted.
    samples = extract_samples(frames, R_ec0, R_le0, rng_s)
    n_res = sum(len(s['p_idx']) for s in samples)
    for s in samples:
        s['gate'] = rng_gate.random(len(s['p_idx'])) < 0.3
    n_gate = sum(int(s['gate'].sum()) for s in samples)
    dof_sig = None

    if args.stage == 'dt':
        # ---- PRIMARY: bounded TEED-distance-field alignment -------------------
        # Mechanics of AutoLidarCameraCalibration's teed_stacked_refinement
        # (Huber-2.5, OOF = 30 px constant barrier, prior 0.08 per dim,
        # bounded least_squares, holdout median AND p90 publish gate with
        # boundary + evidence-collapse rules). Evidence = NEAR-side
        # depth-edge samples of the stacked cloud, extracted ONCE at the
        # base pose, NO image-proximity filtering: points picked for being
        # near edges are near SOME edge at every pose, so the basin
        # flattens (measured at both GT and coarse). Marking evidence
        # starves on one front camera (5% in-frame: paint sits sideways at
        # 4-8 m while the vertical FOV eats only a 4.3-8 m ground wedge;
        # the reference needs 6 surround cameras), and the gravity box
        # center is unusable on this log (road crown biases the ground
        # normal ~1 deg, a box centered on it excludes the truth). The box
        # is tightened to +-0.8 deg: junk samples (canopy, DT ~40 px) have
        # near-zero gradient on the flat field but must not drag the trial
        # to a bound; the Huber loss bounds their influence anyway.
        samples = extract_samples(frames, R_ec0, R_le0, rng_s, near_side=True)
        n_res = sum(len(s['p_idx']) for s in samples)
        for s in samples:
            s['gate'] = rng_gate.random(len(s['p_idx'])) < 0.3

        def frame_res(s, R_le_v, R_ec_v):
            fg = s['frame']
            uv, ok = ev.project_stacked_samples(fg, R_ec_v, R_le_v, s)
            r = np.full(len(uv), 30.0)
            if ok.any():
                d = ev.bilinear(fg.dt_map, uv[ok, 0], uv[ok, 1])
                r[ok] = np.where(np.isfinite(d), d, 30.0)
            return r

        def dt_score_pose(R_le_v, R_ec_v, frs):
            # metric over the FULL residual vector INCLUDING the barrier
            # (reference score() semantics): a valid-only metric lets the
            # optimizer herd points out of frame and read the leftovers as
            # improvement; the valid-count ratio guards the degenerate
            # all-barrier solution on top
            vals = []
            n_valid = 0
            n_all = 0
            for s in frs:
                r = frame_res(s, R_le_v, R_ec_v)
                vals.append(r)
                n_valid += int((r < 30.0).sum())
                n_all += len(r)
            v = np.concatenate(vals)
            return (float(np.median(v)), float(np.percentile(v, 90)),
                    n_valid / max(n_all, 1))

        train = [s for i, s in enumerate(samples)
                 if i % 2 == 0 and len(s['p_idx']) > 0]
        hold = [s for i, s in enumerate(samples)
                if i % 2 == 1 and len(s['p_idx']) > 0]

        center = np.zeros(6)
        step = np.full(6, 0.8)
        lo, up = center - step, center + step

        def dt_cost(x_deg):
            R_le_v, R_ec_v = apply_deltas(R_le0, R_ec0, np.radians(x_deg))
            rs = [frame_res(s, R_le_v, R_ec_v) for s in train]
            rs.append(0.08 * x_deg / step)
            return np.concatenate(rs)

        from scipy.optimize import least_squares
        res = least_squares(dt_cost, center, bounds=(lo, up),
                            loss='huber', f_scale=2.5, max_nfev=100)
        x = res.x
        R_le_t, R_ec_t = apply_deltas(R_le0, R_ec0, np.radians(x))
        report('trial', R_le_t, R_ec_t, R_le_gt, R_ec_gt)
        med0, p900, vf0 = dt_score_pose(R_le0, R_ec0, hold)
        med1, p901, vf1 = dt_score_pose(R_le_t, R_ec_t, hold)
        tm0, tp900, _ = dt_score_pose(R_le0, R_ec0, train)
        tm1, tp901, _ = dt_score_pose(R_le_t, R_ec_t, train)
        boundary_hit = bool(np.any(np.isclose(x, lo, atol=1e-5))
                            or np.any(np.isclose(x, up, atol=1e-5)))
        print(f'near-side samples: {n_res} (train {len(train)}/hold '
              f'{len(hold)} frames); gate holdout: DT med '
              f'{med0:.2f}->{med1:.2f} p90 {p900:.2f}->{p901:.2f} valid '
              f'{vf0:.2f}->{vf1:.2f} | train med {tm0:.2f}->{tm1:.2f} p90 '
              f'{tp900:.2f}->{tp901:.2f}'
              + (' -- BOUNDARY HIT' if boundary_hit else ''))
        keep = bool(res.success and med1 < med0 and p901 < p900
                    and vf1 >= 0.8 * vf0 and not boundary_hit)
        print('  -- PUBLISH' if keep
              else '  -- REJECT (holdout not improved / evidence collapse '
                   '/ boundary hit)')
        if not keep:
            x = np.zeros(6)
        R_le_f, R_ec_f = apply_deltas(R_le0, R_ec0, np.radians(x))
        report('final', R_le_f, R_ec_f, R_le_gt, R_ec_gt)
    else:
        # ---- experimental: contact-line stage ---------------------------------
        # 3D contact lines = ground plane x facade plane intersections of the
        # stacked cloud, extracted ONCE at the base pose (physical scene
        # geometry in lidar frames -- pose errors move their PROJECTIONS, not
        # the lines themselves). Photometric counterpart = ridge search along
        # each projected line's normal (annotation-free; LSD matching was
        # tried and dropped -- LSD does not emit the wall base on this data).
        # Association frozen per pass; per-line holdout publish gate.
        # STATUS: failed GT stability (re-locatable association is
        # pose-adaptive; coherent slide invisible to its gate) -- kept for
        # the evidence-family ablation in the spec.
        rng_ln = np.random.default_rng(7)
        lines_per_frame = []
        n_lines_total = 0
        for fg in frames:
            # frames early in a log have <SWEEPS sweeps to stack; their
            # diluted clouds yield no facades (on the vehicle the stack is
            # always full)
            n_sw = int(fg.stack_sid.max()) + 1
            fg.contact_lines = (fl.contact_lines(fg, rng_ln)
                                if n_sw >= 8 else [])
            for line in fg.contact_lines:
                line['gate'] = bool(rng_ln.random() < 0.3)
            lines_per_frame.append(fg.contact_lines)
            n_lines_total += len(fg.contact_lines)
        print(f'contact lines: {n_lines_total} '
              f'({n_lines_total / len(frames):.1f}/frame)')
        if n_lines_total < 8:
            print('too few contact lines; abort')
            return

        schedule = ((10.0, 0.6), (6.0, 0.3), (4.0, 0.15))  # (win, trust deg)
        for half_win, tr_deg in schedule:
            R_le_x, R_ec_x = apply_deltas(R_le0, R_ec0, x)
            assoc = fl.associate_lines(frames, lines_per_frame, R_ec_x,
                                       R_le_x, half_win)
            n_fit = sum(e is not None and not l['gate']
                        for es, ls in zip(assoc, lines_per_frame)
                        for e, l in zip(es, ls))
            n_hold = sum(e is not None and l['gate']
                         for es, ls in zip(assoc, lines_per_frame)
                         for e, l in zip(es, ls))
            g0, b0, n0 = line_gate(frames, lines_per_frame, R_ec_x, R_le_x,
                                   half_win)
            print(f'pass win {half_win:.0f}px tr {tr_deg:.2f} deg: {n_fit} '
                  f'fit / {n_hold} holdout lines assoc, gate med '
                  f'{g0 if g0 is None else round(g0, 2)}px bias '
                  f'{b0 if b0 is None else round(b0, 2)}px (n={n0})')
            if n_fit < 6:
                print('  too few associated lines; skip pass')
                continue
            solver = LMSolver(make_line_fn(frames, assoc), x, huber=3.0)
            x_new, dof_sig, cost = solver.run(max_iter=15,
                                              max_step_deg=tr_deg)
            R_le_t, R_ec_t = apply_deltas(R_le0, R_ec0, x_new)
            report('trial', R_le_t, R_ec_t, R_le_gt, R_ec_gt)
            g1, b1, n1 = line_gate(frames, lines_per_frame, R_ec_t, R_le_t,
                                   half_win)
            med_ok = (g0 is None or g1 is None or g1 <= g0 * 1.02 + 1e-9)
            bias_ok = (b0 is None or b1 is None
                       or b1 <= max(b0 * 1.02, 0.3))
            improved = (g1 is not None and g0 is not None
                        and (g1 < g0 * 0.98
                             or (b1 < b0 * 0.8 and b0 > 0.5)))
            if med_ok and bias_ok and improved:
                x = x_new
                print(f'  gate: med {g0:.2f}->{g1:.2f}px bias '
                      f'{b0 if b0 is None else round(b0, 2)}->'
                      f'{b1 if b1 is None else round(b1, 2)}px '
                      f'(n={n1}) -- PASS kept')
            else:
                print(f'  gate: med {g0 if g0 is None else round(g0, 2)}->'
                      f'{g1 if g1 is None else round(g1, 2)}px bias '
                      f'{b0 if b0 is None else round(b0, 2)}->'
                      f'{b1 if b1 is None else round(b1, 2)}px '
                      f'(n={n1}) -- REVERTED (no improvement)')
            R_le_x, R_ec_x = apply_deltas(R_le0, R_ec0, x)
            report(f'after win {half_win:.1f}px', R_le_x, R_ec_x, R_le_gt,
                   R_ec_gt)

        R_le_f, R_ec_f = apply_deltas(R_le0, R_ec0, x)
        report('final', R_le_f, R_ec_f, R_le_gt, R_ec_gt)

    # ---- diagnostics ---------------------------------------------------------
    samples_f = extract_samples(frames, R_ec_f, R_le_f, rng_s)
    mean_f, med_f, n_f = px_rmse(frames, samples_f, R_ec_f, R_le_f)
    m1_mean, m1_med, _ = marking_px(frames, R_ec_f, R_le_f)
    print(f'depth-edge px offsets at final: mean {mean_f:.2f} median '
          f'{med_f:.2f} (n={n_f})')
    print(f'marking DT px: init median {m0_med:.2f} -> final {m1_med:.2f} '
          f'(diagnostic, not optimized)')
    if dof_sig is not None:
        print('per-DoF sigma (deg) [le r,p,y | ec r,p,y]: '
              + np.array2string(np.degrees(dof_sig), precision=3))

    # ---- persist -------------------------------------------------------------
    out_dir = ROOT / 'outputs' / 'm1'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f'{args.mode}_result.npz',
             R_le=R_le_f, R_ec=R_ec_f, x=x)
    summary = dict(
        mode=args.mode, stage=args.stage, seed=args.seed, mag=args.mag,
        frames=len(frames), sweeps=SWEEPS, n_samples=n_res,
        n_holdout=n_gate,
        init=dict(lidar=nz.geodesic_deg(R_le0, R_le_gt),
                  cam=nz.geodesic_deg(R_ec0, R_ec_gt)),
        final=dict(lidar=nz.geodesic_deg(R_le_f, R_le_gt),
                   cam=nz.geodesic_deg(R_ec_f, R_ec_gt),
                   px_mean=mean_f, px_median=med_f),
        marking_dt_px=dict(init_median=m0_med, final_median=m1_med),
        dof_sigma_deg=(np.degrees(dof_sig).tolist()
                       if dof_sig is not None else None))
    with open(out_dir / f'{args.mode}_summary.json', 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('saved:', out_dir / f'{args.mode}_summary.json')

    # ---- visualization (first frame): fixed evidence pool at final -----------
    import cv2
    fg = frames[0]
    vis = fg.img.copy()
    uv, ok = ev.project_stacked_samples(fg, R_ec_f, R_le_f, samples[0])
    d = ev.bilinear(fg.dt_map, uv[:, 0], uv[:, 1])
    good = ok & np.isfinite(d)
    for u_, v_, d_ in zip(uv[good, 0], uv[good, 1], d[good]):
        col = (0, 200, 0) if d_ < 2.0 else \
            ((0, 165, 255) if d_ < 5.0 else (0, 0, 255))
        cv2.circle(vis, (int(round(u_)), int(round(v_))), 2, col, -1)
    cv2.imwrite(str(out_dir / f'{args.mode}_vis_after.png'), vis)
    print('saved vis')


if __name__ == '__main__':
    main()
