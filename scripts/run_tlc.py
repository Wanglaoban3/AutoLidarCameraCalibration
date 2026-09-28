# -*- coding: utf-8 -*-
"""TLC-Calib experiment: line-to-line targetless fine calibration (port).

Structure copied from run_m1.py; objective/matching from
auto_extrinsics/fine/tlc.py (TLC-Calib port, frozen one-shot association).

Modes
  gt            init = GT (zero-noise stability check: must stay < 0.10
                deg/sensor; this gate is REQUIRED to pass before real_coarse).
  real_coarse   init from the ACTUAL coarse outputs
                (extrinsic_recovery/results/*_recovered.npy; cam seed 7 /
                lidar seed 11 noise replicated) -- expect ~0.507 / 0.364 deg.

Stages
  measure       extract contact lines + image segments, match at the base
                pose with BOTH variants, report yields (no solve).
  run           match with the chosen variant, solve with the frozen
                objective, apply the holdout publish gate, save
                outputs/m1/tlc_summary.json.

Usage (from repo root):
  python scripts/run_tlc.py --stage measure --mode real_coarse
  python scripts/run_tlc.py --mode gt --variant a
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
from auto_extrinsics.fine import lines as fl                 # noqa: E402
from auto_extrinsics.fine import tlc as tl                   # noqa: E402
from auto_extrinsics.fine.solver import LMSolver, apply_deltas  # noqa: E402
import run_m1 as m1                                          # noqa: E402

LOG = m1.LOG
N_FRAMES = m1.N_FRAMES
OUT_DIR = ROOT / 'outputs' / 'm1'


def yield_report(tag, per_frame, extra):
    tot = sum(per_frame)
    return dict(total=tot, per_frame=tot / max(len(per_frame), 1),
                frames_active=int(sum(c > 0 for c in per_frame)), **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='real_coarse',
                    choices=['gt', 'real_coarse'])
    ap.add_argument('--stage', default='run', choices=['run', 'measure'])
    ap.add_argument('--variant', default='a', choices=['a', 'b'],
                    help='a: line-to-line (faithful), b: point-to-line '
                         '(fallback, per-sample nearest support)')
    args = ap.parse_args()
    t00 = time.perf_counter()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = NuScenesLite()
    recs = ds.frames_of_log(LOG)
    step = max(1, len(recs) // N_FRAMES)
    recs = recs[::step][:N_FRAMES]
    print(f'{len(recs)} frames from {LOG}')

    teed = ev.load_teed(device)
    frames = m1.build_frames(ds, recs, teed, device)
    t_build = time.perf_counter() - t00

    cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
    cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')
    R_ec_gt, R_le_gt = cal_c['R_cs'], cal_l['R_cs']

    # ---- noise + init (run_m1 structure) ------------------------------------
    if args.mode == 'real_coarse':
        coarse_dir = ROOT / 'extrinsic_recovery' / 'results'
        R_ec0 = np.load(coarse_dir / 'camera_R_ec_recovered.npy')
        R_le0 = np.load(coarse_dir / 'lidar_R_le_recovered.npy')
        print('init: actual coarse outputs (cam seed 7 / lidar seed 11 '
              'replicated)')
    else:
        R_ec0, R_le0 = R_ec_gt.copy(), R_le_gt.copy()
        print('init: GT (zero-noise stability check: pose must stay)')

    m1.report('init', R_le0, R_ec0, R_le_gt, R_ec_gt)
    for fg in frames:
        fg.R_ec0 = R_ec0.copy()
        fg.R_le0 = R_le0.copy()

    # ---- fixed evidence: 3D contact lines + 2D segments, extracted ONCE -----
    rng_ln = np.random.default_rng(7)
    rng_gate = np.random.default_rng(123)
    t0 = time.perf_counter()
    lines_per_frame = []
    for fg in frames:
        n_sw = int(fg.stack_sid.max()) + 1
        fg.contact_lines = (fl.contact_lines(fg, rng_ln)
                            if n_sw >= 8 else [])
        for line in fg.contact_lines:
            line['gate'] = bool(rng_gate.random() < 0.3)  # holdout lines
        lines_per_frame.append(fg.contact_lines)
    n_lines = sum(len(l) for l in lines_per_frame)
    n_gate_lines = sum(l['gate'] for ls in lines_per_frame for l in ls)
    seg_methods, n_segs = {}, []
    lsd_err = ''
    for fg in frames:
        fg.tlc_segs, method, err = tl.image_segments(fg)
        if err:
            lsd_err = err
        fg.tlc_seg_arr = tl.seg_arrays(fg.tlc_segs)
        seg_methods[method] = seg_methods.get(method, 0) + 1
        n_segs.append(len(fg.tlc_segs))
    t_extract = time.perf_counter() - t0
    if lsd_err:
        print(f'LSD rejected: {lsd_err}')
    print(f'contact lines: {n_lines} ({n_lines / len(frames):.1f}/frame, '
          f'{n_gate_lines} holdout) | image segments: '
          f'{np.mean(n_segs):.0f}/frame ({sum(n_segs)} total, method '
          f'{seg_methods})')
    if n_lines < 8:
        print('too few contact lines; abort')
        return

    x0 = np.zeros(6)
    R_le_b, R_ec_b = apply_deltas(R_le0, R_ec0, x0)

    # ---- matching (ONCE, at the base pose; frozen for the whole run) --------
    t0 = time.perf_counter()
    ent_a, per_a = tl.match_line2line(frames, lines_per_frame, R_ec_b, R_le_b)
    ent_b, per_b, smp_b = tl.match_point2line(frames, lines_per_frame,
                                              R_ec_b, R_le_b)
    t_match = time.perf_counter() - t0
    y_a = yield_report('A', per_a, dict(entries=len(ent_a)))
    y_b = yield_report('B', per_b, dict(entries=len(ent_b), samples=smp_b))
    print(f'match yield variant A (line-to-line): {y_a["total"]} matches, '
          f'{y_a["per_frame"]:.2f}/frame, {y_a["frames_active"]} frames')
    print(f'match yield variant B (point-to-line): {y_b["total"]} groups, '
          f'{y_b["per_frame"]:.2f}/frame, {y_b["samples"]} samples, '
          f'{y_b["frames_active"]} frames')
    if args.stage == 'measure':
        if ent_b:
            stds = np.array([e['std0'] for e in ent_b])
            offs = np.array([abs(e['r0_med']) for e in ent_b])
            print(f'B group quality: offset |med| p50/p90 '
                  f'{np.percentile(offs, 50):.1f}/{np.percentile(offs, 90):.1f}'
                  f' px, collinearity std p50/p90 '
                  f'{np.percentile(stds, 50):.2f}/'
                  f'{np.percentile(stds, 90):.2f} px')
        for fi, (ea, eb) in enumerate(zip(per_a, per_b)):
            det = [round(e['dmid'], 1) for e in ent_a
                   if e['frame'] is frames[fi]]
            print(f'  f{fi:02d}: lines {len(lines_per_frame[fi])} segs '
                  f'{n_segs[fi]} A-matches {ea} {det} B-groups {eb}')
        summary = dict(mode=args.mode, stage='measure', frames=len(frames),
                       n_contact_lines=n_lines,
                       n_contact_lines_holdout=n_gate_lines,
                       lines_per_frame=n_lines / len(frames),
                       segs_per_frame=float(np.mean(n_segs)),
                       seg_methods=seg_methods,
                       match_yield=dict(line2line=y_a, point2line=y_b),
                       timing=dict(build_s=t_build, extract_s=t_extract,
                                   match_s=t_match,
                                   total_s=time.perf_counter() - t00))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / 'tlc_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print('saved:', OUT_DIR / 'tlc_summary.json')
        return

    entries = ent_a if args.variant == 'a' else ent_b
    per_frame = per_a if args.variant == 'a' else per_b
    n_train = sum(not e['gate'] for e in entries)
    n_hold = sum(e['gate'] for e in entries)
    print(f'variant {args.variant}: {n_train} train / {n_hold} holdout '
          f'entries')
    if n_train < 12:
        print('STARVED: < 12 train entries; not solving. See measure stage.')
        summary = dict(mode=args.mode, stage='run', variant=args.variant,
                       published=False, starved=True, frames=len(frames),
                       n_contact_lines=n_lines, segs_per_frame=float(
                           np.mean(n_segs)), seg_methods=seg_methods,
                       match_yield=dict(line2line=y_a, point2line=y_b),
                       timing=dict(total_s=time.perf_counter() - t00))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / 'tlc_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print('saved:', OUT_DIR / 'tlc_summary.json')
        return

    # ---- solve with the frozen objective ------------------------------------
    g0, b0, n0 = tl.gate_metrics(entries, R_ec_b, R_le_b)
    print(f'gate at init: med {g0 if g0 is None else round(g0, 2)}px bias '
          f'{b0 if b0 is None else round(b0, 2)}px (n={n0})')
    t0 = time.perf_counter()
    solver = LMSolver(tl.make_obj_fn(entries, R_le0, R_ec0), x0,
                      huber=3.0, prior_sigma_deg=0.5)
    x, dof_sig, cost = solver.run(max_iter=25, max_step_deg=0.5)
    t_solve = time.perf_counter() - t0
    R_le_t, R_ec_t = apply_deltas(R_le0, R_ec0, x)
    m1.report('trial', R_le_t, R_ec_t, R_le_gt, R_ec_gt)
    g1, b1, n1 = tl.gate_metrics(entries, R_ec_t, R_le_t)
    bound_hit = bool(np.any(np.abs(x) > np.radians(tl.BOUND_DEG)))
    med_ok = (g0 is not None and g1 is not None and g1 < g0)
    bias_ok = (b0 is None or b1 is None or b1 <= max(0.3, b0 * 1.02))
    n_ok = (n1 or 0) >= tl.MIN_GATE_RES
    publish = bool(med_ok and bias_ok and n_ok and not bound_hit)
    print(f'gate: med {g0 if g0 is None else round(g0, 2)}->'
          f'{g1 if g1 is None else round(g1, 2)}px bias '
          f'{b0 if b0 is None else round(b0, 2)}->'
          f'{b1 if b1 is None else round(b1, 2)}px (n={n1}) '
          f'bound_hit={bound_hit} -> '
          + ('PUBLISH' if publish else 'REJECT'))
    if not publish:
        x = x0
    R_le_f, R_ec_f = apply_deltas(R_le0, R_ec0, x)
    m1.report('final', R_le_f, R_ec_f, R_le_gt, R_ec_gt)

    gt_stable = None
    if args.mode == 'gt':
        d_le = nz.geodesic_deg(R_le_f, R_le_gt)
        d_ec = nz.geodesic_deg(R_ec_f, R_ec_gt)
        gt_stable = bool(d_le < 0.10 and d_ec < 0.10)
        print(f'GT stability: lidar {d_le:.3f} deg cam {d_ec:.3f} deg -> '
              + ('PASS' if gt_stable else 'FAIL'))

    print('per-DoF sigma (deg) [le r,p,y | ec r,p,y]: '
          + (np.array2string(np.degrees(dof_sig), precision=3)
             if dof_sig is not None else 'n/a'))

    summary = dict(
        mode=args.mode, stage='run', variant=args.variant,
        published=publish, gt_stable=gt_stable, bound_hit=bound_hit,
        frames=len(frames), n_contact_lines=n_lines,
        n_contact_lines_holdout=n_gate_lines,
        lines_per_frame=n_lines / len(frames),
        segs_per_frame=float(np.mean(n_segs)), seg_methods=seg_methods,
        n_entries=dict(train=n_train, holdout=n_hold),
        match_yield=dict(line2line=y_a, point2line=y_b),
        gate=dict(med_init=g0, bias_init=b0, med_final=g1, bias_final=b1,
                  n_holdout_res=n1),
        init=dict(lidar=nz.geodesic_deg(R_le0, R_le_gt),
                  cam=nz.geodesic_deg(R_ec0, R_ec_gt)),
        final=dict(lidar=nz.geodesic_deg(R_le_f, R_le_gt),
                   cam=nz.geodesic_deg(R_ec_f, R_ec_gt)),
        x_deg=np.degrees(x).tolist(),
        dof_sigma_deg=(np.degrees(dof_sig).tolist()
                       if dof_sig is not None else None),
        timing=dict(build_s=t_build, extract_s=t_extract, match_s=t_match,
                    solve_s=t_solve, total_s=time.perf_counter() - t00))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'tlc_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('saved:', OUT_DIR / 'tlc_summary.json')
    print(f'total {time.perf_counter() - t00:.0f}s '
          f'(build {t_build:.0f} extract {t_extract:.0f} match '
          f'{t_match:.1f} solve {t_solve:.0f})')


if __name__ == '__main__':
    main()
