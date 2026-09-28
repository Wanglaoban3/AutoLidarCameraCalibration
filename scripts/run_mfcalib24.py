# -*- coding: utf-8 -*-
"""MFCalib-2024 port runner (beam-divergence-inflated point-to-line).

Modes
  falsify      THE GATE: discriminability test BEFORE any optimization.
               4 keyframes of scene-0103 CAM_FRONT; depth-edge samples of
               the stacked cloud, the paper's inflation applied per point
               (HDL-32E divergence 1.29 mrad, incidence-elongated), poses
               GT vs +-0.5 deg per DoF (per sensor) vs +-0.5 deg aggregate;
               point-to-line statistics at GT vs perturbed. GO requires GT
               to beat perturbed clearly (>=1.5x median or monotone V
               curves). NO-GO stops the method here.
  gt           zero-noise stability: init = GT, the published 20->2 px
               stage schedule must hold the pose within 0.10 deg/sensor.
  real_coarse  init from the actual coarse outputs
               (extrinsic_recovery/results/*_recovered.npy, ~0.507/0.364
               deg); publish gate = odd-frame holdout improves, no bound
               hit.

Usage (from repo root):
  python scripts/run_mfcalib24.py --mode falsify
  python scripts/run_mfcalib24.py --mode gt
  python scripts/run_mfcalib24.py --mode real_coarse
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite      # noqa: E402
from auto_extrinsics import noise as nz                          # noqa: E402
from auto_extrinsics.fine import evidence as ev                  # noqa: E402
from auto_extrinsics.fine import mfcalib24 as mf                 # noqa: E402
from auto_extrinsics.fine.geometry import FrameGeom              # noqa: E402
from auto_extrinsics.fine.solver import apply_deltas             # noqa: E402
import run_m1                                                    # noqa: E402

LOG = 'n008-2018-08-01-15-16-36-0400'   # Boston row houses (scene-0103)
SWEEPS = 10                              # 0.5 s of 20 Hz sweeps per frame
PERTURB_DEG = 0.5                        # falsification perturbation


def build_frames(ds, recs):
    """run_m1.build_frames without TEED/dyn-mask (Canny-based objective,
    annotation-free): image + 10-sweep stacked cloud per frame."""
    frames = []
    for k, rec in enumerate(recs):
        cam_sd, lid_sd = rec['CAM_FRONT'], rec['LIDAR_TOP']
        cal_c = ds.calib(cam_sd, 'CAM_FRONT')
        cal_l = ds.calib(lid_sd, 'LIDAR_TOP')
        img = ds.load_image(cam_sd)
        fg = FrameGeom(
            name=f'f{k:02d}', K=cal_c['K'],
            R_ec=cal_c['R_cs'], t_ec=cal_c['t_cs'],
            ego_c=ds.ego_pose(cam_sd),
            R_le=cal_l['R_cs'], t_le=cal_l['t_cs'],
            ego_l=ds.ego_pose(lid_sd), img_shape=img.shape[:2])
        fg.img = img
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


def prepare(frames, base_R_ec, base_R_le, rng, side, max_samples,
            gray_thr=mf.GRAY_THR, len_thr=mf.LEN_THR):
    """Fixed evidence at the base pose: samples, context (inflation),
    Canny edge pixels + KD-tree per frame. dyn_mask stays None."""
    samples, ctxs, edges, n_con = [], [], [], 0
    for fg in frames:
        fg.R_ec0, fg.R_le0 = base_R_ec.copy(), base_R_le.copy()
        s = ev.depth_edge_samples_dense(fg, base_R_ec, base_R_le,
                                        max_samples=max_samples, rng=rng,
                                        near_side=(side == 'near'))
        samples.append(s)
        ctxs.append(mf.frame_context(fg, s))
        uv, n_raw, n_kept = mf.canny_edge_pixels(fg.img, gray_thr=gray_thr,
                                                 len_thr=len_thr)
        edges.append((uv, cKDTree(uv)))
        n_con += n_kept
        print(f'  {fg.name}: {len(s["p_idx"])} depth-edge samples '
              f'({side} side), canny {n_raw} px raw, {n_kept} contours '
              f'kept ({len(uv)} px)')
    print(f'  total: {sum(len(s["p_idx"]) for s in samples)} samples, '
          f'{n_con} kept contours')
    return samples, ctxs, edges


def falsify(ds, recs, args):
    """The GO/NO-GO measurement (see module docstring)."""
    step = max(1, len(recs) // 4)
    recs4 = recs[::step][:4]
    tss = [r['sample']['timestamp'] for r in recs4]
    print(f'falsify: {len(recs4)} keyframes (timestamps {tss})')
    frames = build_frames(ds, recs4)
    cal_c = ds.calib(recs4[0]['CAM_FRONT'], 'CAM_FRONT')
    cal_l = ds.calib(recs4[0]['LIDAR_TOP'], 'LIDAR_TOP')
    R_ec_gt, R_le_gt = cal_c['R_cs'], cal_l['R_cs']

    out = dict(log=LOG, theta_div_mrad=mf.THETA_DIV * 1e3,
               perturb_deg=PERTURB_DEG,
               canny=dict(gray_thr=args.gray_thr, len_thr=args.len_thr),
               sides={})
    for side in ('far', 'near'):
        rng = np.random.default_rng(0)
        samples, ctxs, edges = prepare(frames, R_ec_gt, R_le_gt, rng, side,
                                       args.max_samples,
                                       gray_thr=args.gray_thr,
                                       len_thr=args.len_thr)
        print(f'  inflation: e_m median '
              f'{np.median(np.concatenate([c["e_m"] for c in ctxs])) * 1e2:.1f}'
              f' cm, corridor sigma median '
              f'{np.median(np.concatenate([c["sigma_px"] for c in ctxs])):.2f}'
              f' px, cos(alpha) p10 '
              f'{np.percentile(np.concatenate([c["cos_a"] for c in ctxs]), 10):.2f}')

        zero = np.zeros(6)
        poses = [('gt', zero)]
        for sensor, off in (('le', 0), ('ec', 3)):
            for dof, ax in enumerate('rpy'):
                for sgn in (+1, -1):
                    x = zero.copy()
                    x[off + dof] = np.radians(sgn * PERTURB_DEG)
                    poses.append((f'{sensor}_{ax}{sgn:+d}', x))
        for sgn in (+1, -1):
            poses.append((f'all{sgn:+d}',
                          np.full(6, np.radians(sgn * PERTURB_DEG)
                                  / np.sqrt(3))))

        variants = dict(
            base=0.0,           # plain point-to-line (simplified port)
            pull=mf.OFFSET_PULL,    # paper correction: subtract e_i on beam
            push=-mf.OFFSET_PULL)   # opposite sign (sign not identifiable)
        tab = {}
        for vname, sgn in variants.items():
            rows = {}
            for pname, x in poses:
                R_le, R_ec = apply_deltas(R_le_gt, R_ec_gt, x)
                st = mf.pose_stats(frames, ctxs, samples, edges, R_ec, R_le,
                                   offset_sign=sgn, gate_tau=args.gate_tau)
                rows[pname] = st
            tab[vname] = rows
            gt = rows['gt']
            print(f'  [{side}/{vname}] GT: n={gt["n"]} med {gt["med"]:.2f} '
                  f'med20 {gt["med20"]:.2f} frac5 {gt["frac5"]:.3f} '
                  f'frac10 {gt["frac10"]:.3f} frac_sig {gt["frac_sig"]:.3f} '
                  f'match(tau={args.gate_tau:.0f}) {gt["frac_match"]:.3f} '
                  f'med_match {gt["med_match"]:.2f} '
                  f'(sigma med {gt["med_sig"]:.2f} px)')
            for pname, _ in poses[1:]:
                r = rows[pname]
                dm = r['med20'] / max(gt['med20'], 1e-9)
                df = gt['frac_sig'] / max(r['frac_sig'], 1e-9)
                dmM = r['med_match'] / max(gt['med_match'], 1e-9)
                print(f'    {pname:>7s}: med20 {r["med20"]:6.2f} '
                      f'({dm:4.2f}x) frac10 {r["frac10"]:.3f} '
                      f'frac_sig {r["frac_sig"]:.3f} (gt/r {df:4.2f}x) '
                      f'match {r["frac_match"]:.3f} '
                      f'med_match {r["med_match"]:6.2f} ({dmM:4.2f}x)')
        out['sides'][side] = dict(
            per_variant={
                vn: {pn: {k: (round(vv, 4) if isinstance(vv, float) else vv)
                          for k, vv in st.items()}
                     for pn, st in rows.items()}
                for vn, rows in tab.items()})

        # GO assessment per variant: monotone V across per-DoF curves on
        # the truncated median (>=1.15x both sides) or corridor fraction.
        assess = {}
        for vn, rows in tab.items():
            gt = rows['gt']
            v_med, v_sig, v_mm = 0, 0, 0
            for sensor, off in (('le', 0), ('ec', 3)):
                for dof, ax in enumerate('rpy'):
                    for sgn in (+1, -1):
                        r = rows[f'{sensor}_{ax}{sgn:+d}']
                        if (r['med20'] >= 1.15 * gt['med20']):
                            v_med += 1
                        if (r['frac_sig'] <= 0.85 * gt['frac_sig']):
                            v_sig += 1
                        if (r['med_match'] >= 1.15 * gt['med_match']):
                            v_mm += 1
            agg = max(
                rows['all+1']['med20'] / max(gt['med20'], 1e-9),
                rows['all-1']['med20'] / max(gt['med20'], 1e-9),
                rows['all+1']['med_match'] / max(gt['med_match'], 1e-9),
                rows['all-1']['med_match'] / max(gt['med_match'], 1e-9),
                gt['frac_sig'] / max(rows['all+1']['frac_sig'], 1e-9),
                gt['frac_sig'] / max(rows['all-1']['frac_sig'], 1e-9))
            assess[vn] = dict(v_shape_med20_of_12=v_med,
                              v_shape_fracsig_of_12=v_sig,
                              v_shape_medmatch_of_12=v_mm,
                              best_aggregate_ratio=round(agg, 3),
                              clear=bool(agg >= 1.5))
            print(f'  [{side}/{vn}] V-shaped DoF curves (>=1.15x): med20 '
                  f'{v_med}/12, frac_sig {v_sig}/12, med_match {v_mm}/12; '
                  f'best aggregate ratio {agg:.2f}x')
        out['sides'][side]['assessment'] = assess
        del samples, ctxs, edges

    go = any(a['clear'] for sd in out['sides'].values()
             for a in sd['assessment'].values())
    out['go'] = bool(go)
    print(f'FALSIFY VERDICT: {"GO" if go else "NO-GO"}')
    return out


def calibrate(ds, recs, args, mode):
    """Published stage schedule on the fixed evidence pool (12 frames)."""
    step = max(1, len(recs) // args.frames)
    recs = recs[::step][:args.frames]
    print(f'{len(recs)} frames for calibration')
    frames = build_frames(ds, recs)
    cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
    cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')
    R_ec_gt, R_le_gt = cal_c['R_cs'], cal_l['R_cs']

    if mode == 'real_coarse':
        coarse_dir = ROOT / 'extrinsic_recovery' / 'results'
        R_ec0 = np.load(coarse_dir / 'camera_R_ec_recovered.npy')
        R_le0 = np.load(coarse_dir / 'lidar_R_le_recovered.npy')
        print('init: actual coarse outputs')
    else:
        R_ec0, R_le0 = R_ec_gt.copy(), R_le_gt.copy()
        print('init: GT (zero-noise stability check)')
    run_m1.report('init', R_le0, R_ec0, R_le_gt, R_ec_gt)
    for fg in frames:
        fg.R_ec0, fg.R_le0 = R_ec0.copy(), R_le0.copy()

    rng = np.random.default_rng(0)
    samples, ctxs, edges = prepare(frames, R_ec0, R_le0, rng, args.side,
                                   args.max_samples)
    x0 = np.zeros(6)

    # holdout = odd frames (publish discipline)
    h_frames = [i for i in range(1, len(frames), 2)]
    hctx = [ctxs[i] for i in h_frames]
    hsam = [samples[i] for i in h_frames]
    hedge = [edges[i] for i in h_frames]
    hframes = [frames[i] for i in h_frames]
    med0, n0 = mf.holdout_metrics(hframes, hctx, hsam, hedge,
                                  R_le0, R_ec0, x0, tau=args.gate_tau,
                                  offset_sign=args.offset_sign)
    print(f'holdout init: med '
          f'{med0 if med0 is None else round(med0, 2)} px n={n0}')

    x, history = mf.run_schedule(frames, ctxs, samples, edges,
                                 R_le0, R_ec0, x0,
                                 offset_sign=args.offset_sign,
                                 verbose=True)
    R_le_t, R_ec_t = apply_deltas(R_le0, R_ec0, x)
    run_m1.report('trial', R_le_t, R_ec_t, R_le_gt, R_ec_gt)

    med1, n1 = mf.holdout_metrics(hframes, hctx, hsam, hedge,
                                  R_le0, R_ec0, x, tau=args.gate_tau,
                                  offset_sign=args.offset_sign)
    bound_hit = bool(any(h.get('bound_hit') for h in history))
    med_ok = med0 is not None and med1 is not None and med1 < med0
    count_ok = n1 >= 0.8 * max(n0, 1)
    publish = bool(med_ok and count_ok and not bound_hit)
    print(f'holdout final: med {med1 if med1 is None else round(med1, 2)} '
          f'px n={n1} | med_ok {med_ok} count_ok {count_ok} '
          f'bound_hit {bound_hit} -> {"PUBLISH" if publish else "REJECT"}')

    if publish:
        R_le_f, R_ec_f = R_le_t, R_ec_t
        x_f = x
    else:
        R_le_f, R_ec_f = R_le0, R_ec0
        x_f = x0
    run_m1.report('final', R_le_f, R_ec_f, R_le_gt, R_ec_gt)

    d_le = nz.geodesic_deg(R_le_f, R_le_gt)
    d_ec = nz.geodesic_deg(R_ec_f, R_ec_gt)
    stable = dict(lidar=d_le, cam=d_ec)
    if mode == 'gt':
        stable['within_0.10'] = bool(d_le <= 0.10 and d_ec <= 0.10)
        print(f"gt stability: lidar {d_le:.3f} cam {d_ec:.3f} deg -> "
              f"{'OK' if stable['within_0.10'] else 'VIOLATED'} "
              f'(limit 0.10 deg/sensor)')
    return dict(
        mode=mode, side=args.side, frames=len(frames),
        sweeps=SWEEPS, max_samples=args.max_samples,
        offset_sign=args.offset_sign,
        n_samples=sum(len(s['p_idx']) for s in samples),
        init=dict(lidar=nz.geodesic_deg(R_le0, R_le_gt),
                  cam=nz.geodesic_deg(R_ec0, R_ec_gt)),
        trial=dict(lidar=nz.geodesic_deg(R_le_t, R_le_gt),
                   cam=nz.geodesic_deg(R_ec_t, R_ec_gt)),
        final=dict(lidar=d_le, cam=d_ec),
        rotvec_final_deg=np.degrees(x_f).round(3).tolist(),
        holdout=dict(med_init=med0, med_final=med1, n_init=n0, n_final=n1,
                     publish=publish, bound_hit=bound_hit),
        history=history,
        gt_stability=stable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='falsify',
                    choices=['falsify', 'gt', 'real_coarse'])
    ap.add_argument('--side', default='far', choices=['far', 'near'],
                    help='depth-edge side for gt/real_coarse '
                         '(falsify runs both)')
    ap.add_argument('--offset-sign', type=float, default=mf.OFFSET_PULL,
                    help='radial divergence correction sign for the '
                         'optimizer (falsify tests both)')
    ap.add_argument('--max-samples', type=int, default=1500)
    ap.add_argument('--gate-tau', type=float, default=20.0)
    ap.add_argument('--frames', type=int, default=12)
    ap.add_argument('--gray-thr', type=int, default=mf.GRAY_THR,
                    help='Canny low threshold (reference 20); falsify '
                         'sensitivity runs may raise it')
    ap.add_argument('--len-thr', type=int, default=mf.LEN_THR,
                    help='kept-contour length threshold (reference 100)')
    args = ap.parse_args()

    ds = NuScenesLite()
    recs = ds.frames_of_log(LOG)
    print(f'{len(recs)} key frames from {LOG}')

    out_path = ROOT / 'outputs' / 'm1' / 'mfcalib24_summary.json'
    summary = {}
    if out_path.exists():
        summary = json.loads(out_path.read_text(encoding='utf-8'))

    if args.mode == 'falsify':
        summary.setdefault('falsify', {})
        tag = f'gray{args.gray_thr}_len{args.len_thr}'
        summary['falsify'][tag] = falsify(ds, recs, args)
        summary['falsify'][tag]['go'] = any(
            a['clear']
            for sd in summary['falsify'][tag]['sides'].values()
            for a in sd['assessment'].values())
        print(f"FALSIFY[{tag}] VERDICT: "
              f"{'GO' if summary['falsify'][tag]['go'] else 'NO-GO'}")
    else:
        res = calibrate(ds, recs, args, args.mode)
        summary[args.mode] = res
        summary.setdefault('config', dict(
            theta_div_mrad=mf.THETA_DIV * 1e3, kappa=mf.KAPPA,
            canny=dict(gray_thr=mf.GRAY_THR, len_thr=mf.LEN_THR),
            schedule='20..2 px step 1, second solve only if moved'))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('saved:', out_path)


if __name__ == '__main__':
    main()
