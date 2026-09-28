# -*- coding: utf-8 -*-
"""Run the REFERENCE repo's own cascade (lidar_icp_handeye coarse ->
teed_stacked_refinement polish) AS-IS on all 10 v1.0-mini scenes, via the
devkit shim. Their own benchmark config (README): frames 20, noise rpy
3/-3/4 + translation, train/holdout offsets 4 6 8 / 10 12, sweeps 20,
intensity p70. Reports per-scene coarse residual, refined residual, and
the reference's own publish flags."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402

REF = Path(r'H:\projects\AutoLidarCameraCalibration')
PY = sys.executable
OUT = ROOT / 'outputs' / 'refpipe'
ENV = dict(os.environ,
           PYTHONPATH=str(ROOT / 'compat'),
           AUTOEX_NUSCENES_ROOT=r'H:\datasets\nuscenes-mini')


def run(cmd):
    return subprocess.run(cmd, cwd=str(REF), env=ENV,
                          capture_output=True, text=True)


def main():
    ds = NuScenesLite()
    scenes = ds.table('scene')
    rows = []
    for idx, s in enumerate(scenes):
        name = s['name']
        base = OUT / f'{name}'
        coarse_dir = base / 'coarse'
        refined_dir = base / 'refined'
        print(f'=== [{idx}] {name}: handeye coarse', flush=True)
        r = run([PY, 'lidar_icp_handeye.py', '--dataroot',
                 r'H:\datasets\nuscenes-mini', '--scene', str(idx),
                 '--frames', '20', '--noise-rpy-deg', '3', '-3', '4',
                 '--out', str(coarse_dir)])
        if r.returncode != 0:
            print('  handeye FAILED:', (r.stderr or '')[-300:])
            rows.append(dict(scene=name, ok=False, stage='handeye'))
            continue
        rc = json.loads((coarse_dir / 'report.json').read_text(
            encoding='utf-8'))
        print(f'  coarse residual rpy '
              f'{[round(v, 2) for v in rc["estimated_body_correction"][:3]]}'
              f' vs expected '
              f'{[round(v, 2) for v in rc["expected_body_correction"][:3]]}'
              ' (rad)', flush=True)

        print(f'=== [{idx}] {name}: teed polish', flush=True)
        r = run([PY, 'teed_stacked_refinement.py', '--dataroot',
                 r'H:\datasets\nuscenes-mini', '--scene', str(idx),
                 '--coarse-json', str(coarse_dir / 'report.json'),
                 '--cache-dir', str(OUT / f'teed_cache_{name}'),
                 '--train-offsets', '4', '6', '8',
                 '--holdout-offsets', '10', '12',
                 '--sweeps', '20', '--intensity-percentile', '70',
                 '--out', str(refined_dir)])
        if r.returncode != 0:
            print('  polish FAILED:', (r.stderr or '')[-300:])
            rows.append(dict(scene=name, ok=False, stage='polish',
                             coarse=rc['estimated_body_correction'][:3]))
            continue
        rr = json.loads((refined_dir / 'report.json').read_text(
            encoding='utf-8'))
        print(f'  publish_attitude {rr["publish_attitude"]}, refined rpy '
              f'{[round(v, 3) for v in rr["refined_error_rpy_deg"]]} deg',
              flush=True)
        rows.append(dict(
            scene=name, ok=True,
            coarse_resid_deg=[round(v, 3)
                              for v in rc['estimated_body_correction'][:3]],
            refined_resid_deg=[round(v, 3)
                               for v in rr['refined_error_rpy_deg']],
            coarse_before_deg=[round(v, 3)
                               for v in rc['expected_body_correction'][:3]],
            publish_attitude=rr['publish_attitude'],
            holdout_median_px=[rr['holdout_score_coarse']['median_px'],
                               rr['holdout_score_refined']['median_px']]))

    print('\n=============== 10-clip reference-cascade table ===============')
    print(f"{'scene':12s} {'inject(deg)':>16s} {'coarse resid':>16s} "
          f"{'publish':>7s}")
    for r in rows:
        if not r.get('ok'):
            print(f"{r['scene']:12s} FAILED @ {r['stage']}")
            continue
        inj = [round(v, 2) for v in r['coarse_before_deg']]
        res = [round(v, 2) for v in r['coarse_resid_deg']]
        print(f"{r['scene']:12s} {str(inj):>16s} {str(res):>16s} "
              f"{str(r['publish_attitude']):>7s}")
    (OUT / 'refpipe_all_scenes.json').write_text(
        json.dumps(rows, indent=2), encoding='utf-8')
    print('saved:', OUT / 'refpipe_all_scenes.json')


if __name__ == '__main__':
    main()
