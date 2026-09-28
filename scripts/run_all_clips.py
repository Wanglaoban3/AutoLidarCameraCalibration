# -*- coding: utf-8 -*-
"""10-clip batch harness (method-agnostic).

v1.0-mini = 10 scenes over 8 logs (n015-2018-11-21 holds three scenes),
so the loop enumerates SCENES. For each clip it invokes the runner
subprocess with --log/--scene/--mode/--seed/--mag, then collects the
runner's summary json (naming template configurable) and prints the
verdict table the M1 acceptance needs: per-clip init -> final, published
y/n, so "not just some clips good" is measurable."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402

PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runner', default='scripts/run_m1_multi.py')
    ap.add_argument('--mode', default='fresh')
    ap.add_argument('--mag', type=float, default=1.0)
    ap.add_argument('--seed-base', type=int, default=1000,
                    help='clip i gets seed base+i (reproducible)')
    ap.add_argument('--frames', type=int, default=12)
    ap.add_argument('--summary', default='{log4}_{scene}_{mode}_summary.json',
                    help='summary filename template under outputs/m1')
    ap.add_argument('--only', default=None, help='comma list of scene names')
    args = ap.parse_args()

    ds = NuScenesLite()
    scenes = [(s['name'], logs) for s, logs in
              ((s, next(l['logfile'] for l in ds.table('log')
                        if l['token'] == s['log_token']))
               for s in ds.table('scene'))]
    if args.only:
        keep = set(args.only.split(','))
        scenes = [sc for sc in scenes if sc[0] in keep]
    print(f'{len(scenes)} clips x {args.runner} ({args.mode}, '
          f'mag {args.mag})')

    rows = []
    for i, (scene, log) in enumerate(scenes):
        out_json = ROOT / 'outputs' / 'm1' / args.summary.format(
            log4=log[:4], scene=scene, mode=args.mode)
        cmd = [PY, str(ROOT / args.runner), '--log', log, '--scene', scene,
               '--mode', args.mode, '--mag', str(args.mag),
               '--seed', str(args.seed_base + i), '--frames',
               str(args.frames)]
        print(f'=== {scene} ({log}) seed {args.seed_base + i}')
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0 or not out_json.exists():
            print(f'  runner FAILED for {scene}')
            rows.append(dict(scene=scene, log=log, ok=False))
            continue
        s = json.loads(out_json.read_text(encoding='utf-8'))
        init = s['init']
        final = s['final']
        pub = s.get('gate', {}).get('published', None)
        if 'mean_rel' in init:      # multi-camera schema
            row = dict(scene=scene, log=log, ok=True, published=pub,
                       init=init['mean_rel'], final=final['mean_rel'],
                       init_abs=init['mean_abs'],
                       final_abs=final['mean_abs'])
        else:                        # single-cam schema (lidar/cam)
            row = dict(scene=scene, log=log, ok=True, published=pub,
                       init=(init['lidar'] + init['cam']) / 2,
                       final=(final['lidar'] + final['cam']) / 2)
        rows.append(row)
        print(f"  init {row['init']:.3f} -> final {row['final']:.3f} deg "
              f"(published={pub})")

    print('\n================ 10-clip verdict table ================')
    print(f"{'scene':14s} {'init':>7s} {'final':>7s} {'pub':>5s}")
    n_pub = n_ok = 0
    for r in rows:
        if not r.get('ok'):
            print(f"{r['scene']:14s} FAILED")
            continue
        n_ok += 1
        n_pub += bool(r.get('published'))
        print(f"{r['scene']:14s} {r['init']:7.3f} {r['final']:7.3f} "
              f"{str(bool(r.get('published'))):>5s}")
    fins = [r['final'] for r in rows if r.get('ok')]
    if fins:
        print(f'\npublished {n_pub}/{n_ok}; final mean '
              f'{sum(fins) / len(fins):.3f} deg, max {max(fins):.3f} deg')
    out = ROOT / 'outputs' / 'm1' / f'batch_{args.mode}.json'
    out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('saved:', out)


if __name__ == '__main__':
    main()
