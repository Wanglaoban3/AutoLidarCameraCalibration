# -*- coding: utf-8 -*-
"""Inventory: per-channel key vs non-key sample_data in the local
v1.0-mini metadata + files on disk."""
import collections
import json
from pathlib import Path

root = Path(r'H:\datasets\nuscenes-mini')
meta = root / 'v1.0-mini'
sensors = {s['token']: s['channel'] for s in json.load(
    open(meta / 'sensor.json', encoding='utf-8'))}
cstoch = {c['token']: sensors[c['sensor_token']] for c in json.load(
    open(meta / 'calibrated_sensor.json', encoding='utf-8'))}
rows = json.load(open(meta / 'sample_data.json', encoding='utf-8'))
allc = collections.Counter()
nonkey = collections.Counter()
for r in rows:
    ch = cstoch[r['calibrated_sensor_token']]
    allc[ch] += 1
    if not r['is_key_frame']:
        nonkey[ch] += 1
print('metadata sample_data: total', len(rows))
print('all      :', dict(sorted(allc.items())))
print('non-key  :', dict(sorted(nonkey.items())))
for ch in ('CAM_FRONT', 'LIDAR_TOP'):
    d = root / 'sweeps' / ch
    if d.exists():
        n = len(list(d.glob('*.*')))
        print(f'disk sweeps/{ch}: {n} files')
    d = root / 'samples' / ch
    n = len(list(d.glob('*.*')))
    print(f'disk samples/{ch}: {n} files')
