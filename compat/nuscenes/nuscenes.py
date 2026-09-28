# -*- coding: utf-8 -*-
"""Minimal stand-ins for the two devkit imports mfcalib_nuscenes.py needs
(nuscenes.NuScenes, pyquaternion.Quaternion), backed by the raw JSON
tables -- avoids installing nuscenes-devkit in a download-constrained
environment. Only the attributes that script touches are implemented.

Put this directory on PYTHONPATH when running the reference repo's
benchmark:  set PYTHONPATH=H:\\projects\\auto-extrinsics-adjusting\\compat
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = []


class _NuScenes:
    def __init__(self, version='v1.0-mini', dataroot='', verbose=False):
        self.meta = Path(dataroot) / version
        meta = self.meta
        self.tables = {}
        for name in ('scene', 'sample', 'sample_data',
                     'calibrated_sensor', 'ego_pose', 'log'):
            with open(meta / f'{name}.json', encoding='utf-8') as f:
                self.tables[name] = json.load(f)
        self.scene = self.tables['scene']
        self._by_token = {n: {r['token']: r for r in rows}
                          for n, rows in self.tables.items()}
        # devkit-derived sample['data'] map: key-frame sample_data of one
        # sample grouped by channel
        with open(meta / 'sensor.json', encoding='utf-8') as f:
            sensors = {s['token']: s['channel'] for s in json.load(f)}
        self._sensor_of_cs = {}
        for cs in self.tables['calibrated_sensor']:
            self._sensor_of_cs[cs['token']] = sensors[cs['sensor_token']]
        self._data = {}
        for sd in self.tables['sample_data']:
            if not sd['is_key_frame']:
                continue
            ch = self._sensor_of_cs[sd['calibrated_sensor_token']]
            self._data.setdefault(sd['sample_token'], {})[ch] = sd['token']
        for samp in self.tables['sample']:
            samp.setdefault('data', self._data.get(samp['token'], {}))
            # devkit also carries prev/next links already in raw JSON
        # sample['anns']: annotation tokens of one sample (devkit-derived)
        if (meta / 'sample_annotation.json').exists():
            with open(meta / 'sample_annotation.json',
                      encoding='utf-8') as f:
                anns = json.load(f)
            per = {}
            for a in anns:
                per.setdefault(a['sample_token'], []).append(a['token'])
            for samp in self.tables['sample']:
                samp.setdefault('anns', per.get(samp['token'], []))

    def get(self, table, token):
        if table not in self._by_token:
            with open(self.meta / f'{table}.json', encoding='utf-8') as f:
                rows = json.load(f)
            self.tables[table] = rows
            self._by_token[table] = {r['token']: r for r in rows}
        rec = self._by_token[table][token]
        if table == 'sample_annotation' and 'category_name' not in rec:
            # devkit joins instance -> category; raw JSON does not carry it
            inst = self.get('instance', rec['instance_token'])
            cat = self.get('category', inst['category_token'])
            rec['category_name'] = cat['name']
        return rec


class NuScenes(_NuScenes):
    pass


class Quaternion:
    """pyquaternion-compatible constructor for [w, x, y, z] lists or the
    devkit dict form; only rotation_matrix is used."""

    def __init__(self, q):
        if isinstance(q, dict):
            q = [q['w'], q['x'], q['y'], q['z']]
        self.q = [float(v) for v in q]

    @property
    def rotation_matrix(self):
        w, x, y, z = self.q
        return Rotation.from_quat([x, y, z, w]).as_matrix()
