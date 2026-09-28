# -*- coding: utf-8 -*-
"""Check: do consecutive 20 Hz sweeps carry DISTINCT ego poses (usable for
per-sweep motion bridging), and how much does the ego move between them?"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_extrinsics.data.nuscenes_lite import NuScenesLite

LOG = 'n008-2018-08-01-15-16-36-0400'

ns = NuScenesLite()
frames = ns.frames_of_log(LOG)
lidar_sd = frames[20]['LIDAR_TOP']
sd_of = ns._by_token('sample_data')
cs = lidar_sd['calibrated_sensor_token']

chain, tok = [], lidar_sd['token']
while tok and len(chain) < 12:
    sd = sd_of[tok]
    if sd['calibrated_sensor_token'] == cs:
        chain.append(sd)
    tok = sd['prev']
chain = chain[::-1]

prev_t, prev_p = None, None
for sd in chain:
    T = ns.ego_pose(sd)
    dt = 0.0 if prev_t is None else (sd['timestamp'] - prev_t) / 1e6
    dp = 0.0 if prev_p is None else float(np.linalg.norm(T[:3, 3] - prev_p))
    print(f"key={int(sd['is_key_frame'])} dt={dt*1000:6.1f} ms  "
          f"dp={dp*100:6.1f} cm  ego_token={sd['ego_pose_token'][:8]}")
    prev_t, prev_p = sd['timestamp'], T[:3, 3]
