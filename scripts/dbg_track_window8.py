# -*- coding: utf-8 -*-
"""Debug 8: are non-key CAM_FRONT images unique, and do they move?"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from auto_extrinsics.data.nuscenes_lite import NuScenesLite

ds = NuScenesLite()
recs = ds.frames_of_log_multi('n008-2018-08-01-15-16-36-0400',
                              channels=('CAM_FRONT',))

# take center 3's window records directly
sds = ds.camera_window(recs[12]['CAM_FRONT'], 13, 13)
imgs = [ds.load_image(s) for s in sds]
print('ts (s):', [round(s['timestamp'] * 1e-6, 3) for s in sds])
print('is_key:', [s['is_key_frame'] for s in sds])
for a in range(len(sds) - 1):
    d = np.abs(imgs[a].astype(np.int16) - imgs[a + 1]).mean()
    print(f'pair {a}->{a + 1}: key {sds[a]["is_key_frame"]:d}->'
          f'{sds[a + 1]["is_key_frame"]:d}  L1 diff {d:7.3f}  '
          f'file {os.path.basename(sds[a + 1]["filename"])[-30:]}')

# also global check: L1 diff between a non-key and its neighboring key
key_imgs = {s['token']: s for s in sds if s['is_key_frame']}
