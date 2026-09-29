# -*- coding: utf-8 -*-
"""Debug 10: the two scale-settling numbers.
1) median SIFT-match flow between consecutive keyframes (images only)
2) raw single-sweep lidar range percentiles (no ego bridging)."""
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

sift = cv2.SIFT_create(nfeatures=6000)
bf = cv2.BFMatcher()
for base in (12, 20):
    a = recs[base]['CAM_FRONT']
    b = recs[base + 1]['CAM_FRONT']
    ia = cv2.cvtColor(ds.load_image(a), cv2.COLOR_BGR2GRAY)
    ib = cv2.cvtColor(ds.load_image(b), cv2.COLOR_BGR2GRAY)
    ka, da = sift.detectAndCompute(ia, None)
    kb, db = sift.detectAndCompute(ib, None)
    m2 = bf.knnMatch(da, db, k=2)
    good = [m[0] for m in m2 if len(m) == 2
            and m[0].distance < 0.8 * m[1].distance]
    uva = np.float32([ka[m.queryIdx].pt for m in good])
    uvb = np.float32([kb[m.trainIdx].pt for m in good])
    flow = np.linalg.norm(uvb - uva, axis=1)
    dt = (b['timestamp'] - a['timestamp']) * 1e-6
    print(f'key pair {base}->{base + 1}: {len(good)} matches, dt '
          f'{dt * 1000:.0f} ms, flow px [10/50/90] '
          f'{np.percentile(flow, [10, 50, 90]).round(1)}')

lid = recs[12]['LIDAR_TOP']
arr = ds.load_sweep(lid)
r = np.linalg.norm(arr[:, :2], axis=1)
print(f'raw single sweep LIDAR_TOP: {len(arr)} pts, horizontal range '
      f'[10/50/90] {np.percentile(r, [10, 50, 90]).round(1)} m')
print('x-range pct', np.percentile(arr[:, 0], [10, 50, 90]).round(1),
      '| y-range pct', np.percentile(arr[:, 1], [10, 50, 90]).round(1))
