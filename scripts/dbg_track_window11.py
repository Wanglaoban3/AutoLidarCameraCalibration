# -*- coding: utf-8 -*-
"""Debug 11: metadata consistency of the three ego pose sources.
Keyframe-to-keyframe steps vs non-key camera steps vs lidar sweep steps
-- all must agree on the same trajectory if the mini build is sane."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from auto_extrinsics.data.nuscenes_lite import NuScenesLite

ds = NuScenesLite()
LOG = 'n008-2018-08-01-15-16-36-0400'
recs = ds.frames_of_log_multi(LOG, channels=('CAM_FRONT',))

# 1. keyframe-to-keyframe ego displacement (2 Hz)
kf = []
for r in recs[:20]:
    e = ds.ego_pose(r['CAM_FRONT'])
    kf.append(e[:3, 3])
d_kf = [float(np.linalg.norm(kf[i + 1] - kf[i])) for i in range(len(kf) - 1)]
dts_kf = [(recs[i + 1]['CAM_FRONT']['timestamp']
           - recs[i]['CAM_FRONT']['timestamp']) * 1e-6
          for i in range(len(kf) - 1)]
print('keyframe steps m  :', np.round(d_kf, 2))
print('keyframe dt ms    :', np.round(np.array(dts_kf) * 1000, 0))

# 2. non-key CAMERA record steps within one keyframe interval, and the
#    mismatch between each non-key pose and interpolation of keyframes
mid = ds.camera_window(recs[12]['CAM_FRONT'], 6, 6)
eg = [ds.ego_pose(s) for s in mid]
d_mid = [float(np.linalg.norm(eg[i + 1][:3, 3] - eg[i][:3, 3]))
         for i in range(len(eg) - 1)]
print('non-key cam steps m:', np.round(d_mid, 2),
      f'(sum over ~0.5 s = {sum(d_mid):.2f} m)')
print('  keyframe pair bracketing this window: '
      f'{np.linalg.norm(kf[13] - kf[12]):.2f} m over '
      f'{dts_kf[12] * 1000:.0f} ms')

# 3. lidar sweep steps bracketing the same interval
swp = ds.sweep_history(recs[12]['LIDAR_TOP'], 12)
se = [ds.ego_pose(s) for s in swp]
d_sw = [float(np.linalg.norm(se[i + 1][:3, 3] - se[i][:3, 3]))
        for i in range(len(se) - 1)]
print('lidar sweep steps m:', np.round(d_sw, 2))

# 4. same comparison on a LATER stretch of the log (real motion?)
recs_l = recs[25:32]
kf2 = [ds.ego_pose(r['CAM_FRONT'])[:3, 3] for r in recs_l]
print('later keyframe steps m:',
      np.round([float(np.linalg.norm(kf2[i + 1] - kf2[i]))
                for i in range(len(kf2) - 1)], 2))
mid2 = ds.camera_window(recs[28]['CAM_FRONT'], 6, 6)
eg2 = [ds.ego_pose(s) for s in mid2]
d2 = [float(np.linalg.norm(eg2[i + 1][:3, 3] - eg2[i][:3, 3]))
      for i in range(len(eg2) - 1)]
print('later non-key cam steps m:', np.round(d2, 2),
      f'(sum {sum(d2):.2f})')
swp2 = ds.sweep_history(recs[28]['LIDAR_TOP'], 12)
se2 = [ds.ego_pose(s) for s in swp2]
d_sw2 = [float(np.linalg.norm(se2[i + 1][:3, 3] - se2[i][:3, 3]))
         for i in range(len(se2) - 1)]
print('later lidar sweep steps m:', np.round(d_sw2, 2))
