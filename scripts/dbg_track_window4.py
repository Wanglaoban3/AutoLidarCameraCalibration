# -*- coding: utf-8 -*-
"""Debug 4: pose steps vs image flow in the SAME window."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from auto_extrinsics.data.nuscenes_lite import NuScenesLite
from auto_extrinsics.fine import geometry as geo

ds = NuScenesLite()
LOG = 'n008-2018-08-01-15-16-36-0400'
recs = ds.frames_of_log_multi(LOG, channels=('CAM_FRONT',))
centers = recs[::3][:10]
cal_c = ds.calib(centers[0]['CAM_FRONT'], 'CAM_FRONT')

for ci in (5, 3):
    sds = ds.camera_window(centers[ci]['CAM_FRONT'], 6, 6)
    eg = [ds.ego_pose(s) for s in sds]
    steps = [float(np.linalg.norm(eg[k + 1][:3, 3] - eg[k][:3, 3]))
             for k in range(len(eg) - 1)]
    yaws = []
    for k in range(len(eg) - 1):
        dR = eg[k + 1][:3, :3] @ eg[k][:3, :3].T
        yaws.append(np.degrees(
            Rotation.from_matrix(dR).as_rotvec()[2]))
    imgs = [ds.load_image(s) for s in sds]
    g = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    pc = []
    for k in (6, 7, 8):
        (dx, dy), resp = cv2.phaseCorrelate(
            np.float32(g[k]), np.float32(g[k + 1]))
        pc.append((round(dx, 2), round(dy, 2), round(float(resp), 3)))
    print(f'center {ci}: pose steps m '
          f'{np.array2string(np.array(steps), precision=2)}')
    print(f'  yaw steps deg '
          f'{np.array2string(np.array(yaws), precision=3)}')
    print(f'  phaseCorrelate f6->7,7->8,8->9 (dx,dy,resp): {pc}')
