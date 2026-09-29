# -*- coding: utf-8 -*-
"""Debug 9: pose-free relative-pose check. SIFT+MAGSAC recoverPose vs
the oracle ego chain, for key-key and key-nonkey pairs."""
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
cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
K = cal_c['K']
Kinv = np.linalg.inv(K)

sift = cv2.SIFT_create(nfeatures=6000)


def desc(sd):
    img = ds.load_image(sd)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(g, None)


def check(sd_a, sd_b, fa, fb):
    kp_a, de_a = fa
    kp_b, de_b = fb
    bf = cv2.BFMatcher()
    m2 = bf.knnMatch(de_a, de_b, k=2)
    good = [m[0] for m in m2 if len(m) == 2 and m[0].distance
            < 0.8 * m[1].distance]
    if len(good) < 50:
        print(f'  only {len(good)} ratio matches')
        return
    uva = np.float32([kp_a[m.queryIdx].pt for m in good])
    uvb = np.float32([kp_b[m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(uva, uvb, cv2.USAC_MAGSAC, 2.0,
                                     0.999, 10000)
    inl = int(mask.sum()) if mask is not None else 0
    E = K.T @ F
    uva_i, uvb_i = uva[mask.ravel() == 1], uvb[mask.ravel() == 1]
    out = cv2.recoverPose(E, uva_i, uvb_i, K)
    R_f, t_f = out[1], out[2]
    # oracle
    Ta = ds.ego_pose(sd_a) @ np.eye(4)
    Tb = ds.ego_pose(sd_b) @ np.eye(4)
    T_ec = np.eye(4)
    T_ec[:3, :3], T_ec[:3, 3] = cal_c['R_cs'], cal_c['t_cs']
    Wa = Ta @ T_ec
    Wb = Tb @ T_ec
    T_b_a = np.linalg.inv(Wa) @ Wb
    R_o, t_o = T_b_a[:3, :3], T_b_a[:3, 3]
    dang = np.degrees(np.arccos(np.clip(
        (np.trace(R_f @ R_o.T) - 1) / 2, -1, 1)))
    if np.linalg.norm(t_f) > 1e-9 and np.linalg.norm(t_o) > 1e-9:
        tdir = np.degrees(np.arccos(np.clip(
            float(t_f.ravel() @ t_o.ravel()
                  / (np.linalg.norm(t_f) * np.linalg.norm(t_o))), -1,
            1)))
    else:
        tdir = float('nan')
    # point-to-epipolar px under ORACLE E for the inliers
    x1 = (Kinv @ np.c_[uva_i, np.ones(len(uva_i))].T).T[:, :3]
    x2 = (Kinv @ np.c_[uvb_i, np.ones(len(uvb_i))].T).T[:, :3]
    E1 = x1 @ E_o if False else x1 @ (np.cross(t_o, np.eye(3)) @ R_o).T
    E2 = x2 @ (np.cross(t_o, np.eye(3)) @ R_o)
    num = (E1 * x2).sum(1)
    den = (E1 ** 2).sum(1) + (E2 ** 2).sum(1)
    d_px = np.abs(num) / np.sqrt(np.maximum(den, 1e-12)) \
        * np.linalg.norm(K[0, 0:1] * [1, 1]) / np.sqrt(2)
    print(f'  matches {len(good)}, F-inliers {inl}; '
          f'recovered-vs-oracle dR {dang:6.2f} deg, t-dir {tdir:6.2f} '
          f'deg; oracle-epipolar px med {np.median(d_px):6.2f}')


base = 12
pairs = [
    (recs[base]['CAM_FRONT'], recs[base + 1]['CAM_FRONT'],
     'key -> key (+0.5s)'),
    (recs[base]['CAM_FRONT'], recs[base + 2]['CAM_FRONT'],
     'key -> key (+1.0s)'),
    (recs[base]['CAM_FRONT'], recs[base + 4]['CAM_FRONT'],
     'key -> key (+2.0s)'),
]
cache = {}
for a, b, tag in pairs:
    print(f'{tag}:')
    for sd in (a, b):
        if sd['token'] not in cache:
            cache[sd['token']] = desc(sd)
    check(a, b, cache[a['token']], cache[b['token']])
