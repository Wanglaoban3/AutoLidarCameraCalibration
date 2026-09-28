# -*- coding: utf-8 -*-
"""Refine CAM_FRONT_LEFT extrinsic ROTATION by epipolar consistency.

Objective: minimize robust Sampson epipolar error of ALIKED+LightGlue
matches over N same-timestamp (front, front-left) frame pairs of one scene.
No ground-plane assumption; translation is trusted (rig mount), rotation is
the only unknown (3 params, bounded +-1.5 deg).

Validation (not optimization target): ground-lane BEV chamfer between
front @ t-1s and front-left @ t+1s around frame 18, before vs after."""
import os
import re
import sys

import cv2
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = r'H:\datasets\nuscenes-mini'
META = r'H:\datasets\nuscenes-mini\v1.0-mini'
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\outputs\bev\frames\frame18_lane_match'
NB = r'H:\projects\auto-extrinsics-adjusting\outputs\bev\frames\frame18_neighbor_bev'

RES = 0.05
X_MIN, X_MAX = -25.0, 45.0
Y_MIN, Y_MAX = -25.0, 25.0
H = int((X_MAX - X_MIN) / RES)
W = int((Y_MAX - Y_MIN) / RES)
MAX_FRAMES = 12
LOG_PREFIX = 'n008-2018-08-01-15-16-36-0400'   # frame18's log
BOUND_DEG = 1.5

sys.path.insert(0, r'H:\projects\LightGlue-main')
from lightglue import ALIKED, LightGlue  # noqa: E402
from lightglue.utils import numpy_image_to_torch  # noqa: E402

sys.path.insert(0, r'H:\projects\auto-extrinsics-adjusting')
from bev_prevnext import load_json, quat_to_mat, frame18_sample_data  # noqa
from bev_lane_match import T_of, extract_lane_pixels  # noqa


def main():
    calibrated = {r['token']: r for r in load_json('calibrated_sensor')}
    egos = {r['token']: r for r in load_json('ego_pose')}
    sensors = {s['token']: s['channel'] for s in load_json('sensor')}
    channel_of = {c['token']: sensors[c['sensor_token']]
                  for c in load_json('calibrated_sensor')}
    cams18, _ = frame18_sample_data()
    tok18 = cams18['CAM_FRONT']['sample_token']
    sd_all = load_json('sample_data')

    # --- pick N same-timestamp (front, front-left) pairs of one scene -----
    front_kf = [sd for sd in sd_all if sd['is_key_frame']
                and channel_of[sd['calibrated_sensor_token']] == 'CAM_FRONT'
                and LOG_PREFIX in sd['filename']]
    front_kf.sort(key=lambda s: s['timestamp'])
    step = max(1, len(front_kf) // MAX_FRAMES)
    frames = []
    for sd_f in front_kf[::step][:MAX_FRAMES]:
        sd_fl = next((sd for sd in sd_all
                      if sd['sample_token'] == sd_f['sample_token']
                      and sd['is_key_frame']
                      and channel_of[sd['calibrated_sensor_token']]
                      == 'CAM_FRONT_LEFT'), None)
        if sd_fl is not None:
            frames.append((sd_f, sd_fl))
    print(f'frame pairs: {len(frames)}')

    def setup(sd):
        calib = calibrated[sd['calibrated_sensor_token']]
        T_ec = np.eye(4)
        T_ec[:3, :3] = quat_to_mat(calib['rotation'])
        T_ec[:3, 3] = calib['translation']
        return dict(T_ec=T_ec, K=np.array(calib['camera_intrinsic']),
                    T_ge=T_of(egos[sd['ego_pose_token']]),
                    img=cv2.imread(os.path.join(ROOT, sd['filename'])))

    views = [(setup(a), setup(b)) for a, b in frames]

    # --- match every pair ---------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.0,
                       weights='official').eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)

    all_matches = []   # (view_idx, kp1 Nx2, kp2 Nx2)
    with torch.no_grad():
        for i, (A, B) in enumerate(views):
            f1 = extractor.extract(numpy_image_to_torch(A['img']).to(device))
            f2 = extractor.extract(numpy_image_to_torch(B['img']).to(device))
            m01 = matcher({'image0': f1, 'image1': f2})
            m = m01['matches'][0].cpu().numpy()
            kp1 = f1['keypoints'][0].cpu().numpy()[m[:, 0]]
            kp2 = f2['keypoints'][0].cpu().numpy()[m[:, 1]]
            all_matches.append((i, kp1, kp2))
            print(f'pair {i}: {len(kp1)} matches')

    # --- objective: robust sampson error in pixels ---------------------------
    # perturbation: front-left cam-frame rotvec; x2 = R21 x1 + t21 with
    # T_w2' = T_ge2 @ T_ec2 @ Exp(delta)
    def sampson_px(kp1, kp2, F):
        p1h = np.hstack([kp1, np.ones((len(kp1), 1))])
        p2h = np.hstack([kp2, np.ones((len(kp2), 1))])
        Fp1 = p1h @ F.T
        Ftp2 = p2h @ F
        num = np.sum(p2h * Fp1, axis=1) ** 2
        den = Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 + Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2
        return np.sqrt(num / np.maximum(den, 1e-12))

    def frame_F(A, B, delta):
        dR = Rotation.from_rotvec(delta).as_matrix()
        T_pert = np.eye(4)
        T_pert[:3, :3] = dR
        T_w1 = A['T_ge'] @ A['T_ec']
        T_w2 = B['T_ge'] @ B['T_ec'] @ T_pert
        T_21 = np.linalg.inv(T_w2) @ T_w1
        R_rel, t_rel = T_21[:3, :3], T_21[:3, 3]
        E = np.array([[0, -t_rel[2], t_rel[1]],
                      [t_rel[2], 0, -t_rel[0]],
                      [-t_rel[1], t_rel[0], 0]]) @ R_rel
        return np.linalg.inv(B['K']).T @ E @ np.linalg.inv(A['K'])

    def huber(d, delta=3.0):
        a = np.abs(d)
        return np.where(a <= delta, 0.5 * a ** 2,
                        delta * (a - 0.5 * delta))

    def total_loss(x):
        d = Rotation.from_rotvec(np.asarray(x)).as_matrix()
        loss = 0.0
        for (i, kp1, kp2) in all_matches:
            A, B = views[i]
            F = frame_F(A, B, np.asarray(x))
            loss += huber(sampson_px(kp1, kp2, F)).mean()
        return loss / len(all_matches)

    e0 = [sampson_px(kp1, kp2, frame_F(views[i][0], views[i][1], np.zeros(3)))
          for (i, kp1, kp2) in all_matches]
    print(f'BEFORE: mean-frame sampson medians '
          f'{np.median(np.concatenate(e0)):.2f} px '
          f'(frame means: {[round(float(np.mean(e))) for e in e0]})')

    # --- optimize (Nelder-Mead, bounded start box) ---------------------------
    x0 = np.zeros(3)
    res = minimize(total_loss, x0, method='Nelder-Mead',
                   options=dict(initial_simplex=np.vstack([
                       x0,
                       x0 + np.eye(3) * np.radians(0.25)]),
                       xatol=1e-5, fatol=1e-5, maxiter=600))
    delta = np.clip(res.x, np.radians(-BOUND_DEG), np.radians(BOUND_DEG))
    print(f'delta-R (front-left cam frame, deg): '
          f'rx={np.degrees(res.x[0]):+.3f} ry={np.degrees(res.x[1]):+.3f} '
          f'rz={np.degrees(res.x[2]):+.3f}  '
          f'(|d|={np.degrees(Rotation.from_rotvec(res.x).magnitude()):.3f}, '
          f'clamped to +-{"%g" % BOUND_DEG} if needed)')
    e1 = [sampson_px(kp1, kp2, frame_F(views[i][0], views[i][1], res.x))
          for (i, kp1, kp2) in all_matches]
    print(f'AFTER:  sampson median {np.median(np.concatenate(e1)):.2f} px '
          f'(frame means: {[round(float(np.mean(e))) for e in e1]})')

    # --- save optimized extrinsic -------------------------------------------
    _, B0 = views[0]
    T_ec_opt = B0['T_ec'].copy()
    T_ec_opt[:3, :3] = T_ec_opt[:3, :3] @ Rotation.from_rotvec(res.x).as_matrix()
    np.save(os.path.join(OUT_DIR, 'frontleft_T_ec_epipolar.npy'), T_ec_opt)
    np.save(os.path.join(OUT_DIR, 'frontleft_delta_rotvec_epipolar.npy'), res.x)
    print('optimized front-left R_ec:\n', np.round(T_ec_opt[:3, :3], 5))

    # --- validation: BEV lane chamfer around frame18 (not optimized on) -----
    import bev_prevnext as bp
    from ultralytics import YOLO
    bp.yolo = YOLO(r'H:\models\yolo26x-sem.pt')

    sd_f = walk_chain_local(sd_all, sd_by_token_local(), cams18, 'CAM_FRONT', 'prev')
    sd_fl18 = next(sd for sd in sd_all if sd['sample_token'] == tok18
                   and sd['is_key_frame']
                   and channel_of[sd['calibrated_sensor_token']] == 'CAM_FRONT_LEFT')
    sd_fl = walk_chain_local(sd_all, sd_by_token_local(), cams18, 'CAM_FRONT_LEFT',
                             'next', base=sd_fl18)
    T_ref_inv = np.linalg.inv(T_of(egos[cams18['CAM_FRONT']['ego_pose_token']]))

    def lane_ref_pts(sd, road):
        img = cv2.imread(os.path.join(ROOT, sd['filename']))
        lane = extract_lane_pixels(img, road)
        vv, uu = np.nonzero(lane)
        calib = calibrated[sd['calibrated_sensor_token']]
        T_ec = np.eye(4)
        T_ec[:3, :3] = quat_to_mat(calib['rotation'])
        T_ec[:3, 3] = calib['translation']
        K = np.array(calib['camera_intrinsic'])
        T_gc = T_of(egos[sd['ego_pose_token']]) @ T_ec
        o = T_gc[:3, 3]
        d = T_gc[:3, :3] @ (np.linalg.inv(K) @
                            np.vstack([uu, vv, np.ones_like(uu)]).astype(float))
        s = -o[2] / d[2]
        return o[:, None] + d * s

    def to_bev_canvas(p_ref):
        u = np.round((Y_MAX - p_ref[1]) / RES).astype(int)
        v = np.round((X_MAX - p_ref[0]) / RES).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        c = np.zeros((H, W), bool)
        c[v[ok], u[ok]] = True
        return c

    road_f = cv2.imread(os.path.join(NB, 'minus1s_front_mask_CAM_FRONT.png'),
                        0) > 127
    p_ref = (T_ref_inv @ np.vstack(
        [lane_ref_pts(sd_f, road_f), np.ones(
            lane_ref_pts(sd_f, road_f).shape[1])]))[:3]
    ref = to_bev_canvas(p_ref)
    D = distance_transform(ref)

    road_fl = bp.segment_ground(cv2.imread(
        os.path.join(ROOT, sd_fl['filename'])))
    g2 = lane_ref_pts(sd_fl, road_fl)
    calib_fl = calibrated[sd_fl['calibrated_sensor_token']]
    K_fl = np.array(calib_fl['camera_intrinsic'])
    T_ge_fl = T_of(egos[sd_fl['ego_pose_token']])
    vv0, uu0 = None, None

    def chamfer(delta_vec):
        T_ec = np.eye(4)
        T_ec[:3, :3] = quat_to_mat(calib_fl['rotation']) @ \
            Rotation.from_rotvec(delta_vec).as_matrix()
        T_ec[:3, 3] = calib_fl['translation']
        T_gc = T_ge_fl @ T_ec
        o = T_gc[:3, 3]
        # front-left lane pixels
        img = cv2.imread(os.path.join(ROOT, sd_fl['filename']))
        lane = extract_lane_pixels(img, road_fl)
        vv, uu = np.nonzero(lane)
        d = T_gc[:3, :3] @ (np.linalg.inv(K_fl) @
                            np.vstack([uu, vv, np.ones_like(uu)]).astype(float))
        s = -o[2] / d[2]
        pg = o[:, None] + d * s
        p = (T_ref_inv @ np.vstack([pg, np.ones(pg.shape[1])]))[:3]
        c = to_bev_canvas(p)
        if c.sum() == 0:
            return np.nan, c
        return D[c].mean(), c

    d0, c0 = chamfer(np.zeros(3))
    d1, c1 = chamfer(res.x)
    print(f'BEV lane chamfer around frame18: BEFORE {d0 * RES * 100:.1f} cm -> '
          f'AFTER {d1 * RES * 100:.1f} cm')

    bev = np.full((H, W, 3), 255, np.uint8)
    bev[ref] = (200, 200, 255)
    bev[c1 & ~ref] = (0, 160, 0)
    bev[c0 & ~ref & ~c1] = (150, 150, 150)
    cv2.circle(bev, (W // 2, int((X_MAX - 0) / RES)), 6, (255, 0, 0), -1)
    out = os.path.join(OUT_DIR, 'bev_frontleft_epipolar_opt.png')
    cv2.imwrite(out, bev)
    print('saved:', out)


def walk_chain_local(sd_all, sd_by_token, cams18, cam, direction, base=None):
    from bev_prevnext import walk_chain
    sd = base if base is not None else sd_by_token[cams18[cam]['token']]
    return walk_chain(sd, direction, 1.0, sd_by_token)


def sd_by_token_local():
    return {sd['token']: sd for sd in load_json('sample_data')}


def distance_transform(ref):
    from scipy.ndimage import distance_transform_edt
    import cv2 as _cv2
    D = distance_transform_edt(~ref)
    return _cv2.GaussianBlur(D, (0, 0), 1.5)


if __name__ == '__main__':
    main()
