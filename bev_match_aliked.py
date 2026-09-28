# -*- coding: utf-8 -*-
"""ALIKED + LightGlue matching between the two temporal views of frame 18
(CAM_FRONT @ t-1s vs CAM_FRONT_RIGHT @ t+1s), then known-pose epipolar
filtering and ground back-projection into the common BEV."""
import os
import sys

import cv2
import numpy as np
import torch

ROOT = r'H:\datasets\nuscenes-mini'
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\outputs\bev\frames\frame18_lane_match'

RES = 0.05
X_MIN, X_MAX = -25.0, 45.0
Y_MIN, Y_MAX = -25.0, 25.0
H = int((X_MAX - X_MIN) / RES)
W = int((Y_MAX - Y_MIN) / RES)
EPIPOLAR_PX = 5.0

sys.path.insert(0, r'H:\projects\LightGlue-main')
from lightglue import ALIKED, LightGlue  # noqa: E402
from lightglue.utils import numpy_image_to_torch  # noqa: E402

sys.path.insert(0, r'H:\projects\auto-extrinsics-adjusting')
from bev_prevnext import frame18_sample_data, load_json, quat_to_mat, walk_chain  # noqa
from bev_lane_match import T_of  # noqa


def ground_pts_global_plane(u_px, v_px, K, T_ec, T_ge):
    T_gc = T_ge @ T_ec
    o = T_gc[:3, 3]
    d = T_gc[:3, :3] @ (np.linalg.inv(K) @
                        np.vstack([u_px, v_px, np.ones_like(u_px)]).astype(float))
    s = -o[2] / d[2]
    return o[:, None] + d * s


def main():
    calibrated = {r['token']: r for r in load_json('calibrated_sensor')}
    egos = {r['token']: r for r in load_json('ego_pose')}
    cams18, sd_by_token = frame18_sample_data()
    T_ref = T_of(egos[cams18['CAM_FRONT']['ego_pose_token']])
    T_ref_inv = np.linalg.inv(T_ref)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.0,
                       weights='official').eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)
    print('ALIKED + LightGlue loaded on', device)

    plan = [('minus1s_front', 'CAM_FRONT', 'prev'),
            ('plus1s_front_right', 'CAM_FRONT_RIGHT', 'next')]
    views = {}
    for tag, cam, key in plan:
        sd = walk_chain(sd_by_token[cams18[cam]['token']], key, 1.0, sd_by_token)
        calib = calibrated[sd['calibrated_sensor_token']]
        T_ec = np.eye(4)
        T_ec[:3, :3] = quat_to_mat(calib['rotation'])
        T_ec[:3, 3] = calib['translation']
        img = cv2.imread(os.path.join(ROOT, sd['filename']))
        road = cv2.imread(os.path.join(
            OUT_DIR, '..', 'frame18_neighbor_bev', f'{tag}_mask_{cam}.png'),
            0) > 127
        feats = extractor.extract(numpy_image_to_torch(img).to(device))
        kps = feats['keypoints'][0].cpu().numpy()
        views[tag] = dict(sd=sd, img=img, road=road, kps=kps, T_ec=T_ec,
                          T_ge=T_of(egos[sd['ego_pose_token']]),
                          K=np.array(calib['camera_intrinsic']))
        print(f'{tag}: {len(kps)} ALIKED keypoints, '
              f'{int(road[kps[:, 1].astype(int), kps[:, 0].astype(int)].sum())} '
              f'on ground')

    f0 = extractor.extract(numpy_image_to_torch(views['minus1s_front']['img']).to(device))
    f1 = extractor.extract(numpy_image_to_torch(views['plus1s_front_right']['img']).to(device))
    with torch.no_grad():
        m01 = matcher({'image0': f0, 'image1': f1})
    m = m01['matches'][0].cpu().numpy()
    kp1 = views['minus1s_front']['kps'][m[:, 0]]
    kp2 = views['plus1s_front_right']['kps'][m[:, 1]]
    conf = m01['matching_scores0'][0].cpu().numpy()
    print(f'LightGlue matches: {len(m)} (median conf {np.median(conf):.2f})')

    # known-pose epipolar filter: relative pose cam1 -> cam2 through world
    A, B = views['minus1s_front'], views['plus1s_front_right']
    T_w1 = A['T_ge'] @ A['T_ec']
    T_w2 = B['T_ge'] @ B['T_ec']
    T_21 = np.linalg.inv(T_w2) @ T_w1
    R_rel, t_rel = T_21[:3, :3], T_21[:3, 3]
    E = np.array([[0, -t_rel[2], t_rel[1]],
                  [t_rel[2], 0, -t_rel[0]],
                  [-t_rel[1], t_rel[0], 0]]) @ R_rel

    def epi_dist(p1, p2, K1, K2):
        x1 = np.linalg.inv(K1) @ np.array([p1[0], p1[1], 1.0])
        l_px = np.linalg.inv(K2).T @ (E @ x1)
        n = np.hypot(l_px[0], l_px[1])
        return abs(l_px @ np.array([p2[0], p2[1], 1.0])) / n if n > 1e-9 else 1e9

    inl = [i for i in range(len(kp1))
           if epi_dist(kp1[i], kp2[i], A['K'], B['K']) < EPIPOLAR_PX]
    print(f'epipolar inliers: {len(inl)}')

    bev = np.full((H, W, 3), 255, np.uint8)
    colors = [(0, 0, 255), (0, 160, 0)]
    for ci, tag in enumerate(['minus1s_front', 'plus1s_front_right']):
        vw = views[tag]
        bev_tmp = np.zeros((H, W), np.uint8)
        # paint all edge points faintly? keep only matched keypoints here
        vw['bev_uv'] = []
    for i in inl:
        for ci, tag in enumerate(['minus1s_front', 'plus1s_front_right']):
            vw = views[tag]
            p2 = kp2[i] if tag != 'minus1s_front' else kp1[i]
            K, R = vw['K'], vw['T_ec'][:3, :3]
            t = vw['T_ec'][:3, 3]
            rays = np.linalg.inv(K) @ np.array([p2[0], p2[1], 1.0])
            Rd = R @ rays
            s = -t[2] / Rd[2]
            p_ego = R @ (rays * s) + t[:, None]
            p_glob = vw['T_ge'] @ np.concatenate([p_ego.ravel()[:3], [1.0]])
            p_ref = (T_ref_inv @ p_glob)[:3]
            u = int(round((Y_MAX - p_ref[1]) / RES))
            v = int(round((X_MAX - p_ref[0]) / RES))
            vw['bev_uv'].append((u, v))

    uv1 = views['minus1s_front']['bev_uv']
    uv2 = views['plus1s_front_right']['bev_uv']
    for (u1, v1), (u2, v2) in zip(uv1, uv2):
        if 0 <= u1 < W and 0 <= v1 < H and 0 <= u2 < W and 0 <= v2 < H:
            cv2.line(bev, (u1, v1), (u2, v2), (0, 220, 255), 1, cv2.LINE_AA)
    for (u, v) in uv1:
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(bev, (u, v), 4, (0, 0, 255), -1)
    for (u, v) in uv2:
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(bev, (u, v), 4, (0, 160, 0), -1)

    legend = [('matched pt front t-1s', (0, 0, 255)),
              ('matched pt fr t+1s', (0, 160, 0)),
              ('pair connector', (0, 220, 255))]
    for i, (txt, c) in enumerate(legend):
        y = 30 + i * 30
        cv2.rectangle(bev, (10, y - 12), (34, y + 8), c, -1)
        cv2.putText(bev, txt, (44, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
    cv2.circle(bev, (W // 2, int((X_MAX - 0) / RES)), 6, (255, 0, 0), -1)

    out = os.path.join(OUT_DIR, 'bev_matched_aliked_lightglue.png')
    cv2.imwrite(out, bev)
    print('saved:', out)


if __name__ == '__main__':
    main()
