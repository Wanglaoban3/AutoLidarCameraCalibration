# -*- coding: utf-8 -*-
"""Visualize where the ALIKED+LightGlue matched points are on the ORIGINAL
images: inliers (passed the known-pose epipolar check) in green, rejected
matches in red. Also report the funnel statistics."""
import os
import sys

import cv2
import numpy as np
import torch

ROOT = r'H:\datasets\nuscenes-mini'
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\outputs\bev\frames\frame18_lane_match'

sys.path.insert(0, r'H:\projects\LightGlue-main')
from lightglue import ALIKED, LightGlue  # noqa: E402
from lightglue.utils import numpy_image_to_torch  # noqa: E402

sys.path.insert(0, r'H:\projects\auto-extrinsics-adjusting')
from bev_prevnext import frame18_sample_data, load_json, quat_to_mat, walk_chain  # noqa
from bev_lane_match import T_of  # noqa

EPIPOLAR_PX = 5.0
RES = 0.05
X_MIN, X_MAX = -25.0, 45.0
Y_MIN, Y_MAX = -25.0, 25.0


def in_bev(p_ref):
    return (X_MIN <= p_ref[0] <= X_MAX) and (Y_MIN <= p_ref[1] <= Y_MAX)


def main():
    calibrated = {r['token']: r for r in load_json('calibrated_sensor')}
    egos = {r['token']: r for r in load_json('ego_pose')}
    cams18, sd_by_token = frame18_sample_data()
    T_ref_inv = np.linalg.inv(T_of(egos[cams18['CAM_FRONT']['ego_pose_token']]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.0,
                       weights='official').eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)

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
        views[tag] = dict(sd=sd, img=img, T_ec=T_ec,
                          T_ge=T_of(egos[sd['ego_pose_token']]),
                          K=np.array(calib['camera_intrinsic']))

    f0 = extractor.extract(numpy_image_to_torch(views['minus1s_front']['img']).to(device))
    f1 = extractor.extract(numpy_image_to_torch(views['plus1s_front_right']['img']).to(device))
    print('keypoints:', f0['keypoints'].shape[1], '/', f1['keypoints'].shape[1])
    with torch.no_grad():
        m01 = matcher({'image0': f0, 'image1': f1})
    m = m01['matches'][0].cpu().numpy()
    kp1 = f0['keypoints'][0].cpu().numpy()
    kp2 = f1['keypoints'][0].cpu().numpy()
    conf = m01['matching_scores0'][0].cpu().numpy()
    print(f'matches: {len(m)}, conf median {np.median(conf):.2f}')

    A, B = views['minus1s_front'], views['plus1s_front_right']
    T_w1 = A['T_ge'] @ A['T_ec']
    T_w2 = B['T_ge'] @ B['T_ec']
    T_21 = np.linalg.inv(T_w2) @ T_w1
    R_rel, t_rel = T_21[:3, :3], T_21[:3, 3]
    E = np.array([[0, -t_rel[2], t_rel[1]],
                  [t_rel[2], 0, -t_rel[0]],
                  [-t_rel[1], t_rel[0], 0]]) @ R_rel

    def epi_dist(p1, p2):
        x1 = np.linalg.inv(A['K']) @ np.array([p1[0], p1[1], 1.0])
        l_px = np.linalg.inv(B['K']).T @ (E @ x1)
        n = np.hypot(l_px[0], l_px[1])
        return abs(l_px @ np.array([p2[0], p2[1], 1.0])) / n if n > 1e-9 else 1e9

    stats = {'inlier': 0, 'epi_rej': 0, 'outside': 0}
    for i in range(len(m)):
        p1, p2 = kp1[m[i, 0]], kp2[m[i, 1]]
        if epi_dist(p1, p2) < EPIPOLAR_PX:
            stats['inlier'] += 1
            # check BEV reachability of both ends
            ok_bev = True
            for vw, p in [(A, p1), (B, p2)]:
                K, R, t = vw['K'], vw['T_ec'][:3, :3], vw['T_ec'][:3, 3]
                rays = np.linalg.inv(K) @ np.array([p[0], p[1], 1.0])
                Rd = R @ rays
                if Rd[2] <= 0 or t[2] / Rd[2] <= 0:
                    ok_bev = False
                    continue
                s = -t[2] / Rd[2]
                p_ego = (R @ (rays.reshape(3) * s) + t).reshape(3)
                p_glob = vw['T_ge'] @ np.concatenate([p_ego, [1.0]])
                pr = (T_ref_inv @ p_glob)[:3]
                if not in_bev(pr):
                    ok_bev = False
            if ok_bev:
                stats.setdefault('plotted', 0)
                stats['plotted'] += 1
            else:
                stats['outside'] += 1
        else:
            stats['epi_rej'] += 1
    print('funnel:', stats)

    # draw
    def thicken(img, p, color, r=5):
        cv2.circle(img, (int(round(p[0])), int(round(p[1]))), r, color, -1)
        cv2.circle(img, (int(round(p[0])), int(round(p[1]))), r + 1,
                   (0, 0, 0), 1)

    im1 = views['minus1s_front']['img'].copy()
    im2 = views['plus1s_front_right']['img'].copy()
    cv2.putText(im1, 'front @ t-1s  (green=inlier red=rejected)',
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    cv2.putText(im2, 'front-right @ t+1s', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    for i in range(len(m)):
        p1, p2 = kp1[m[i, 0]], kp2[m[i, 1]]
        ok = epi_dist(p1, p2) < EPIPOLAR_PX
        c = (0, 200, 0) if ok else (0, 0, 255)
        thicken(im1, p1, c, 4 if ok else 3)
        thicken(im2, p2, c, 4 if ok else 3)

    panel = np.hstack([cv2.resize(im1, (1200, 675)),
                       cv2.resize(im2, (1200, 675))])
    out = os.path.join(OUT_DIR, 'aliked_matches_on_images.png')
    cv2.imwrite(out, panel)
    print('saved:', out)


if __name__ == '__main__':
    main()
