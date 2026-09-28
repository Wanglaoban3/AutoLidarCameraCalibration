# -*- coding: utf-8 -*-
"""Visualization for the epipolar extrinsic refinement:
1. per-frame median Sampson error, nominal vs optimized (bar chart)
2. for the densest frame pair: matches drawn on BOTH original images,
   colored by residual, before vs after side by side."""
import os
import sys

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = r'H:\datasets\nuscenes-mini'
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\outputs\bev\frames\frame18_lane_match'
MAX_FRAMES = 12
LOG_PREFIX = 'n008-2018-08-01-15-16-36-0400'

sys.path.insert(0, r'H:\projects\LightGlue-main')
from lightglue import ALIKED, LightGlue  # noqa: E402
from lightglue.utils import numpy_image_to_torch  # noqa: E402

sys.path.insert(0, r'H:\projects\auto-extrinsics-adjusting')
from bev_prevnext import load_json, quat_to_mat  # noqa: E402
from bev_lane_match import T_of  # noqa


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


def err_color(d):
    if d < 1:
        return (0, 200, 0)       # green
    if d < 3:
        return (0, 200, 255)     # yellow-orange
    return (0, 0, 255)           # red


def main():
    calibrated = {r['token']: r for r in load_json('calibrated_sensor')}
    egos = {r['token']: r for r in load_json('ego_pose')}
    sensors = {s['token']: s['channel'] for s in load_json('sensor')}
    channel_of = {c['token']: sensors[c['sensor_token']]
                  for c in load_json('calibrated_sensor')}
    sd_all = load_json('sample_data')

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
        if sd_fl:
            frames.append((sd_f, sd_fl))

    def setup(sd):
        calib = calibrated[sd['calibrated_sensor_token']]
        T_ec = np.eye(4)
        T_ec[:3, :3] = quat_to_mat(calib['rotation'])
        T_ec[:3, 3] = calib['translation']
        return dict(T_ec=T_ec, K=np.array(calib['camera_intrinsic']),
                    T_ge=T_of(egos[sd['ego_pose_token']]),
                    img=cv2.imread(os.path.join(ROOT, sd['filename'])))

    views = [(setup(a), setup(b)) for a, b in frames]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.0,
                       weights='official').eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)

    all_matches = []
    with torch.no_grad():
        for i, (A, B) in enumerate(views):
            f1 = extractor.extract(numpy_image_to_torch(A['img']).to(device))
            f2 = extractor.extract(numpy_image_to_torch(B['img']).to(device))
            m01 = matcher({'image0': f1, 'image1': f2})
            m = m01['matches'][0].cpu().numpy()
            kp1 = f1['keypoints'][0].cpu().numpy()[m[:, 0]]
            kp2 = f2['keypoints'][0].cpu().numpy()[m[:, 1]]
            all_matches.append((i, kp1, kp2))

    delta = np.load(os.path.join(OUT_DIR, 'frontleft_delta_rotvec_epipolar.npy'))

    med_before, med_after = [], []
    for (i, kp1, kp2) in all_matches:
        A, B = views[i]
        med_before.append(np.median(sampson_px(kp1, kp2,
                                               frame_F(A, B, np.zeros(3)))))
        med_after.append(np.median(sampson_px(kp1, kp2,
                                              frame_F(A, B, delta))))

    # --- 1) bar chart --------------------------------------------------------
    n = len(all_matches)
    chart_w, chart_h = 1200, 520
    chart = np.full((chart_h, chart_w, 3), 255, np.uint8)
    cv2.putText(chart, 'median Sampson error per frame (px): blue = nominal, '
                       'green = optimized', (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    bw = chart_w // (2 * n + 1)
    vmax = max(max(med_before), max(med_after)) * 1.15
    for i in range(n):
        x0 = 20 + i * 2 * bw
        for j, (val, col) in enumerate([(med_before[i], (255, 120, 0)),
                                        (med_after[i], (0, 170, 0))]):
            hgt = int(val / vmax * (chart_h - 130))
            cv2.rectangle(chart, (x0 + j * bw, chart_h - 60 - hgt),
                          (x0 + (j + 1) * bw - 3, chart_h - 60), col, -1)
            cv2.putText(chart, f'{val:.1f}', (x0 + j * bw, chart_h - 70 - hgt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(chart, str(i), (x0 + bw - 8, chart_h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(chart, f'delta-R: '
                f'rx={np.degrees(delta[0]):+.2f} ry={np.degrees(delta[1]):+.2f} '
                f'rz={np.degrees(delta[2]):+.2f} deg',
                (20, chart_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, 'epi_residual_per_frame.png'), chart)

    # --- 2) matches on original images for the densest pair ------------------
    i_best = int(np.argmax([len(k) for _, k, _ in all_matches]))
    i, kp1, kp2 = all_matches[i_best]
    A, B = views[i]
    d_before = sampson_px(kp1, kp2, frame_F(A, B, np.zeros(3)))
    d_after = sampson_px(kp1, kp2, frame_F(A, B, delta))
    sel = np.linspace(0, len(kp1) - 1, 70).astype(int)

    def draw_panel(errs, title):
        im1 = A['img'].copy()
        im2 = B['img'].copy()
        for s in sel:
            c = err_color(errs[s])
            p1 = tuple(np.round(kp1[s]).astype(int))
            p2 = tuple(np.round(kp2[s]).astype(int))
            cv2.circle(im1, p1, 5, c, -1)
            cv2.circle(im1, p1, 6, (0, 0, 0), 1)
            cv2.circle(im2, p2, 5, c, -1)
            cv2.circle(im2, p2, 6, (0, 0, 0), 1)
        cv2.rectangle(im1, (0, 0), (900, 50), (30, 30, 30), -1)
        cv2.putText(im1, f'front  {title}', (15, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.rectangle(im2, (0, 0), (900, 50), (30, 30, 30), -1)
        cv2.putText(im2, f'front-left  {title}', (15, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return np.hstack([cv2.resize(im1, (1200, 675)),
                          cv2.resize(im2, (1200, 675))])

    panel_b = draw_panel(d_before, 'BEFORE (nominal extrinsic)')
    panel_a = draw_panel(d_after, 'AFTER (delta-R applied)')
    gap = np.full((12, panel_b.shape[1], 3), 30, np.uint8)
    out2 = np.vstack([panel_b, gap, panel_a])
    cv2.imwrite(os.path.join(OUT_DIR, 'epi_matches_before_after.png'), out2)
    print(f'pair {i_best} with {len(kp1)} matches; saved visualizations')


if __name__ == '__main__':
    main()
