# -*- coding: utf-8 -*-
"""BEV stitch: project 6 nuScenes camera images onto a common ground plane
using camera intrinsics/extrinsics, producing a stitched top-down view.

Method: lay a dense grid of ground points on the plane z = 0 of the ego
frame, transform each point into every camera (extrinsics) and project it
with the intrinsics, then sample pixel colors and blend them on the BEV
canvas (edge pixels are down-weighted)."""
import json
import os

import cv2
import numpy as np

ROOT = r'H:\datasets\nuscenes-mini'
META = os.path.join(ROOT, 'v1.0-mini')
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\outputs\bev'

# BEV grid (nuScenes ego frame): x = forward, y = left, ground plane z = 0
X_MIN, X_MAX = -25.0, 45.0     # forward (x)
Y_MIN, Y_MAX = -25.0, 25.0     # lateral (y, positive = left)
RES = 0.05                     # meters per pixel

CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def load_json(name):
    with open(os.path.join(META, name + '.json'), 'r') as f:
        return json.load(f)


def quat_to_mat(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def pick_sample():
    sample_data = load_json('sample_data')
    sensors = {s['token']: s['channel'] for s in load_json('sensor')}
    channel_of = {c['token']: sensors[c['sensor_token']]
                  for c in load_json('calibrated_sensor')}
    by_sample = {}
    for sd in sample_data:
        if sd['is_key_frame']:
            cam = channel_of[sd['calibrated_sensor_token']]
            if cam in CAMS:
                by_sample.setdefault(sd['sample_token'], {})[cam] = sd
    for tok, cams in by_sample.items():
        if len(cams) == len(CAMS):
            return tok, cams
    raise RuntimeError('no sample with all 6 cameras')


def main():
    tok, cams = pick_sample()
    print(f'sample token: {tok}')

    calibrated_sensor = {r['token']: r for r in load_json('calibrated_sensor')}

    W = int((Y_MAX - Y_MIN) / RES)   # columns  = lateral
    H = int((X_MAX - X_MIN) / RES)   # rows    = forward

    # ground grid in ego frame; row 0 = X_MAX (far ahead), col 0 = Y_MAX (left)
    xs = np.arange(X_MAX - RES / 2, X_MIN, -RES)
    ys = np.arange(Y_MAX - RES / 2, Y_MIN, -RES)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')  # (H, W) = (forward, lateral)
    pts_ego = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])

    color_sum = np.zeros((gx.size, 3), np.float64)
    weight_sum = np.zeros(gx.size, np.float64)
    cam_color = {c: np.zeros((gx.size, 3), np.float64) for c in CAMS}
    cam_weight = {c: np.zeros(gx.size, np.float64) for c in CAMS}

    for cam in CAMS:
        sd = cams[cam]
        img = cv2.imread(os.path.join(ROOT, sd['filename']))
        h, w = img.shape[:2]

        calib = calibrated_sensor[sd['calibrated_sensor_token']]
        R = quat_to_mat(calib['rotation'])
        t = np.array(calib['translation'])
        K = np.array(calib['camera_intrinsic'])

        # ego ground points -> camera frame
        pts_cam = R.T @ (pts_ego - t[:, None])
        valid_idx = np.nonzero(pts_cam[2] > 0.1)[0]
        uv = K @ pts_cam[:, valid_idx]
        uv = uv[:2] / uv[2]
        u = np.round(uv[0]).astype(int)
        v = np.round(uv[1]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        idx = valid_idx[ok]
        uu, vv = u[ok], v[ok]
        colors = img[vv, uu].astype(np.float64)[:, ::-1]  # BGR -> RGB

        # pixels near the image edge are unreliable; down-weight them
        dist = np.minimum.reduce([uu, w - 1 - uu, vv, h - 1 - vv]).astype(np.float64)
        wgt = np.clip(dist / 8.0, 0.0, 1.0)

        color_sum[idx] += colors * wgt[:, None]
        weight_sum[idx] += wgt
        cam_color[cam][idx] += colors * wgt[:, None]
        cam_weight[cam][idx] += wgt
        print(f'{cam:18s} covers {ok.sum() / pts_ego.shape[1]:6.1%} of grid')

    cov = weight_sum > 0
    bev = np.full((gx.size, 3), 255.0)
    bev[cov] = color_sum[cov] / weight_sum[cov, None]
    bev = bev.reshape(H, W, 3).astype(np.uint8)  # RGB, forward = up
    bev = np.ascontiguousarray(bev[:, :, ::-1])  # -> BGR for cv2

    # per-camera BEV: each camera alone on the shared canvas
    for cam in CAMS:
        per = np.full((gx.size, 3), 255.0)
        m = cam_weight[cam] > 0
        per[m] = cam_color[cam][m] / cam_weight[cam][m, None]
        per = per.reshape(H, W, 3).astype(np.uint8)
        per = np.ascontiguousarray(per[:, :, ::-1])
        p_path = os.path.join(OUT_DIR, f'bev_single_{cam}_{tok[:8]}.png')
        cv2.imwrite(p_path, per)
        print('saved:', p_path)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'bev_stitch_{tok[:8]}.png')
    cv2.imwrite(out_path, bev)
    print('saved:', out_path, bev.shape)

    # annotated copy: ego marker + 10 m rings
    ann = bev.copy()
    cx = int((Y_MAX - 0) / RES)
    cy = int((X_MAX - 0) / RES)
    for d in range(10, 71, 10):
        cv2.circle(ann, (cx, cy), int(d / RES), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(ann, (cx, cy), 6, (0, 0, 255), -1)
    cv2.putText(ann, 'ego', (cx + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)
    out2 = out_path.replace('.png', '_annotated.png')
    cv2.imwrite(out2, ann)
    print('saved:', out2)


if __name__ == '__main__':
    main()
