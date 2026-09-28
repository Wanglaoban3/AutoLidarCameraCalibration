# -*- coding: utf-8 -*-
"""Randomly sample 20 nuScenes key frames and visualize each as a combined
figure: 6 raw camera views + the stitched BEV ground projection."""
import json
import os
import random

import cv2
import numpy as np

ROOT = r'H:\datasets\nuscenes-mini'
META = os.path.join(ROOT, 'v1.0-mini')
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\outputs\bev\frames'

N_FRAMES = 20
RES = 0.05                     # meters per pixel
X_MIN, X_MAX = -25.0, 45.0     # ego x = forward
Y_MIN, Y_MAX = -25.0, 25.0     # ego y = left
CAM_THUMB_W = 640              # display width of each camera thumbnail
EDGE_W = 8.0                   # edge-fade width (pixels) for BEV blending
SEED = 0

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


def pick_random_samples(n):
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
    full = [cams for cams in by_sample.values() if len(cams) == len(CAMS)]
    rng = random.Random(SEED)
    return rng.sample(full, n)


def render_bev(cams, calibrated_sensor):
    W = int((Y_MAX - Y_MIN) / RES)
    H = int((X_MAX - X_MIN) / RES)
    xs = np.arange(X_MAX - RES / 2, X_MIN, -RES)
    ys = np.arange(Y_MAX - RES / 2, Y_MIN, -RES)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')  # (H, W)
    pts_ego = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])

    color_sum = np.zeros((gx.size, 3), np.float64)
    weight_sum = np.zeros(gx.size, np.float64)

    for cam in CAMS:
        sd = cams[cam]
        img = cv2.imread(os.path.join(ROOT, sd['filename']))
        h, w = img.shape[:2]
        calib = calibrated_sensor[sd['calibrated_sensor_token']]
        R = quat_to_mat(calib['rotation'])
        t = np.array(calib['translation'])
        K = np.array(calib['camera_intrinsic'])

        pts_cam = R.T @ (pts_ego - t[:, None])
        valid_idx = np.nonzero(pts_cam[2] > 0.1)[0]
        uv = K @ pts_cam[:, valid_idx]
        uv = uv[:2] / uv[2]
        u = np.round(uv[0]).astype(int)
        v = np.round(uv[1]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        idx = valid_idx[ok]
        uu, vv = u[ok], v[ok]
        dist = np.minimum.reduce([uu, w - 1 - uu, vv, h - 1 - vv]).astype(np.float64)
        wgt = np.clip(dist / EDGE_W, 0.0, 1.0)
        color_sum[idx] += img[vv, uu].astype(np.float64) * wgt[:, None]
        weight_sum[idx] += wgt

    bev = np.full((gx.size, 3), 255.0)
    m = weight_sum > 0
    bev[m] = color_sum[m] / weight_sum[m, None]
    return np.ascontiguousarray(bev.reshape(H, W, 3).astype(np.uint8)[:, :, ::-1])


def camera_thumbs(cams):
    thumbs = []
    for cam in CAMS:
        img = cv2.imread(os.path.join(ROOT, cams[cam]['filename']))
        h, w = img.shape[:2]
        nh = int(h * CAM_THUMB_W / w)
        img = cv2.resize(img, (CAM_THUMB_W, nh))
        cv2.putText(img, cam, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)
        thumbs.append(img)
    return thumbs


def compose_frame(bev, thumbs):
    bev_disp_w = CAM_THUMB_W * 3  # match the width of the cam rows
    bh, bw = bev.shape[:2]
    bev = cv2.resize(bev, (bev_disp_w, int(bh * bev_disp_w / bw)))
    row1 = np.hstack(thumbs[:3])
    row2 = np.hstack(thumbs[3:])
    gap = np.full((10, bev_disp_w, 3), 30, np.uint8)
    return np.vstack([bev, gap, row1, gap, row2])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    calibrated_sensor = {r['token']: r for r in load_json('calibrated_sensor')}
    frames = pick_random_samples(N_FRAMES)
    for i, cams in enumerate(frames):
        print(f'frame {i + 1}/{len(frames)}  {cams["CAM_FRONT"]["filename"]}')
        bev = render_bev(cams, calibrated_sensor)
        vis = compose_frame(bev, camera_thumbs(cams))
        out = os.path.join(OUT_DIR, f'frame_{i:02d}.png')
        cv2.imwrite(out, vis)
    print('done ->', OUT_DIR)


if __name__ == '__main__':
    main()
