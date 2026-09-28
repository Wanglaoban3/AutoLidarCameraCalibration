# -*- coding: utf-8 -*-
"""Full pipeline (ground mask -> TEED edges -> BEV projection) on the
temporal neighbors of frame 18: its prev and next key frames, for
CAM_FRONT and CAM_FRONT_RIGHT."""
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

ROOT = r'H:\datasets\nuscenes-mini'
META = r'H:\datasets\nuscenes-mini\v1.0-mini'
TEED_ROOT = r'H:\projects\TEED-main'
MODEL_PATH = r'H:\models\yolo26x-sem.pt'
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\outputs\bev\frames\frame18_neighbor_bev'

FRAME_INDEX = 18
RES = 0.05
X_MIN, X_MAX = -25.0, 45.0
Y_MIN, Y_MAX = -25.0, 25.0
EDGE_W = 8.0
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT']
SEED = 0
GROUND_CLASSES = {0, 1}   # road, sidewalk

sys.path.insert(0, TEED_ROOT)
os.chdir(TEED_ROOT)
from ted import TED  # noqa: E402

CKPT = os.path.join(TEED_ROOT, 'checkpoints', 'BIPED', '7', '7_model.pth')

H = int((X_MAX - X_MIN) / RES)
W = int((Y_MAX - Y_MIN) / RES)


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


def frame18_sample_data():
    sample_data = load_json('sample_data')
    sensors = {s['token']: s['channel'] for s in load_json('sensor')}
    channel_of = {c['token']: sensors[c['sensor_token']]
                  for c in load_json('calibrated_sensor')}
    by_sample = {}
    sd_by_token = {}
    for sd in sample_data:
        sd_by_token[sd['token']] = sd
        if sd['is_key_frame']:
            cam = channel_of[sd['calibrated_sensor_token']]
            if cam in CAMS:
                by_sample.setdefault(sd['sample_token'], {})[cam] = sd
    full = [cams for cams in by_sample.values()
            if all(c in cams for c in CAMS)]
    cams18 = random.Random(SEED).sample(full, 20)[FRAME_INDEX]
    return cams18, sd_by_token


def canny_edges(img):
    """Canny edge map at native resolution."""
    return cv2.Canny(img, 50, 150)


def segment_ground(img):
    r = yolo.predict(img, verbose=False)[0]
    sem = r.semantic_mask.data.cpu().numpy()
    return np.isin(sem, list(GROUND_CLASSES))


def run_frame(tag, cams, calibrated_sensor):
    """Full pipeline for one frame; writes masks, edge BEVs, RGB BEVs."""
    xs = np.arange(X_MAX - RES / 2, X_MIN, -RES)
    ys = np.arange(Y_MAX - RES / 2, Y_MIN, -RES)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    pts_ego = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])

    for cam in CAMS:
        if cam not in cams:
            continue
        sd = cams[cam]
        img = cv2.imread(os.path.join(ROOT, sd['filename']))
        h, w = img.shape[:2]
        calib = calibrated_sensor[sd['calibrated_sensor_token']]
        R = quat_to_mat(calib['rotation'])
        t = np.array(calib['translation'])
        K = np.array(calib['camera_intrinsic'])

        ground = segment_ground(img)
        edge = canny_edges(img)

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

        gmask = ground[vv, uu]
        gi, gw = idx[gmask], wgt[gmask]

        # ground RGB BEV (weighted mean)
        csum = np.zeros((H * W, 3), np.float64)
        np.add.at(csum, gi, img[vv, uu].astype(np.float64)[:, ::-1][gmask] * gw[:, None])
        wsum = np.zeros(H * W, np.float64)
        np.add.at(wsum, gi, gw)
        cov = wsum > 0
        rgb = np.full((H * W, 3), 255.0)
        rgb[cov] = csum[cov] / wsum[cov, None]
        rgb = np.ascontiguousarray(rgb.reshape(H, W, 3).astype(np.uint8)[:, :, ::-1])

        # ground-restricted TEED edge BEV (max response)
        ev = edge[vv, uu].astype(np.float64) * wgt * gmask
        eout = np.zeros(H * W, np.float64)
        np.maximum.at(eout, idx, ev)
        edge_bev = np.ascontiguousarray(eout.reshape(H, W)[::-1].astype(np.uint8))

        cv2.imwrite(os.path.join(OUT_DIR, f'{tag}_mask_{cam}.png'),
                    (ground * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUT_DIR, f'{tag}_rgb_{cam}.png'), rgb)
        cv2.imwrite(os.path.join(OUT_DIR, f'{tag}_edge_{cam}.png'), edge_bev)

        # combo: rgb | hot edge
        edg_c = cv2.applyColorMap(edge_bev, cv2.COLORMAP_HOT)
        cx = int((Y_MAX - 0) / RES)
        cy = int((X_MAX - 0) / RES)
        for p in (rgb, edg_c):
            cv2.circle(p, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(p, f'{tag} {cam}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(OUT_DIR, f'{tag}_combo_{cam}.png'),
                    np.hstack([rgb, edg_c]))
        print(f'{tag} {cam}: ground {ground.mean():.1%}, saved')


def walk_chain(sd, direction, seconds=1.0, sd_by_token=None):
    """Walk prev/next sample_data chain until ~seconds away from sd."""
    t0 = sd['timestamp']
    cur = sd
    while True:
        nxt = cur.get(direction)
        if not nxt:
            break
        cur = sd_by_token[nxt]
        if abs(cur['timestamp'] - t0) >= seconds * 1e6:
            break
    return cur


def main():
    global yolo
    from ultralytics import YOLO
    yolo = YOLO(MODEL_PATH)
    print('YOLO-seg loaded:', MODEL_PATH)

    os.makedirs(OUT_DIR, exist_ok=True)
    calibrated_sensor = {r['token']: r for r in load_json('calibrated_sensor')}
    cams18, sd_by_token = frame18_sample_data()

    # front camera: frame 1 s before; front-right: frame 1 s after
    plan = [('minus1s_front', 'CAM_FRONT', 'prev'),
            ('plus1s_front_right', 'CAM_FRONT_RIGHT', 'next')]
    for tag, cam, key in plan:
        sd = walk_chain(sd_by_token[cams18[cam]['token']], key, 1.0, sd_by_token)
        dt = (sd['timestamp'] - sd_by_token[cams18[cam]['token']]['timestamp']) / 1e6
        cams = {cam: sd}
        import tarfile
        if not os.path.exists(os.path.join(ROOT, sd['filename'])):
            with tarfile.open(r'H:\datasets\v1.0-mini\v1.0-mini.tar') as tf:
                print('extracting', sd['filename'])
                tf.extract(sd['filename'], ROOT)
        print(f'--- {tag}: {cam} dt={dt:+.2f}s ({sd["filename"]})')
        run_frame(tag, {cam: sd}, calibrated_sensor)
    print('done ->', OUT_DIR)


if __name__ == '__main__':
    main()
