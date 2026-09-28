# -*- coding: utf-8 -*-
"""Semantic lane/curb matching between the two temporal views of frame 18:
CAM_FRONT @ t-1s and CAM_FRONT_RIGHT @ t+1s.

Instead of point-descriptor matching, extract semantic ground features and
associate them as vectors in the pose-compensated common BEV:
  1. lane-marking pixels  : white/yellow color rules inside the road mask
  2. curb/road-boundary   : boundary of the YOLO road mask
  3. project both views' features into frame 18's ego ground plane
     (ego_pose motion compensation, same as the RGB BEV)
  4. vectorize each BEV feature map into line segments (Hough + merging)
  5. associate segments with Hungarian assignment on (angle, midpoint)
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = r'H:\datasets\nuscenes-mini'
META = r'H:\datasets\nuscenes-mini\v1.0-mini'
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\outputs\bev\frames\frame18_lane_match'

RES = 0.05
X_MIN, X_MAX = -25.0, 45.0
Y_MIN, Y_MAX = -25.0, 25.0
H = int((X_MAX - X_MIN) / RES)
W = int((Y_MAX - Y_MIN) / RES)

MAX_ANGLE_DIFF_DEG = 20.0   # segment association gates
MAX_MIDPOINT_DIST_M = 2.0
ANGLE_W = 1.0               # association cost = ANGLE_W * (ang/20) + DIST_W * (dist/2)
DIST_W = 1.0

sys.path.insert(0, r'H:\projects\auto-extrinsics-adjusting')
from bev_prevnext import frame18_sample_data, load_json, quat_to_mat, walk_chain  # noqa


def T_of(pose):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(pose['rotation'])
    T[:3, 3] = pose['translation']
    return T


def extract_lane_pixels(img, road_mask):
    """White + yellow lane-marking pixels inside the road mask.

    A morphological top-hat on V makes this adaptive to absolute brightness
    (sunlit asphalt stays out, shaded markings stay in)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kern)
    thin_bright = tophat > 45
    white = thin_bright & (s < 110)
    yellow = thin_bright & (h >= 10) & (h <= 45) & (s >= 60)
    lane = (white | yellow) & road_mask
    # connect dashes and suppress specks
    lane = cv2.morphologyEx(lane.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    lane = cv2.morphologyEx(lane, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return lane.astype(bool)


def extract_curb(road_mask):
    """Road boundary: road-mask edge = mask minus its erosion."""
    er = cv2.erode(road_mask.astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return (road_mask.astype(np.uint8) - er).astype(bool)


def project_to_ref(pts_ego, T_ge, T_ref_inv):
    p_glob = T_ge @ np.vstack([pts_ego, np.ones(pts_ego.shape[1])])
    return (T_ref_inv @ p_glob)[:3]


def ego_pts_from_pixels(u_px, v_px, K, R, t):
    rays = np.linalg.inv(K) @ np.vstack([u_px, v_px, np.ones_like(u_px)]).astype(float)
    Rd = R @ rays
    s = -t[2] / Rd[2]
    return R @ (rays * s) + t[:, None]


def bev_canvas_from_pixels(u_px, v_px, K, R, t, T_ge, T_ref_inv):
    """Image pixels -> common-BEV bool canvas (with ego-frame coordinates)."""
    pts = ego_pts_from_pixels(u_px.astype(float), v_px.astype(float), K, R, t)
    p_ref = project_to_ref(pts, T_ge, T_ref_inv)
    u = (Y_MAX - p_ref[1]) / RES
    v = (X_MAX - p_ref[0]) / RES
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    canvas = np.zeros((H, W), np.uint8)
    canvas[v[ok].astype(int), u[ok].astype(int)] = 255
    return canvas, p_ref, ok


def extract_segments(canvas):
    """Canvas -> list of segments (x1, y1, x2, y2) in BEV pixel coords."""
    c = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    lines = cv2.HoughLinesP(c, 1, np.pi / 180, threshold=30,
                            minLineLength=25, maxLineGap=6)
    segs = [] if lines is None else [tuple(l) for l in np.asarray(lines).reshape(-1, 4)]
    return merge_segments(segs)


def merge_segments(segs, ang_tol_deg=8.0, dist_tol_px=20):
    """Greedily merge near-collinear segments."""
    def seg_stats(s):
        x1, y1, x2, y2 = s
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        ln = np.hypot(x2 - x1, y2 - y1)
        return ang, mid, ln

    segs = [s for s in segs if seg_stats(s)[2] >= 25]
    merged = True
    while merged:
        merged = False
        out = []
        used = [False] * len(segs)
        for i in range(len(segs)):
            if used[i]:
                continue
            ai, mi, li = seg_stats(segs[i])
            cur = list(segs[i])
            for j in range(i + 1, len(segs)):
                if used[j]:
                    continue
                aj, mj, lj = seg_stats(segs[j])
                d = np.hypot((cur[0] + cur[2]) / 2 - mj[0],
                             (cur[1] + cur[3]) / 2 - mj[1])
                da = min(abs(ai - aj), 180 - abs(ai - aj))
                if da < ang_tol_deg and d < dist_tol_px:
                    # merge by longest extension: keep convex hull endpoints
                    pts = [(cur[0], cur[1]), (cur[2], cur[3]),
                           (segs[j][0], segs[j][1]), (segs[j][2], segs[j][3])]
                    dx, dy = segs[j][2] - segs[j][0], segs[j][3] - segs[j][1]
                    n = np.hypot(dx, dy) + 1e-9
                    ux, uy = dx / n, dy / n
                    projs = [((p[0]-cur[0])*ux + (p[1]-cur[1])*uy, p) for p in pts]
                    lo, hi = min(projs), max(projs)
                    cur = [int(round(lo[1][0])), int(round(lo[1][1])),
                           int(round(hi[1][0])), int(round(hi[1][1]))]
                    used[j] = True
                    merged = True
                    ai, mi, li = seg_stats(tuple(cur))
            out.append(tuple(cur))
        segs = out
    return segs


def associate(segs_a, segs_b, scale=RES):
    """Hungarian assignment on (angle, midpoint distance). Returns pairs."""
    def stats(s):
        x1, y1, x2, y2 = s
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        return ang, ((x1 + x2) / 2 * scale, (y1 + y2) / 2 * scale)  # meters

    if not segs_a or not segs_b:
        return []
    cost = np.full((len(segs_a), len(segs_b)), 1e6)
    for i, sa in enumerate(segs_a):
        aa, ma = stats(sa)
        for j, sb in enumerate(segs_b):
            ab, mb = stats(sb)
            da = min(abs(aa - ab), 180 - abs(aa - ab))
            dm = np.hypot(ma[0] - mb[0], ma[1] - mb[1])
            if da <= MAX_ANGLE_DIFF_DEG and dm <= MAX_MIDPOINT_DIST_M:
                cost[i, j] = ANGLE_W * da / MAX_ANGLE_DIFF_DEG + \
                             DIST_W * dm / MAX_MIDPOINT_DIST_M
    ri, cj = linear_sum_assignment(cost)
    return [(segs_a[i], segs_b[j], cost[i, j])
            for i, j in zip(ri, cj) if cost[i, j] < 1e5]


def draw_segments(img, segs, color, thick=2):
    for s in segs:
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])),
                 color, thick, cv2.LINE_AA)


def main():
    calibrated = {r['token']: r for r in load_json('calibrated_sensor')}
    egos = {r['token']: r for r in load_json('ego_pose')}
    cams18, sd_by_token = frame18_sample_data()
    T_ref = T_of(egos[cams18['CAM_FRONT']['ego_pose_token']])
    T_ref_inv = np.linalg.inv(T_ref)

    os.makedirs(OUT_DIR, exist_ok=True)
    plan = [('minus1s_front', 'CAM_FRONT', 'prev'),
            ('plus1s_front_right', 'CAM_FRONT_RIGHT', 'next')]
    views = {}
    for tag, cam, key in plan:
        sd = walk_chain(sd_by_token[cams18[cam]['token']], key, 1.0, sd_by_token)
        calib = calibrated[sd['calibrated_sensor_token']]
        R = quat_to_mat(calib['rotation'])
        t = np.array(calib['translation'])
        K = np.array(calib['camera_intrinsic'])
        img = cv2.imread(os.path.join(ROOT, sd['filename']))
        road_mask = cv2.imread(os.path.join(
            OUT_DIR, '..', 'frame18_neighbor_bev', f'{tag}_mask_{cam}.png'),
            0) > 127
        lane = extract_lane_pixels(img, road_mask)
        curb = extract_curb(road_mask)

        # sanity: highlight extracted lane pixels on the image
        vis = img.copy()
        vis[lane] = (0, 255, 255)
        cv2.imwrite(os.path.join(OUT_DIR, f'lane_pixels_{tag}_{cam}.png'), vis)

        feats = {}
        for name, px in [('lane', lane), ('curb', curb)]:
            vv, uu = np.nonzero(px)
            canvas, p_ref, ok = bev_canvas_from_pixels(
                uu, vv, K, R, t, T_of(egos[sd['ego_pose_token']]), T_ref_inv)
            feats[name] = dict(canvas=canvas, segs=extract_segments(canvas))
            print(f'{tag} {cam} {name}: {ok.sum()} px in view, '
                  f'{len(feats[name]["segs"])} BEV segments')
        views[tag] = dict(cam=cam, K=K, R=R, t=t,
                          T_ge=T_of(egos[sd['ego_pose_token']]), feats=feats)

    # --- associate segments per feature type ----------------------------
    report = {}
    pairs_all = {}
    for name in ['lane', 'curb']:
        pairs = associate(views['minus1s_front']['feats'][name]['segs'],
                          views['plus1s_front_right']['feats'][name]['segs'])
        pairs_all[name] = pairs
        matched_a = {id(a) for a, b, c in pairs}
        report[name] = (len(views['minus1s_front']['feats'][name]['segs']),
                        len(views['plus1s_front_right']['feats'][name]['segs']),
                        len(pairs))
        print(f'{name}: matched {len(pairs)} segment pairs')

    # --- visualization ---------------------------------------------------
    # 1) feature-type BEVs: lane = red/green, curb = orange/purple, overlap yellow
    bev = np.full((H, W, 3), 255, np.uint8)
    STYLE = {'minus1s_front': dict(lane=(0, 0, 255), curb=(0, 120, 255)),
             'plus1s_front_right': dict(lane=(0, 160, 0), curb=(160, 0, 160))}
    for tag, vw in views.items():
        for name in ['lane', 'curb']:
            bev[vw['feats'][name]['canvas'] > 0] = STYLE[tag][name]
    # overlap cells in yellow
    la = views['minus1s_front']['feats']['lane']['canvas'] > 0
    lb = views['plus1s_front_right']['feats']['lane']['canvas'] > 0
    ca = views['minus1s_front']['feats']['curb']['canvas'] > 0
    cb = views['plus1s_front_right']['feats']['curb']['canvas'] > 0
    bev[(la & lb) | (ca & cb)] = (0, 220, 255)

    # 2) segment association drawing
    seg_img = bev.copy()
    for name, mk in [('lane', 3), ('curb', 2)]:
        draw_segments(seg_img, views['minus1s_front']['feats'][name]['segs'],
                      (60, 60, 60), 1)      # dim gray: unpaired front
        draw_segments(seg_img, views['plus1s_front_right']['feats'][name]['segs'],
                      (60, 60, 60), 1)
    for name in ['lane', 'curb']:
        for a, b, c in pairs_all[name]:
            draw_segments(seg_img, [a], (0, 0, 200), 2)   # front matched: red
            draw_segments(seg_img, [b], (0, 180, 0), 2)   # fr matched: green
            ma = ((a[0]+a[2])//2, (a[1]+a[3])//2)
            mb = ((b[0]+b[2])//2, (b[1]+b[3])//2)
            cv2.line(seg_img, ma, mb, (0, 220, 255), 1, cv2.LINE_AA)

    legend = [('front lane @ t-1s', (0, 0, 255)),
              ('front-right lane @ t+1s', (0, 160, 0)),
              ('front curb', (0, 120, 255)),
              ('front-right curb', (160, 0, 160)),
              ('overlap (matched)', (0, 220, 255)),
              ('associated segment pair', (255, 255, 255))]
    for i, (txt, c) in enumerate(legend):
        y = 30 + i * 30
        cv2.rectangle(bev, (10, y - 12), (34, y + 8), c, -1)
        cv2.putText(bev, txt, (44, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)
    cv2.circle(bev, (W // 2, int((X_MAX - 0) / RES)), 6, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(OUT_DIR, 'bev_features.png'), bev)
    cv2.imwrite(os.path.join(OUT_DIR, 'bev_segment_association.png'), seg_img)
    print('saved bev_features.png / bev_segment_association.py ->', OUT_DIR)


if __name__ == '__main__':
    main()
