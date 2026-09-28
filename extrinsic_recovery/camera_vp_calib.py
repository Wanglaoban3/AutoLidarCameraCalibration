# -*- coding: utf-8 -*-
"""Recover camera pitch/roll from building vertical lines (VP method).

GT = official nuScenes calibrated_sensor. Noise is injected into the
camera extrinsic roll/pitch (ego-frame), then:
  YOLO-seg(building) -> TEED edges (native res, inside buildings) ->
  near-vertical line segments -> RANSAC vertical vanishing point ->
  gravity direction in camera frame -> pitch/roll correction
  (yaw comes from the IMU/ego attitude and is left untouched).
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = r'H:\datasets\nuscenes-mini'
META = r'H:\datasets\nuscenes-mini\v1.0-mini'
CANBUS_DIR = r'H:\datasets\can_bus_extract\can_bus'
TEED_ROOT = r'H:\projects\TEED-main'
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\extrinsic_recovery\results'
CAM = 'CAM_FRONT'
LOG_PREFIXES = ['n008-2018-08-01-15-16-36-0400',   # Boston row houses
                'n015-2018-10-02-10-50-40+0800',   # Singapore high-rise
                'n015-2018-11-21-19-38-26+0800']   # Singapore high-rise
N_FRAMES = 30
NOISE_DEG = 2.0          # uniform +/- on roll and pitch
SEED = 7

sys.path.insert(0, TEED_ROOT)
os.chdir(TEED_ROOT)
from ted import TED  # noqa: E402

sys.path.insert(0, r'H:\projects\auto-extrinsics-adjusting')
from bev_prevnext import load_json, quat_to_mat  # noqa: E402
from bev_lane_match import T_of  # noqa: E402


def teed_edges(img, model, device):
    """Return (binary edge map, float probability map) at native resolution."""
    h, w = img.shape[:2]
    x = img.astype(np.float32)
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        x = cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x -= np.array([104.007, 116.669, 122.679], dtype=np.float32)
    x = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(device)
    with torch.no_grad():
        preds = model(x)
    fused = torch.sigmoid(preds[-1])[0, 0].cpu().numpy()
    prob = (fused - fused.min()) / (np.ptp(fused) + 1e-8)
    e = (prob * 255).astype(np.uint8)
    return e[:h, :w], prob[:h, :w]


def subpixel_refine(seg, gm, prob, gx, gy, min_pts=12):
    """Gradient-based sub-pixel localization: walk along the segment, at each
    sample search the gradient-magnitude peak in the perpendicular direction
    (quadratic interpolation), validate against the TEED probability map,
    then weighted TLS-fit the refined points. Returns (line_abc, length,
    ang_std_deg) where ang_std_deg is the residual angular scatter of the
    refined points -- the per-line quality metric."""
    x1, y1, x2, y2 = seg
    L = np.hypot(x2 - x1, y2 - y1)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    nx, ny = -uy, ux
    h, w = gm.shape
    ts = np.linspace(-L / 2 + 3, L / 2 - 3, max(16, int(L / 3)))
    pts, wts = [], []
    for t in ts:
        bx, by = mx + ux * t, my + uy * t
        xi, yi = int(round(bx)), int(round(by))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        vals = np.zeros(9)
        for ko, o in enumerate(range(-4, 5)):
            xx, yy = int(round(bx + nx * o)), int(round(by + ny * o))
            if 0 <= xx < w and 0 <= yy < h:
                vals[ko] = gm[yy, xx]
        k = int(np.argmax(vals))
        if vals[k] < 30:
            continue
        delta = 0.0
        if 0 < k < 8:
            den = vals[k - 1] - 2 * vals[k] + vals[k + 1]
            if abs(den) > 1e-9:
                delta = float(np.clip(0.5 * (vals[k - 1] - vals[k + 1]) / den,
                                      -1, 1))
        o_sub = (k - 4) + delta
        px_, py_ = bx + nx * o_sub, by + ny * o_sub
        pi_, pj_ = int(round(px_)), int(round(py_))
        if not (0 <= pi_ < w and 0 <= pj_ < h) or prob[pj_, pi_] < 0.2:
            continue                                  # TEED validation gate
        pts.append((px_, py_))
        wts.append(vals[k])
    if len(pts) < min_pts:
        return None, 0, 0.0
    pts = np.array(pts)
    wts = np.array(wts)
    mean = (pts * wts[:, None]).sum(0) / wts.sum()
    P = pts - mean
    M = (P * wts[:, None]).T @ P
    w, V = np.linalg.eigh(M)
    if w[1] < 6 * w[0]:
        return None, 0, 0.0                           # not line-like
    d = V[:, 1]
    n = np.array([-d[1], d[0]])
    c = -n @ mean
    ang_std = np.degrees(np.sqrt(max(w[0], 1e-9) / wts.sum()))
    return np.array([n[0], n[1], c]), L, ang_std


def line_abc(p1, p2):
    """3-homogeneous line l = (nx, ny, c) with n = (dy, -dx) / len."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ln = np.hypot(dx, dy)
    n = np.array([dy, -dx]) / ln
    c = -np.dot(n, p1)
    return np.array([n[0], n[1], c])


def intersect(l1, l2):
    A = np.vstack([l1[:2], l2[:2]])
    b = -np.array([l1[2], l2[2]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, b)


def estimate_vertical_dir(lines, lens, K):
    """Estimate world-vertical direction d_cam from segments, weighted by
    segment length: M = sum w (K^T l)(K^T l)^T, d = smallest eigenvector."""
    q = np.array([K.T @ l for l in lines])
    w_all = np.asarray(lens, float) / np.max(lens)
    q_norm = np.linalg.norm(q, axis=1)

    def fit(idx):
        M = np.zeros((3, 3))
        for k in idx:
            M += w_all[k] * np.outer(q[k], q[k])
        w, V = np.linalg.eigh(M)
        d = V[:, 0]
        return d / np.linalg.norm(d)

    def inliers(d, tol_sin=np.sin(np.radians(3.0))):
        res = np.abs(q @ d) / np.maximum(q_norm, 1e-9)
        return np.nonzero(res < tol_sin)[0]

    def vertical_vp_gate(d):
        """The vanishing point K d must lie near the image vertical axis;
        this rejects consensus on families of horizontal/oblique lines."""
        vp = K @ d
        return abs(vp[0]) < 0.25 * abs(vp[1])

    best = (None, -1, None)
    rng = np.random.default_rng(0)
    n = len(lines)
    if n < 2:
        return None, 0, []
    for _ in range(400):
        i, j = rng.choice(n, 2, replace=False)
        d = fit([i, j])
        if not vertical_vp_gate(d):
            continue
        inl = inliers(d)
        if len(inl) > best[1]:
            best = (d, len(inl), inl)
    if best[1] < 4:
        return None, best[1], []
    idx = list(best[2])
    d = fit(idx)
    # IRLS: repeatedly drop the worst-residual 20% to reject leaning lines
    for _ in range(5):
        res = np.abs(q @ d) / np.maximum(q_norm, 1e-9)
        order = np.argsort(res[idx])
        keep = [idx[k] for k in order[:max(4, int(0.8 * len(idx)))]]
        if len(keep) < 4:
            break
        idx = keep
        d = fit(idx)
    info = float(sum(lens[k] * w_all[k] for k in idx))  # measurement info
    ang_res = np.arcsin(np.clip(np.abs(q @ d) / np.maximum(q_norm, 1e-9), 0, 1))
    ang_std = float(np.degrees(np.std(ang_res[idx])))
    return d, len(idx), idx, info, ang_std


def scene_of_log_map():
    scene_meta = load_json('scene')
    log_meta = {l['token']: l['logfile'] for l in load_json('log')}
    return {log_meta[s['log_token']]: s['name'] for s in scene_meta}


_imu_cache = {}

def imu_gravity_window(logfile, scene, t_us, half_win_us=500000, dbg=False):
    """CAN-IMU gravity direction in the IMU frame for a timestamp window."""
    if dbg:
        print(f'[imu dbg] logfile={logfile!r} scene={scene!r}')
    if logfile not in _imu_cache:
        p = os.path.join(CANBUS_DIR, f'{scene}_ms_imu.json')
        if dbg:
            print(f'[imu dbg] path={p!r} exists={os.path.exists(p)}')
        if not os.path.exists(p):
            _imu_cache[logfile] = None
            return None
        with open(p, 'r', encoding='utf-8') as f:
            _imu_cache[logfile] = json.load(f)
    msgs = _imu_cache[logfile]
    if msgs is None:
        return None
    lo, hi = t_us - half_win_us, t_us + half_win_us
    accs = [m['linear_accel'] for m in msgs if lo <= m['utime'] <= hi]
    if len(accs) < 20:
        return None
    a = np.mean(np.array(accs, float), axis=0)
    n = np.linalg.norm(a)
    if n < 0.2:                       # physically implausible (≈1g in m/s²)
        return None
    return a / n


def estimate_horizon(h_lines, K):
    """Second vanishing-point input: fit the horizon line from near-horizontal
    building segments (all their pairwise intersections lie on it)."""
    pts = []
    for i in range(len(h_lines)):
        for j in range(i + 1, len(h_lines)):
            p = intersect(h_lines[i], h_lines[j])
            if p is not None and abs(p[0]) < 3e4 and abs(p[1]) < 3e4:
                pts.append(p)
    if len(pts) < 4:
        return None, 0
    pts = np.array(pts)
    best = (None, -1)
    rng = np.random.default_rng(1)
    for _ in range(300):
        i, j = rng.choice(len(pts), 2, replace=False)
        if np.allclose(pts[i], pts[j]):
            continue
        l = line_abc(pts[i], pts[j])
        n = l[:2] / np.hypot(l[0], l[1])
        c = l[2]
        res = np.abs(pts @ n + c)
        inl = int((res < 15.0).sum())
        if inl > best[1]:
            best = (l, inl)
    if best[1] < 4:
        return None, 0
    # total-least-squares refit of the horizon through inlier points
    nl2 = best[0][:2]
    res = np.abs(pts @ nl2 + best[0][2]) / np.hypot(nl2[0], nl2[1])
    keep = pts[res < 12.0]
    cmean = keep.mean(0)
    _, _, Vt = np.linalg.svd(keep - cmean)
    nd = Vt[1]
    l = np.array([nd[0], nd[1], -np.dot(nd, cmean)])
    return l, best[1]


def main():
    calibrated = {r['token']: r for r in load_json('calibrated_sensor')}
    egos = {r['token']: r for r in load_json('ego_pose')}
    sensors = {s['token']: s['channel'] for s in load_json('sensor')}
    channel_of = {c['token']: sensors[c['sensor_token']]
                  for c in load_json('calibrated_sensor')}
    sd_all = load_json('sample_data')
    frames = [sd for sd in sd_all if sd['is_key_frame']
              and channel_of[sd['calibrated_sensor_token']] == CAM
              and any(p in sd['filename'] for p in LOG_PREFIXES)]
    frames.sort(key=lambda s: s['timestamp'])
    frames = frames[:N_FRAMES]    # full time window across scenes

    # CAN-IMU gravity: windowed accelerometer direction per frame; the
    # IMU->ego rotation is calibrated per log against ego-pose attitudes
    scene_of_log = scene_of_log_map()
    frames_meta = []
    logfiles = [os.path.basename(sd['filename']).split('__')[0]
                for sd in frames]
    for sd, logfile in zip(frames, logfiles):
        a_dir = imu_gravity_window(logfile, scene_of_log.get(logfile),
                                   sd['timestamp'],
                                   dbg=(len(frames_meta) == 0))
        up_ego = T_of(egos[sd['ego_pose_token']])[:3, :3].T @ np.array(
            [0., 0., 1.])
        frames_meta.append(dict(a_dir=a_dir, up=up_ego, v_imu=None))
    R_ie_of = {}
    for lf in set(logfiles):
        pairs = [(m['a_dir'], m['up']) for m, l2 in zip(frames_meta, logfiles)
                 if l2 == lf and m['a_dir'] is not None]
        if len(pairs) >= 2:
            A = np.array([p[0] for p in pairs])
            Bup = np.array([p[1] for p in pairs])
            R_ie_of[lf] = Rotation.align_vectors(Bup, A)[0]
    for m, lf in zip(frames_meta, logfiles):
        if lf in R_ie_of and m['a_dir'] is not None:
            m['v_imu'] = R_ie_of[lf].apply(m['a_dir'])
    imu_ok = [m for m in frames_meta if m['v_imu'] is not None]
    if imu_ok:
        ag = [np.degrees(np.arccos(np.clip(np.dot(m['v_imu'], m['up']), 0, 1)))
              for m in imu_ok]
        print(f'CAN-IMU gravity available for {len(imu_ok)}/{len(frames_meta)} '
              f'frames; agreement with ego-pose attitude: '
              f'mean {np.mean(ag):.2f} deg, median {np.median(ag):.2f} deg')
    else:
        print('CAN-IMU gravity unavailable; falling back to ego-pose attitude')
    if True:  # diagnostics
        n_a = sum(m['a_dir'] is not None for m in frames_meta)
        print(f'[imu-diag] frames with a_dir: {n_a}/{len(frames_meta)}, '
              f'calibrated logs: {list(R_ie_of)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teed = TED().to(device)
    teed.load_state_dict(torch.load(
        os.path.join(TEED_ROOT, 'checkpoints', 'BIPED', '7', '7_model.pth'),
        map_location=device))
    teed.eval()
    from ultralytics import YOLO
    yolo = YOLO(r'H:\models\yolo26x-sem.pt')

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # mounting error is FIXED per sensor: sample once, measure every frame,
    # then aggregate the per-frame corrections (realistic online-calibration)
    er = np.radians(rng.uniform(-NOISE_DEG, NOISE_DEG))
    ep = np.radians(rng.uniform(-NOISE_DEG, NOISE_DEG))
    Cn = Rotation.from_euler('xyz', [er, ep, 0]).as_matrix()
    print(f'injected mounting noise: roll {np.degrees(er):+.2f} deg, '
          f'pitch {np.degrees(ep):+.2f} deg')
    errs_noisy, errs_rec = [], []
    info_list = []
    corr_list = []
    rows = []

    for fi, sd in enumerate(frames):
        calib = calibrated[sd['calibrated_sensor_token']]
        R_gt = quat_to_mat(calib['rotation'])
        K = np.array(calib['camera_intrinsic'])
        e_gt = Rotation.from_matrix(R_gt).as_euler('xyz')

        R_noisy = Cn @ R_gt
        e_no = Rotation.from_matrix(R_noisy).as_euler('xyz')

        img = cv2.imread(os.path.join(ROOT, sd['filename']))
        r = yolo.predict(img, verbose=False)[0]
        sem = r.semantic_mask.data.cpu().numpy()
        building = sem == 2
        building = cv2.dilate(building.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        edge, prob = teed_edges(img, teed, device)
        edge_b = ((edge > 40) & (building > 0)).astype(np.uint8) * 255

        # image gradients for sub-pixel edge localization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gm = cv2.magnitude(gx, gy)

        # LSD gives candidate segments; gradient-based refinement localizes
        # each edge to sub-pixel precision, TEED validates it
        lsd = cv2.createLineSegmentDetector()
        det = lsd.detect(gray)
        segs = [] if det[0] is None else [tuple(s) for s in
                                          np.asarray(det[0]).reshape(-1, 4)]
        bmask_d = cv2.dilate(building.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cands = []
        ang_stds = []
        for x1, y1, x2, y2 in segs:
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if not (0 <= mx < img.shape[1] and 0 <= my < img.shape[0]):
                continue
            if not bmask_d[my, mx]:
                continue
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dy < 50 or dx > 0.4 * dy:      # near-vertical candidates only
                continue
            l3, ln, astd = subpixel_refine((x1, y1, x2, y2), gm, prob,
                                           gx, gy)
            if l3 is None or ln < 60:
                continue
            cands.append((l3, (x1, y1, x2, y2), ln))
            ang_stds.append(astd)
        lines_abc = [c[0] for c in cands]
        lines_len = [c[2] for c in cands]

        if len(lines_abc) < 2:
            print(f'frame {fi}: too few vertical candidate lines')
            continue
        d_meas, n_inl, inl_idx, meas_info, line_std = estimate_vertical_dir(
            lines_abc, lines_len, K)
        if d_meas is None:
            print(f'frame {fi}: vertical-direction fit failed '
                  f'({len(cands)} candidate lines)')
            continue

        # gravity target: ego-pose attitude (post-processed, clean) wins.
        # CAN-IMU accel is contaminated by vehicle dynamics (measured
        # disagreement ~5 deg) and is used only as a cross-check diagnostic.
        T_ge = T_of(egos[sd['ego_pose_token']])
        g_target = T_ge[:3, :3].T @ np.array([0., 0., 1.])   # up in ego frame

        # second VP: horizon from near-horizontal building lines
        h_cands = []
        for x1, y1, x2, y2 in segs:
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if not (0 <= mx < img.shape[1] and 0 <= my < img.shape[0]):
                continue
            if not bmask_d[my, mx]:
                continue
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx < 100 or dx < 2.0 * dy:
                continue
            h_cands.append(line_abc((x1, y1), (x2, y2)))
        l_h, h_inl = estimate_horizon(h_cands, K)

        d_cam_noisy = R_noisy.T @ g_target     # gravity-up in cam (noisy)
        v_meas = d_meas / np.linalg.norm(d_meas)
        v2 = None
        if l_h is not None:
            v2 = K.T @ l_h
            nv = np.linalg.norm(v2)
            if nv > 1e-9:
                v2 /= nv
                if np.dot(v2, d_cam_noisy) < 0:
                    v2 = -v2
                # consistency gate: the horizon-based estimate measures the
                # SAME up direction; fuse only when it agrees with the
                # vertical-line estimate, otherwise it is a poison
                if np.dot(v2, v_meas) > np.cos(np.radians(5.0)):
                    v_meas = (v_meas + v2) / 2
                    v_meas /= np.linalg.norm(v_meas)
                else:
                    v2 = None
        v_meas /= np.linalg.norm(v_meas)
        if np.dot(v_meas, d_cam_noisy) < 0:
            v_meas = -v_meas

        u = R_noisy @ v_meas                    # measured up, in ego frame
        rr = Rotation.align_vectors(g_target[None, :], u[None, :])[0]
        e_c = Rotation.from_matrix(rr.as_matrix()).as_euler('xyz')
        C2 = Rotation.from_euler('xyz', [e_c[0], e_c[1], 0.0]).as_matrix()
        R_rec = C2 @ R_noisy

        # metrics: R_rec = C2 @ R_noisy equals R_gt iff C2 == Cn^{-1}, so the
        # post-error is the rotation angle of (Cn @ C2); pre-error is |Cn|.
        err_no = np.degrees(np.linalg.norm(
            Rotation.from_matrix(Cn).as_rotvec()))
        err_re = np.degrees(np.linalg.norm(
            (Rotation.from_matrix(Cn) * Rotation.from_matrix(C2)).as_rotvec()))
        errs_noisy.append(err_no)
        errs_rec.append(err_re)
        corr_list.append(C2)
        info_list.append(meas_info)
        rows.append((fi, np.degrees(er), np.degrees(ep),
                     np.degrees(e_c[0]), np.degrees(e_c[1]),
                     err_no, err_re, n_inl, len(cands)))
        print(f'frame {fi}: vertical-line inliers {n_inl}/{len(cands)} | '
              f'line ang std {line_std:.2f} deg | '
              f'noise (r,p)=({np.degrees(er):+.2f},{np.degrees(ep):+.2f})deg | '
              f'recovered correction (r,p)=({np.degrees(e_c[0]):+.2f},'
              f'{np.degrees(e_c[1]):+.2f})deg | '
              f'geodesic err {err_re:.3f}deg (noise {err_no:.3f}deg)')

        # visualization for the first two frames
        if fi < 2:
            vis = img.copy()
            vis[edge_b > 0] = (vis[edge_b > 0] * 0.4
                               + np.array([0, 140, 255]) * 0.6).astype(np.uint8)
            for k in inl_idx:
                x1, y1, x2, y2 = cands[k][1]
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 0, 255), 2)
            cv2.putText(vis, f'vertical inliers {n_inl}  '
                             f'err {err_re:.2f} deg',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            cv2.imwrite(os.path.join(OUT_DIR, f'camera_vp_frame{fi}.png'), vis)

    if len(corr_list) < 4:
        print('too few frames with vertical-line estimates; abort')
        return
    errs_noisy, errs_rec = np.array(errs_noisy), np.array(errs_rec)
    # multi-frame statistics: corrections should cluster around the true
    # mounting error; select the consistent ones (within 2 deg of the
    # component-wise median) and average only those
    rv = np.array([Rotation.from_matrix(C).as_rotvec() for C in corr_list])
    med = np.median(rv, axis=0)
    dev = np.degrees(np.linalg.norm(rv - med, axis=1))
    good = np.nonzero(dev < 2.0)[0]
    # inverse-variance weighting: measurement info = sum of inlier
    # (length x TEED probability) -> sigma_i ~ 1/sqrt(info_i)
    iw = np.array([info_list[k] for k in good], float)
    iw /= iw.mean()
    qs = np.array([Rotation.from_matrix(corr_list[k]).as_quat()
                   for k in good])
    qs *= np.sign(qs[:, [3]])
    # chordal-L2 weighted mean in a small-angle neighborhood:
    # weight each quaternion by measurement info
    q_mean = (qs * iw[:, None]).sum(axis=0)
    q_mean /= np.linalg.norm(q_mean)
    C2_mean = Rotation.from_quat(q_mean).as_matrix()
    print(f'correction selection: {len(good)}/{len(rv)} frames kept '
          f'(deviations deg: {np.array2string(np.sort(dev), precision=2)})')
    R_rec = C2_mean @ Cn @ R_gt
    err_agg = np.degrees(np.linalg.norm(
        Rotation.from_matrix(C2_mean @ Cn).as_rotvec()))

    print('\n=== camera VP recovery summary ===')
    print(f'frames used: {len(errs_noisy)}')
    print(f'per-frame |error| before: {errs_noisy.mean():.3f} deg '
          f'(= injected noise)')
    print(f'per-frame |error| after : {errs_rec.mean():.3f} deg')
    print(f'AGGREGATED (all-frame) |error| after: {err_agg:.3f} deg')
    np.save(os.path.join(OUT_DIR, 'camera_R_ec_recovered.npy'), R_rec)
    print('header: frame, noise_roll, noise_pitch, recovered_corr_roll, '
          'recovered_corr_pitch, noise_mag_deg, geodesic_err_deg, '
          'vp_inliers, lines')
    for r in rows:
        print('  ' + ', '.join(f'{v:.2f}' if isinstance(v, float) else str(v)
                               for v in r))


if __name__ == '__main__':
    main()
