# -*- coding: utf-8 -*-
"""Recover LiDAR pitch/roll from vertical planes + ground, with PER-SWEEP
IMU gravity comparison and joint optimization.

Each sweep t has its own ego attitude R_ge(t) (vehicle pitch on slopes), so
the gravity direction in the lidar frame varies per sweep:
    g_l(t) = R_le^T @ R_ge(t)^T @ ẑ_global
Per sweep we estimate a plane-based gravity candidate (vertical-plane
eigenvector + ground normal) and COMPARE it with the IMU direction. The
mounting correction delta is then optimized so that ALL sweeps' plane
constraints are satisfied jointly under R_le(delta) = Exp(delta) @ R_noisy."""
import os
import sys

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = r'H:\datasets\nuscenes-mini'
META = r'H:\datasets\nuscenes-mini\v1.0-mini'
OUT_DIR = r'H:\projects\auto-extrinsics-adjusting\extrinsic_recovery\results'
LOG_PREFIX = 'n008-2018-08-01-15-16-36-0400'
N_SWEEPS = 12
NOISE_DEG = 2.0
FACADE_W = 0.05
SEED = 11

sys.path.insert(0, r'H:\projects\auto-extrinsics-adjusting')
from bev_prevnext import load_json, quat_to_mat  # noqa: E402
from bev_lane_match import T_of  # noqa: E402


def load_lidar_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]


def ransac_planes(pts, n_planes=20, iters=500, tol=0.03, min_inliers=500):
    remaining = pts
    planes = []
    rng = np.random.default_rng(0)
    for _ in range(n_planes):
        m = len(remaining)
        if m < min_inliers:
            break
        best_n, best_c, best_in = None, None, np.array([], int)
        for _ in range(iters):
            idx = rng.choice(m, 3, replace=False)
            p0, p1, p2 = remaining[idx]
            n = np.cross(p1 - p0, p2 - p0)
            ln = np.linalg.norm(n)
            if ln < 1e-6:
                continue
            n = n / ln
            dist = np.abs((remaining - p0) @ n)
            inl = np.nonzero(dist < tol)[0]
            if len(inl) > len(best_in):
                best_n, best_in = n, inl
        if best_n is None or len(best_in) < min_inliers:
            break
        P = remaining[best_in]
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c)
        n = Vt[2]
        dist = np.abs((remaining - c) @ n)
        best_in = np.nonzero(dist < tol)[0]
        if len(best_in) < min_inliers:
            break
        planes.append((n, -float(np.dot(n, c)), len(best_in)))
        remaining = np.delete(remaining, best_in, axis=0)
    return planes


def per_sweep_gravity(vert_normals, g0_prior):
    """Estimate gravity from near-vertical plane normals of ONE sweep.
    Anchor at prior g0 (tangent-plane solve, weak regularization)."""
    if len(vert_normals) == 0:
        return None
    e1 = np.cross(g0_prior, [1., 0., 0.])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(g0_prior, [0., 1., 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g0_prior, e1)
    w = np.array([len(vert_normals)] * len(vert_normals), float)
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for n in vert_normals:
        p, q = np.dot(n, e1), np.dot(n, e2)
        c = np.dot(n, g0_prior)
        A += w[0] * np.array([[p * p, p * q], [p * q, q * q]])
        b += -w[0] * c * np.array([p, q])
    A += np.eye(2) * 1e-3 * w[0]
    try:
        ab = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    g = g0_prior + ab[0] * e1 + ab[1] * e2
    g /= np.linalg.norm(g)
    # reliability: second-smallest eigenvalue ratio of the normal scatter
    M = sum(np.outer(n, n) for n in vert_normals)
    ev = np.linalg.eigvalsh(M)
    reliable = ev[1] > 5 * ev[0]
    return g, reliable


def main():
    calibrated = {r['token']: r for r in load_json('calibrated_sensor')}
    egos = {r['token']: r for r in load_json('ego_pose')}
    sensors = {s['token']: s['channel'] for s in load_json('sensor')}
    channel_of = {c['token']: sensors[c['sensor_token']]
                  for c in load_json('calibrated_sensor')}
    sd_all = load_json('sample_data')
    sweeps = [sd for sd in sd_all if sd['is_key_frame']
              and channel_of[sd['calibrated_sensor_token']] == 'LIDAR_TOP'
              and LOG_PREFIX in sd['filename']]
    sweeps.sort(key=lambda s: s['timestamp'])
    step = max(1, len(sweeps) // N_SWEEPS)
    sweeps = sweeps[::step][:N_SWEEPS]

    calib = calibrated[sweeps[0]['calibrated_sensor_token']]
    R_gt = quat_to_mat(calib['rotation'])
    rng = np.random.default_rng(SEED)
    er = np.radians(rng.uniform(-NOISE_DEG, NOISE_DEG))
    ep = np.radians(rng.uniform(-NOISE_DEG, NOISE_DEG))
    Cn = Rotation.from_euler('xyz', [er, ep, 0]).as_matrix()
    R_noisy = Cn @ R_gt
    print(f'injected mounting noise: roll {np.degrees(er):+.2f} deg, '
          f'pitch {np.degrees(ep):+.2f} deg')

    os.makedirs(OUT_DIR, exist_ok=True)
    sweep_data = []
    for si, sd in enumerate(sweeps):
        fn = os.path.join(ROOT, sd['filename'])
        if not os.path.exists(fn):
            import tarfile
            with tarfile.open(r'H:\datasets\v1.0-mini\v1.0-mini.tar') as tf:
                tf.extract(sd['filename'], ROOT)
        pts = load_lidar_bin(fn)
        planes = ransac_planes(pts)
        R_ge = T_of(egos[sd['ego_pose_token']])[:3, :3]
        g_imu_ego = R_ge.T @ np.array([0., 0., 1.])       # IMU up (ego)
        g_imu_l = R_noisy.T @ g_imu_ego                   # IMU up (lidar)
        vert, ground_n, ground_w = [], None, 0.0
        ground_d = 0.0
        cells_l, cell_ws = [], []
        for n_l, d_l, n_in in planes:
            n_ego = R_noisy @ n_l
            zn = n_ego[2]
            if abs(zn) < 0.25:
                vert.append(n_l)
            elif abs(zn) > 0.9 and n_in > ground_w:
                ground_n = n_l if zn > 0 else -n_l
                ground_d = d_l if zn > 0 else -d_l
                ground_w = float(n_in)
        # multi-cell ground normals: 4 m BEV cells fitted independently to
        # remove slope/settlement bias of a single global plane
        if ground_n is not None:
            dist_g = np.abs(pts @ ground_n + ground_d)
            inl_pts = pts[dist_g < 0.03]
            cx = np.floor(inl_pts[:, 0] / 4.0).astype(int)
            cy = np.floor(inl_pts[:, 1] / 4.0).astype(int)
            for key in np.unique(np.stack([cx, cy]), axis=1):
                sel = (cx == key[0]) & (cy == key[1])
                if sel.sum() < 150:
                    continue
                P = inl_pts[sel]
                c = P.mean(0)
                _, _, Vt = np.linalg.svd(P - c)
                n_c = Vt[2]
                if n_c @ ground_n < 0:
                    n_c = -n_c
                cells_l.append(n_c)
                cell_ws.append(float(sel.sum()))
        res_g = per_sweep_gravity(vert, g_imu_l)
        g_pl = res_g[0] if res_g is not None else None
        sweep_data.append(dict(si=si, vert=vert, ground=ground_n,
                               gw=ground_w, g_pl=g_pl,
                               g_imu=g_imu_l, R_ge=R_ge,
                               cells=cells_l, cell_w=cell_ws))
        msg = f'sweep {si}: {len(planes)} planes, {len(vert)} vertical'
        if g_pl is not None:
            ang = np.degrees(np.arccos(np.clip(abs(np.dot(g_pl, g_imu_l)), 0, 1)))
            msg += f', plane-gravity vs IMU: {ang:.2f} deg'
        else:
            msg += ', plane-gravity: n/a'
        msg += f', ground: {"yes" if ground_n is not None else "no"}'
        reliable = False
        if g_pl is not None:
            cosv = np.clip(abs(np.dot(g_pl, g_imu_l)), 0.0, 1.0)
            reliable = bool(np.degrees(np.arccos(cosv)) < 8.0)
            msg += f', reliable: {reliable}'
        print(msg)
        sweep_data[-1]['reliable'] = reliable

    # --- per-sweep gravity vs IMU comparison --------------------------------
    angs = []
    for sd_ in sweep_data:
        if sd_['g_pl'] is None:
            continue
        cosv = np.clip(abs(np.dot(sd_['g_pl'], sd_['g_imu'])), 0.0, 1.0)
        angs.append(np.degrees(np.arccos(cosv)))
    if angs:
        print(f'plane-gravity vs IMU gravity: mean {np.mean(angs):.2f} deg, '
              f'median {np.median(angs):.2f} deg over {len(angs)} sweeps')

    # --- joint optimization: 2 params in the gravity tangent plane ---------
    # rotations about the gravity axis are unobservable from planes/ground
    # and are NOT part of pitch/roll, so optimize only in the plane
    # perpendicular to the prior vertical (nominal up in lidar frame).
    g0_prior = R_noisy.T @ np.array([0., 0., 1.])
    e1 = np.cross(g0_prior, [1., 0., 0.])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(g0_prior, [0., 1., 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g0_prior, e1)

    def v_lidar(ab, sweep):
        dv = ab[0] * e1 + ab[1] * e2
        R_le = Rotation.from_rotvec(dv).as_matrix() @ R_noisy
        return R_le.T @ sweep['R_ge'].T @ np.array([0., 0., 1.])

    def total_loss(ab):
        loss = 0.0
        for sd_ in sweep_data:
            v = v_lidar(ab, sd_)
            # ground cells: strong per-sweep verticality constraints
            # (local normals are immune to single-plane slope bias)
            for n_c in sd_['cells']:
                loss += 0.5 * min((n_c @ v) ** 2, 0.25)
            if FACADE_W > 0 and sd_.get('reliable'):
                # facades carry ~1 deg real-world lean in this scene; kept
                # only as a weak validation term
                for n in sd_['vert']:
                    loss += FACADE_W * 0.5 * min((n @ v) ** 2, 0.25)
        return loss

    x0 = np.zeros(2)
    res = minimize(total_loss, x0, method='Nelder-Mead',
                   options=dict(initial_simplex=np.vstack(
                       [x0, x0 + np.eye(2) * np.radians(0.1)]),
                       xatol=1e-7, fatol=1e-9, maxiter=2000))
    ab = res.x
    delta = ab[0] * e1 + ab[1] * e2
    dR = Rotation.from_rotvec(delta).as_matrix()
    R_rec = dR @ R_noisy
    print(f'delta-R (lidar frame, deg): '
          f'rx={np.degrees(delta[0]):+.3f} ry={np.degrees(delta[1]):+.3f} '
          f'rz={np.degrees(delta[2]):+.3f}  (pitch/roll correction only)')
    err_no = np.degrees(np.linalg.norm(
        Rotation.from_matrix(Cn).as_rotvec()))
    err_re = np.degrees(np.linalg.norm(
        Rotation.from_matrix(Cn @ dR).as_rotvec()))
    print(f'|error| before: {err_no:.3f} deg  ->  after: {err_re:.3f} deg')

    np.save(os.path.join(OUT_DIR, 'lidar_R_le_recovered.npy'), R_rec)

    # BEV visualization of the first sweep colored by plane membership
    sd = sweeps[0]
    fn = os.path.join(ROOT, sd['filename'])
    pts = load_lidar_bin(fn)
    planes = ransac_planes(pts)
    colors = [(60, 60, 60), (0, 160, 0), (0, 0, 255), (255, 0, 0),
              (0, 200, 200), (200, 0, 200), (0, 140, 255), (180, 120, 0)]
    bev = np.full((700, 700, 3), 255, np.uint8)
    RES_B = 0.12

    def to_bev(P):
        u = np.round(P[:, 0] / RES_B + 350).astype(int)
        v = np.round(-P[:, 1] / RES_B + 350).astype(int)
        ok = (u >= 0) & (u < 700) & (v >= 0) & (v < 700)
        return u[ok], v[ok]

    u, v = to_bev(pts)
    bev[v, u] = (160, 160, 160)
    for pi, (n_l, d_l, n_in) in enumerate(planes[:7]):
        n_ego = R_noisy @ n_l
        if abs(n_ego[2]) >= 0.25:
            continue
        dist = np.abs(pts @ n_l + d_l)
        inl = dist < 0.03
        uu, vv = to_bev(pts[inl])
        bev[vv, uu] = colors[pi + 1]
    cv2.circle(bev, (350, 350), 5, (255, 0, 0), -1)
    cv2.putText(bev, f'vertical planes (colored), err after {err_re:.2f} deg',
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    out = os.path.join(OUT_DIR, 'lidar_vertical_planes_bev.png')
    cv2.imwrite(out, bev)
    print('saved:', out)


if __name__ == '__main__':
    main()
