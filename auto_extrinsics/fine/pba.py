# -*- coding: utf-8 -*-
"""PBACalib port (arXiv:2308.12629, RA-L 2023): plane-constrained
multi-frame bundle adjustment -- minimal faithful core for the
targetless nuScenes-mini fine-calibration protocol.

Paper -> port map (adaptations for v1.0-mini @2 Hz in reports/pba.md):
  COLMAP SfM tracks     -> SIFT cross-keyframe matching (ratio + mutual
                           test), two-view triangulation per (ref, i)
                           pair (star tracks; the paper uses multi-view
                           COLMAP tracks).
  COLMAP/BA poses       -> ORACLE nuScenes ego_pose chain (the paper
                           needs an odometry source for structure;
                           ego_pose is the GT stand-in -- ORACLE_NOTE).
  LiDAR planes (BALM)   -> facade RANSAC on the stacked cloud in
                           reference-ego (lines._fit_planes_ego reused
                           as-is; vertical facades only, ground excluded
                           -- measured ~1 deg road-crown bias on this
                           log would poison the normals).
  local-covariance      -> one-time, pose-independent membership test
  planarity test           per track: kNN covariance around the
                           triangulated point (planarity lambda2/lambda3)
                           + plane-distance band (wide 0.30 m round
                           first under coarse error, re-association at
                           the paper's 0.15 m after the first solve;
                           frozen association per round).
  joint BA (Ceres, Huber)-> bounded Huber least_squares over
                           x=[rotvec_le(3), rotvec_ec(3)] via
                           solver.apply_deltas, point-to-plane
                           residuals + weak prior; per-round holdout
                           publish gate.

Residual algebra (exact under the sensor->ego chain; see self-test):
  evidence u   : triangulated point in "base camera->ego" coordinates,
                 u = R_ec0 @ p_c + t_ec (p_c in the reference camera).
  camera error : u_est = Exp(d_ec)(u_true - t_ec) + t_ec, so the
                 correction is Exp(-x[3:])(u - t_ec) + t_ec; back to
                 reference-ego via the (oracle) T_ge_c.
  lidar error  : stacked point p_e = G_s(R_le0 p_l + t_le); under a
                 hypothesized error d the evidence cloud transforms as
                 M(d) = G Exp(d) G^-1 about c0 = G t_le (per sweep;
                 sweeps conjugate d within ~1 deg over the 0.5 s stack
                 -- sub-mm effect on the corrected plane, so the key
                 sweep's G is used for all plane points, documented).
                 Corrected plane: n_c = M(x[:3])^T n,
                 d_c = d + n.c0 - n_c.c0.
  residual     : n_c . p_corr + d_c  (meters); identically 0 at x =
                 true errors when the evidence is exact.
"""
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from auto_extrinsics.fine import lines as fl
from auto_extrinsics.fine.solver import apply_deltas

ORACLE_NOTE = ('[oracle-trajectory dependency] triangulation uses the '
               'nuScenes ego_pose chain as an odometry stand-in (the '
               'paper triangulates through LiDAR odometry/FAST-LIO2 '
               'poses); a real run must replace it with odometry of '
               'comparable accuracy.')

SIFT_FEATURES = 8000      # per keyframe
SIFT_RATIO = 0.8          # Lowe ratio (plain mutual matching fallback)
MIN_MATCHES = 30          # pair kept only above this (post gates)
GUIDED_SAMPSON = 1e-3     # epipolar gate for guided candidate proposals
GUIDED_K = 3              # descriptor-nearest candidates per keypoint
REPROJ_MAX_PX = 1.5       # two-view reprojection gate
MIN_TRI_ANGLE_DEG = 0.7   # triangulation convergence angle
DEPTH_MIN, DEPTH_MAX = 2.0, 60.0
FACADE_MASK_DILATE = 41   # px, tolerance for base-pose projection error
FACADE_MASK_Z_MAX = 35.0  # m, near wall segments only: forward-motion
                          # stereo has no convergence angle down-street,
                          # and 32-beam density collapses beyond ~35 m
DEPTH_CELL = 8            # px, lidar depth-map cell
DEPTH_TOL = 0.30          # relative triangulated-vs-lidar depth gate
DEPTH_TOL_ABS = 2.0       # m, absolute floor of the same gate
NB_K = 60                 # paper's 'l nearest neighbors' local patch (a
                          # fixed-radius ball at this density swallows
                          # the wall base + ground and is never planar)
PLANARITY_MIN = 10.0      # lambda2 / lambda3 (mid / smallest eigval)
NORMAL_ALIGN = 0.75       # |cos| local normal vs facade normal
GROUND_MARGIN = 0.3       # m, ground exclusion for the membership cloud
BAND_WIDE = 0.50          # m, membership round 1 (coarse-error open)
BAND_PAPER = 0.15         # m, membership round 2 (paper band)
F_SCALE = 0.05            # m, Huber scale on plane distances
PRIOR_SIGMA_DEG = 0.3     # weak MAP prior per DoF
BOX_DEG = 1.5             # +- bound per sensor rotvec
PUBLISH_RATIO = 0.8       # holdout median must drop below 0.8 * init
PUBLISH_P90 = 1.05        # holdout p90 must not grow above 1.05 * init
PUBLISH_FLOOR = 0.02      # m, absolute holdout-median improvement


# ---- features ------------------------------------------------------------

def detect_features(img, mask=None):
    """SIFT keypoints/descriptors on grayscale (ORB/HAMMING fallback).
    mask: optional detection mask (used to concentrate keypoints on the
    projected facade regions of the reference view)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
        return sift.detectAndCompute(gray, mask), cv2.NORM_L2
    except AttributeError:
        orb = cv2.ORB_create(nfeatures=SIFT_FEATURES)
        return orb.detectAndCompute(gray, mask), cv2.NORM_HAMMING


def match_guided(fg_ref, fg_i, kp_ref, des_ref, kp_i, des_i,
                 kmax=GUIDED_K, sampson_max=GUIDED_SAMPSON):
    """Epipolar-guided candidate proposals under the ORACLE relative
    pose: per reference keypoint, the kmax descriptor-nearest keypoints
    of frame i among those with Sampson error < sampson_max.

    Plain mutual + Lowe matching fails structurally on this data: on
    repetitive facades windows repeat ALONG epipolar rays under forward
    motion, so the true near-wall correspondent always loses the
    descriptor race to a far alias (identical texture, smaller
    viewpoint change) while remaining epipolar-perfect -- the ambiguity
    is unresolvable in image space. All candidates are triangulated and
    resolved by the LiDAR-depth gate (the paper's point-to-plane
    mismatch dismissal), which is annotation-free."""
    uv_r = np.array([k.pt for k in kp_ref])
    uv_i = np.array([k.pt for k in kp_i])
    Kinv = np.linalg.inv(fg_ref.K)
    x_r = (Kinv @ np.c_[uv_r, np.ones(len(uv_r))].T).T[:, :3]
    x_i = (Kinv @ np.c_[uv_i, np.ones(len(uv_i))].T).T[:, :3]
    T_ec = np.eye(4)
    T_ec[:3, :3], T_ec[:3, 3] = fg_ref.R_ec0, fg_ref.t_ec
    T_i_ref = np.linalg.inv(
        np.linalg.inv(fg_ref.T_ge_c @ T_ec) @ (fg_i.T_ge_c @ T_ec))
    R, t = T_i_ref[:3, :3], T_i_ref[:3, 3]
    E = np.cross(t, np.eye(3)) @ R
    E1 = x_r @ E.T                                  # E x_r   [n_ref, 3]
    E2 = x_i @ E                                    # x_i^T E [n_i, 3]
    num = E1 @ x_i.T                                # [n_ref, n_i]
    den = (E1[:, 0] ** 2 + E1[:, 1] ** 2 + E1[:, 2] ** 2)[:, None] \
        + (E2[:, 0] ** 2 + E2[:, 1] ** 2 + E2[:, 2] ** 2)[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        samp = np.abs(num) / np.maximum(den, 1e-12)
    dr2 = (des_ref.astype(np.float64) ** 2).sum(1)
    di2 = (des_i.astype(np.float64) ** 2).sum(1)
    idx_a, idx_b = [], []
    for s in range(0, len(uv_r), 512):
        sl = slice(s, s + 512)
        dd = dr2[sl, None] + di2[None, :] \
            - 2.0 * (des_ref[sl].astype(np.float64) @ des_i.T)
        dd[samp[sl] > sampson_max] = np.inf
        top = np.argpartition(dd, min(kmax, dd.shape[1] - 1),
                              axis=1)[:, :kmax]
        rows = np.repeat(np.arange(dd.shape[0]), kmax)
        cand_b = top.ravel()
        good = np.isfinite(dd[rows, cand_b])
        idx_a.append(rows[good] + s)
        idx_b.append(cand_b[good])
    return np.concatenate(idx_a), np.concatenate(idx_b)


# ---- triangulation ---------------------------------------------------------

def build_depth_map(fg, cell=DEPTH_CELL):
    """Sparse near-surface depth map of the reference view from its OWN
    stacked cloud projected at the BASE pose (per-cell 10th-percentile
    depth, 3x3-min dilated). Used to verify triangulated depths: on
    repetitive facades, cross-keyframe SIFT matches alias RADIALLY
    (windows align along epipolar rays under forward motion), giving
    epipolar-perfect matches at the wrong depth -- PBACalib's
    point-to-plane mismatch filter needs the same dismissal, done here
    against LiDAR depth (annotation-free, LiDAR-only)."""
    h, w = fg.shape
    p_e = fg.stacked_ego_ref()
    pc = (p_e - fg.t_ec) @ fg.R_ec0.T
    z = pc[:, 2]
    ok = (z > DEPTH_MIN) & (z < DEPTH_MAX)
    uv = (fg.K @ pc[ok].T).T[:, :2] / z[ok, None]
    gh, gw = int(np.ceil(h / cell)), int(np.ceil(w / cell))
    dm = np.full((gh, gw), np.inf)
    vi = np.clip((uv[:, 1] / cell).astype(int), 0, gh - 1)
    ui = np.clip((uv[:, 0] / cell).astype(int), 0, gw - 1)
    np.minimum.at(dm, (vi, ui), z[ok])
    dm3 = dm.copy()
    dm3[1:-1, 1:-1] = np.minimum(np.minimum(
        np.minimum(dm[1:-1, 1:-1], dm[:-2, 1:-1]), dm[2:, 1:-1]),
        np.minimum(np.minimum(dm[1:-1, :-2], dm[1:-1, 2:]),
                   np.minimum(dm[:-2, :-2], dm[2:, 2:])))
    return dm3, cell


def triangulate_pair(fg_ref, fg_i, uv_ref, uv_i, dyn_ok_ref, dyn_ok_i,
                     depth_map=None, dm_cell=DEPTH_CELL, verbose=False,
                     reproj_max=None):
    """Two-view triangulation under the oracle ego relative poses and the
    BASE sensor rotations. Returns dict with the 3D evidence:
      X_ref  [n,3] points in the reference CAMERA frame (fixed evidence),
      u      [n,3] in base camera->ego coords: R_ec0 X_ref + t_ec,
      ok     [n] bool gate: depth, angle, reprojection, dyn-mask, in-fov.
    verbose: print gate-component percentiles (gate calibration).
    reproj_max: reprojection gate override (build_tracks_pairs relaxes
    it -- depth adjudication, not the in-pair reproj, filters aliases)."""
    reproj_max = REPROJ_MAX_PX if reproj_max is None else reproj_max
    T_ec = np.eye(4)
    T_ec[:3, :3], T_ec[:3, 3] = fg_ref.R_ec0, fg_ref.t_ec
    T_wc_ref = fg_ref.T_ge_c @ T_ec                 # cam -> ego -> global
    T_wc_i = fg_i.T_ge_c @ T_ec
    T_ref_i = np.linalg.inv(T_wc_ref) @ T_wc_i      # cam_i -> cam_ref
    T_i_ref = np.linalg.inv(T_ref_i)                # cam_ref -> cam_i
    P1 = fg_ref.K @ np.eye(3, 4)
    P2 = fg_i.K @ T_i_ref[:3]
    X = cv2.triangulatePoints(P1, P2,
                              uv_ref.T.copy(), uv_i.T.copy()).T
    # DLM homogeneous sign is arbitrary: divide by the SIGNED w (clamping
    # negative w to a tiny positive epsilon inverts points through the
    # origin and yields bogus ~1e11 depths)
    w = X[:, 3:4]
    X = X[:, :3] / np.where(np.abs(w) < 1e-12, np.nan, w)
    Xi = (T_i_ref[:3, :3] @ X.T).T + T_i_ref[:3, 3]

    z1, z2 = X[:, 2], Xi[:, 2]
    uv1p = (fg_ref.K @ X.T).T[:, :2] / np.maximum(z1[:, None], 1e-9)
    uv2p = (fg_i.K @ Xi.T).T[:, :2] / np.maximum(z2[:, None], 1e-9)
    e1 = np.linalg.norm(uv1p - uv_ref, axis=1)
    e2 = np.linalg.norm(uv2p - uv_i, axis=1)
    C2 = T_ref_i[:3, 3]
    d1 = X / np.maximum(np.linalg.norm(X, axis=1)[:, None], 1e-9)
    d2 = (X - C2) / np.maximum(
        np.linalg.norm(X - C2, axis=1)[:, None], 1e-9)
    ang = np.degrees(np.arccos(np.clip((d1 * d2).sum(1), -1, 1)))
    h, w = fg_ref.shape
    inb1 = (uv_ref[:, 0] > 1) & (uv_ref[:, 0] < w - 2) \
        & (uv_ref[:, 1] > 1) & (uv_ref[:, 1] < h - 2)
    inb2 = (uv_i[:, 0] > 1) & (uv_i[:, 0] < w - 2) \
        & (uv_i[:, 1] > 1) & (uv_i[:, 1] < h - 2)
    ok = ((z1 > DEPTH_MIN) & (z1 < DEPTH_MAX)
          & (z2 > DEPTH_MIN) & (z2 < DEPTH_MAX)
          & (e1 < reproj_max) & (e2 < reproj_max)
          & (ang > MIN_TRI_ANGLE_DEG) & inb1 & inb2
          & dyn_ok_ref & dyn_ok_i)
    if depth_map is not None:
        # LiDAR-depth verification (radial-alias / mismatch filter)
        gh, gw = depth_map.shape
        vi = np.clip((uv_ref[:, 1] / dm_cell).astype(int), 0, gh - 1)
        ui = np.clip((uv_ref[:, 0] / dm_cell).astype(int), 0, gw - 1)
        z_l = depth_map[vi, ui]
        tol = np.maximum(DEPTH_TOL * z_l, DEPTH_TOL_ABS)
        ok &= np.isfinite(z_l) & (np.abs(z1 - z_l) < tol)
    if verbose:
        def pct(v, m):
            v = v[m & np.isfinite(v)]
            if len(v) == 0:
                return '  --  '
            return np.array2string(np.percentile(v, [5, 50, 95]),
                                   precision=2)
        allm = np.isfinite(z1) | np.isfinite(z2)
        dep = ((z1 > DEPTH_MIN) & (z1 < DEPTH_MAX)
               & (z2 > DEPTH_MIN) & (z2 < DEPTH_MAX)).sum()
        rep = ((e1 < REPROJ_MAX_PX) & (e2 < REPROJ_MAX_PX)).sum()
        anm = (ang > MIN_TRI_ANGLE_DEG).sum()
        fov = (inb1 & inb2).sum()
        dyn = (dyn_ok_ref & dyn_ok_i).sum()
        print(f'   gate z1 {pct(z1, allm)} | e1 {pct(e1, allm)} | e2 '
              f'{pct(e2, allm)} | ang {pct(ang, allm)} | pass d {dep} r '
              f'{rep} a {anm} fov {fov} dyn {dyn} / {len(ok)}')
    u = (fg_ref.R_ec0 @ X[ok].T).T + fg_ref.t_ec
    return dict(X_ref=X[ok], u=u, uv_ref=uv_ref[ok], ok=ok, X_all=X,
                n_in=int(len(X)), n_ok=int(ok.sum()))


def _proj_ref(frames, ref_idx):
    """Projection matrices of every frame in the reference CAMERA frame:
    P_ref = K [I|0], P_j = K T_{j<-ref}."""
    T_ec = np.eye(4)
    T_ec[:3, :3], T_ec[:3, 3] = frames[ref_idx].R_ec0, frames[ref_idx].t_ec
    T_wc_ref = frames[ref_idx].T_ge_c @ T_ec
    P = {ref_idx: frames[ref_idx].K @ np.eye(3, 4)}
    for j, fg in enumerate(frames):
        if j == ref_idx:
            continue
        T_j_ref = np.linalg.inv(T_wc_ref) @ (fg.T_ge_c @ T_ec)
        P[j] = fg.K @ T_j_ref[:3]
    return P


def _dlt(P_list, uvs):
    """N-view DLT: X (3,) from homogeneous least squares over views."""
    A = []
    for P, uv in zip(P_list, uvs):
        u, v = uv
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, vh = np.linalg.svd(np.asarray(A))
    Xh = vh[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    return Xh[:3] / Xh[3]


def build_tracks(frames, ref_idx, feats_ref, feats_full, depth_map=None,
                 verbose=False):
    """Multi-frame tracks from ONE reference frame (see
    build_tracks_all for the multi-reference driver). Observations are
    gated pairwise (reprojection, angle, LiDAR-depth verification --
    which also resolves the radial-alias ambiguity of repetitive
    facades), grouped per reference keypoint and re-triangulated by
    N-view DLT when the views agree; each track keeps the ego bridge of
    ITS reference frame (tracks from different reference frames are
    assembled in build_tracks_all)."""
    fg_ref = frames[ref_idx]
    kp_ref = feats_ref[ref_idx][0][0]
    order = []
    for off in (1, -1, 2, -2, 3, -3, 4, -4):
        j = ref_idx + off
        if 0 <= j < len(frames):
            order.append(j)
    obs = []                                      # (kp_idx, j, uv_i)
    for j in order:
        fg_i = frames[j]
        (kp_i, des_i), _ = feats_full[j]
        t0 = time.time()
        idx_a, idx_b = match_guided(fg_ref, fg_i, kp_ref,
                                    feats_ref[ref_idx][0][1], kp_i, des_i)
        if len(idx_a) < 2:
            print(f'pair ref-f{ref_idx:02d} <-> f{j:02d}: 0 epipolar '
                  f'candidates')
            continue
        uv_ref = np.array([kp_ref[i].pt for i in idx_a])
        uv_i = np.array([kp_i[i].pt for i in idx_b])
        # NOTE: no GT-box dynamic gating on track acceptance (a targetless
        # run has no boxes; the placeholder dyn_mask stays unused here) --
        # misassociations on cars are rejected by the LiDAR-depth gate
        # and the plane band instead.
        dok_ref = np.ones(len(uv_ref), bool)
        dok_i = np.ones(len(uv_i), bool)
        res = triangulate_pair(fg_ref, fg_i, uv_ref, uv_i,
                               dok_ref, dok_i, depth_map=depth_map,
                               verbose=verbose)
        verbose = False
        keep = np.nonzero(res['ok'])[0]
        X_all = res['X_all']
        e_ref = np.linalg.norm(
            (fg_ref.K @ X_all[keep].T).T[:, :2]
            / X_all[keep, 2:3] - uv_ref[keep], axis=1)
        obs.extend((int(idx_a[i]), j, uv_i[i], X_all[i], float(er))
                   for i, er in zip(keep, e_ref))
        print(f'pair ref-f{ref_idx:02d} <-> f{j:02d}: {len(idx_a)} '
              f'candidates, {len(keep)} gate-surviving observations '
              f'({time.time() - t0:.1f}s)')
    # group per reference keypoint (uv_ref of the keypoint is itself an
    # observation); N-view DLT when >= 2 non-ref views agree, otherwise
    # keep the gated pairwise point
    groups = {}
    for k, j, uv, X2, e2 in obs:
        groups.setdefault(k, []).append((j, uv, X2, e2))
    P = _proj_ref(frames, ref_idx)
    h, w = fg_ref.shape
    tracks = []
    n_multi = n_fallback = 0
    for k, items in groups.items():
        uv_ref = np.array(kp_ref[k].pt, float)
        best = {}
        for j, uv, X2, e2 in items:
            if j not in best or e2 < best[j][3]:
                best[j] = (j, uv, X2, e2)
        vs = sorted(best.values(), key=lambda t: t[3])
        X, views = vs[0][2], [(ref_idx, uv_ref)] \
            + [(j, uv) for j, uv, _, _ in vs]
        if len(vs) >= 2:
            Xn = _dlt([P[j] for j, uv, _, _ in vs],
                      [uv for _, uv, _, _ in vs])
            if Xn is not None and np.isfinite(Xn).all():
                es = []
                for j, uv in views:
                    p = P[j] @ np.r_[Xn, 1.0]
                    es.append(np.linalg.norm(p[:2] / p[2] - uv))
                if np.median(es) <= REPROJ_MAX_PX:
                    X = Xn
                    n_multi += 1
                else:
                    n_fallback += 1
        if not (DEPTH_MIN < X[2] < DEPTH_MAX):
            continue
        if depth_map is not None:
            gh, gw = depth_map.shape
            vi = int(np.clip(uv_ref[1] / DEPTH_CELL, 0, gh - 1))
            ui = int(np.clip(uv_ref[0] / DEPTH_CELL, 0, gw - 1))
            z_l = depth_map[vi, ui]
            if not np.isfinite(z_l) \
                    or abs(X[2] - z_l) > max(DEPTH_TOL * z_l,
                                             DEPTH_TOL_ABS):
                continue
        tracks.append(dict(u=(fg_ref.R_ec0 @ X).T + fg_ref.t_ec,
                           X_ref=X, uv_ref=uv_ref,
                           frame=vs[0][0], n_views=len(views),
                           T_ge=fg_ref.T_ge_c))
    print(f'  ref f{ref_idx:02d}: {len(tracks)} points '
          f'({n_multi} N-view, {n_fallback} DLT-rejected -> pairwise '
          f'fallback)')
    return tracks


def build_tracks_pairs(frames, ref_idx, feats_ref, feats_full, depth_map,
                       offsets, verbose=False):
    """Short-pair star tracks with per-candidate LiDAR depth
    adjudication (the measured-correct evolution of build_tracks).

    Measurement history (dbg_track_window*, 2026-09-29):
      * v1.0-mini ego metadata is internally EXACT (keyframe, 20 Hz
        sweep and 12 Hz camera poses agree to 1 mm) -- poses exonerated.
      * KLT/NCC chains through the 12 Hz intermediate camera frames
        drift ~2 px/step on repetitive facades (LK and global-template
        agree; FB checks cannot see consensus slide) -> anchor chains
        are unusable at +-6 steps.
      * Direct keyframe-pair matching at 0.4-1 s baselines shows
        2-3 px oracle-epipolar consistency; at 2 s it degrades (17 px).
      * The 2 Hz matcher's real failure was the descriptor race:
        epipolar-perfect radial aliases win over the true far-wall
        correspondent.

    So: for each SHORT offset, keep the GUIDED_K epipolar-consistent
    candidates per reference keypoint (match_guided), triangulate them
    ALL (triangulate_pair, no depth gate inside), then rank per
    keypoint by |z_tri - z_lidar| at the reference pixel -- aliases
    sit ~one texture period off in depth, the true correspondent does
    not. Ambiguous votes (two candidates inside tolerance) are dropped.
    A keypoint winning at >= 2 offsets is re-triangulated by N-view
    DLT over its winning views. Returns the same track dicts."""
    fg_ref = frames[ref_idx]
    kp_ref = feats_ref[ref_idx][0][0]
    des_ref = feats_ref[ref_idx][0][1]
    gh, gw = depth_map.shape

    def z_lidar(uv):
        vi = int(np.clip(uv[1] / DEPTH_CELL, 0, gh - 1))
        ui = int(np.clip(uv[0] / DEPTH_CELL, 0, gw - 1))
        return depth_map[vi, ui]

    wins = {}                                     # kp_idx -> [(j, uv, X)]
    for off in offsets:
        j = ref_idx + off
        if not (0 <= j < len(frames)) or j == ref_idx:
            continue
        fg_i = frames[j]
        (kp_i, des_i), _ = (feats_full(j) if callable(feats_full)
                            else feats_full[j])
        idx_a, idx_b = match_guided(fg_ref, fg_i, kp_ref, des_ref,
                                    kp_i, des_i)
        if len(idx_a) < 2:
            continue
        uv_ref = np.array([kp_ref[i].pt for i in idx_a])
        uv_i = np.array([kp_i[i].pt for i in idx_b])
        ones = np.ones(len(uv_ref), bool)
        res = triangulate_pair(fg_ref, fg_i, uv_ref, uv_i, ones, ones,
                               reproj_max=PAIR_REPROJ_MAX_PX)
        e_ref = np.linalg.norm(
            (fg_ref.K @ res['X_all'][res['ok']].T).T[:, :2]
            / res['X_all'][res['ok'], 2:3] - uv_ref[res['ok']], axis=1)
        # depth of each surviving candidate vs the reference depth map
        n_ok = n_zl = n_tol = 0
        for c, k in enumerate(np.nonzero(res['ok'])[0]):
            i = int(idx_a[k])
            z_l = z_lidar(uv_ref[k])
            if not np.isfinite(z_l) or z_l > PAIR_Z_MAX:
                continue
            n_zl += 1
            tol = max(PAIR_DEPTH_TOL * z_l, DEPTH_TOL_ABS)
            err = abs(float(res['X_all'][k][2]) - float(z_l))
            if err <= tol:
                n_tol += 1
            wins.setdefault(i, []).append(
                (err, tol, j, uv_i[k].copy(), res['X_all'][k].copy(),
                 float(e_ref[c])))
        n_ok = int(res['ok'].sum())
        if verbose:
            print(f'  off {off:+d}: {len(idx_a)} guided candidates, '
                  f'{n_ok} tri-ok, {n_zl} near-field with lidar depth, '
                  f'{n_tol} in tolerance')
    # per keypoint: drop ambiguous votes, keep the depth-best candidate
    # per offset, then N-view DLT over >= 2 offset wins
    P = _proj_ref(frames, ref_idx)
    tracks = []
    n_amb = n_single = n_multi = 0
    for i, cands in wins.items():
        uv_ref = np.array(kp_ref[i].pt, float)
        best = {}
        for err, tol, j, uv, X, _ in cands:
            if err > tol:
                continue
            if j not in best or err < best[j][0]:
                best[j] = (err, j, uv, X)
        in_tol = [c for c in cands if c[0] <= c[1]]
        # ambiguity: any offset with 2+ candidates inside tolerance at
        # similar depth (within half the tolerance of each other)
        amb = False
        for c in in_tol:
            rivals = [o for o in in_tol
                      if o[2] == c[2] and o is not c
                      and abs(o[0] - c[0]) < 0.5 * c[1]]
            if rivals:
                amb = True
                break
        if amb:
            n_amb += 1
            continue
        vs = sorted(best.values(), key=lambda t: t[0])
        if not vs:
            continue
        X, views = vs[0][3], [(ref_idx, uv_ref)] \
            + [(j, uv) for _, j, uv, _ in vs]
        if len(vs) >= 2:
            Xn = _dlt([P[j] for _, j, _, _ in vs],
                      [uv for _, _, uv, _ in vs])
            if Xn is not None and np.isfinite(Xn).all():
                es = []
                for j, uv in views:
                    p = P[j] @ np.r_[Xn, 1.0]
                    es.append(np.linalg.norm(p[:2] / p[2] - uv))
                if np.median(es) <= REPROJ_MAX_PX:
                    X = Xn
                    n_multi += 1
        if not (DEPTH_MIN < X[2] < PAIR_Z_MAX):
            continue
        z_l = z_lidar(uv_ref)
        if not np.isfinite(z_l) or z_l > PAIR_Z_MAX \
                or abs(X[2] - z_l) > max(PAIR_DEPTH_TOL * z_l,
                                         DEPTH_TOL_ABS):
            continue
        n_single += 1
        # lidar-pinned evidence: same pixel ray, depth pinned to the
        # lidar map (kills the drift-correlated triangulation depth
        # bias -- recovery then lands the point on the true surface)
        X_pin = X * (z_l / X[2])
        tracks.append(dict(u=(fg_ref.R_ec0 @ X).T + fg_ref.t_ec,
                           u_pin=(fg_ref.R_ec0 @ X_pin).T + fg_ref.t_ec,
                           X_ref=X, uv_ref=uv_ref, frame=ref_idx,
                           z_l=float(z_l),
                           n_views=len(views), T_ge=fg_ref.T_ge_c))
    if verbose:
        print(f'  ref f{ref_idx:02d}: {len(tracks)} tracks '
              f'({n_multi} N-view, {n_amb} ambiguous-dropped, '
              f'{n_single} final)')
    return tracks


def local_plane_membership(points_e, cloud, tree, band, k=30,
                            plan_min=8.0, rms_max=0.15, debug=False):
    """Per-track LOCAL plane association: each track's k nearest LIDAR
    points define a candidate plane (PCA), kept when the patch is
    strongly planar and the track sits within `band` of it.

    Why not the global plane list: measured on scene-0103 (dbg13), the
    adjudicated tracks sit on an oblique intersection-corner facade
    that the merged top-12 plane set misses entirely (nearest global
    plane 4-6.6 m away for 100% of tracks), so the global funnel
    starves on plane COVERAGE, not on track quality. The local variant
    keeps the paper's physics -- constraint plane comes from LIDAR
    points only (pose-independent, computed once), image point supplies
    the evidence -- while covering walls of any orientation. Returns
    (sel [n] bool, dists [n], weights [n]); each selected track i is
    constrained by plane (n_i, d_i) = (normals[i], ds[i]) -- pass
    planes=list(zip(normals[sel], ds[sel])) with pk=identity to
    PBAResidual."""
    n_pts = len(points_e)
    k = min(k, len(cloud))
    _, nb = tree.query(points_e, k=k)
    sel = np.zeros(n_pts, bool)
    dists = np.full(n_pts, np.inf)
    normals = np.zeros((n_pts, 3))
    ds = np.zeros(n_pts)
    var_n = np.ones(n_pts)
    n_plan = 0
    for j in range(n_pts):
        Q = cloud[nb[j]]
        c = Q.mean(0)
        C = (Q - c).T @ (Q - c) / len(Q)
        evals, evecs = np.linalg.eigh(C)            # ascending
        lam_min, lam_mid = max(evals[0], 1e-9), evals[1]
        if lam_mid / lam_min < plan_min:
            continue
        n_loc = evecs[:, 0]
        d_loc = -float(n_loc @ c)
        rms = float(np.sqrt(max(evals[0], 0)))
        if rms > rms_max:
            continue
        d = abs(float(points_e[j] @ n_loc) + d_loc)
        if d > band:
            continue
        n_plan += 1
        sel[j] = True
        dists[j] = d
        normals[j] = n_loc
        ds[j] = d_loc
        var_n[j] = max(float(n_loc @ C @ n_loc), 1e-4)
    if debug:
        print(f'   [local membership] {n_plan}/{n_pts} locally '
              f'planar+associated @ band {band:.2f} m')
    w = 1.0 / var_n
    med = np.median(w[sel]) if sel.any() else 1.0
    w = np.clip(w / max(med, 1e-9), 0.25, 4.0)
    return sel, dists, w, normals, ds


def build_tracks_all(frames, masks, dms, feats_ref, feats_full,
                     verbose=False):
    """Multi-reference driver: run the star tracking from EVERY keyframe
    (each with its own facade mask, depth map and ego bridge) and pool
    all tracks. Triangulated points stay in their reference frame's ego
    (track['u'], track['T_ge']); the residual transforms each track
    through its own bridge, and membership maps everything into the
    scene reference ego. This is what lifts the evidence base to the
    paper's multi-frame scale on 2 Hz mini keyframes."""
    tracks = []
    for r in range(len(frames)):
        if not any(0 <= r + o < len(frames) for o in (1, -1, 2, -2)):
            continue
        t = build_tracks(frames, r, feats_ref, feats_full,
                         depth_map=dms[r], verbose=verbose)
        verbose = False
        tracks.extend(t)
    return tracks


# ---- tight-baseline (12 Hz) tracking ---------------------------------------

KLT_WIN = 21               # px, LK window (per-step motion is ~10-40 px
KLT_PYR = 4                # at city speed / 15 m depth -> pyramids)
KLT_FB_MAX = 1.0           # px, forward-backward round-trip gate per step
KLT_MIN_ANCHORS = 3        # keyframe observations incl. the center
KLT_ANCHOR_SPAN = 5        # min window-position span of the anchor set
KLT_REPROJ_MAX_PX = 1.5    # median DLT reprojection (= REPROJ_MAX_PX)
KLT_MAX_REPROJ_PX = 3.0    # per-view cap in the DLT check

# short-pair adjudication (build_tracks_pairs): constraint planes live
# at 3-16 m lateral, so far-scene triangulations are useless even when
# their RELATIVE depth error passes -- cap absolutely and tighten.
PAIR_REPROJ_MAX_PX = 3.0   # in-pair gate (depth adjudicates aliases)
PAIR_Z_MAX = 30.0          # m, absolute track depth cap
PAIR_DEPTH_TOL = 0.15      # relative adjudication tolerance (noise is
                           # ~2%; one texture period off is ~25%+)


def _proj_centers(frames, ref_idx):
    """Projection matrices AND camera centers of every window frame in
    the reference CAMERA frame (oracle ego chain, base sensor rotations;
    same bridges as _proj_ref plus C_j for the convergence-angle gate)."""
    fg0 = frames[ref_idx]
    T_ec = np.eye(4)
    T_ec[:3, :3], T_ec[:3, 3] = fg0.R_ec0, fg0.t_ec
    T_wc_ref = fg0.T_ge_c @ T_ec
    P = {ref_idx: fg0.K @ np.eye(3, 4)}
    C = {ref_idx: np.zeros(3)}
    for j, fg in enumerate(frames):
        if j == ref_idx:
            continue
        T_j_ref = np.linalg.inv(T_wc_ref) @ (fg.T_ge_c @ T_ec)
        P[j] = fg.K @ T_j_ref[:3]
        C[j] = -T_j_ref[:3, :3].T @ T_j_ref[:3, 3]
    return P, C


def _ray0(X):
    return X / max(float(np.linalg.norm(X)), 1e-9)


def track_window(frames, ref_idx, feats_ref, key_mask, depth_map=None,
                 verbose=False):
    """Star tracks over ONE window of consecutive ~10-12 Hz camera
    frames spanning +-~1 s around a 2 Hz keyframe center.

    MEASURED DATA CONSTRAINT (dbg_track_window2/3): triangulating
    THROUGH the intermediate ego poses is poisoned -- their epipolar
    consistency degrades from ~2 px (one frame out) to ~12 px (six
    frames out; irregular 50/100/200 ms trigger cadence + per-record
    pose time jitter), i.e. the 12 Hz pose chain carries meter-level
    ray inconsistency. The 2 Hz KEYFRAME poses are the proven-quality
    subset. So the intermediate frames contribute CORRESPONDENCE ONLY
    (pose-free KLT chaining, where the 83-100 ms steps make the true
    correspondent the photometric continuation of the point itself and
    kill the repetitive-facade radial alias BY CONSTRUCTION), while
    triangulation uses ONLY the keyframe anchors (>= 3 of them, e.g.
    k-1, k, k+1) via N-view DLT under the 2 Hz ego chain -- the same
    conditioning the 2 Hz pipeline had, without its descriptor-race
    aliasing.

    SIFT (facade-masked) on the reference; LK chained step-by-step
    forward AND backward; each step verified by a forward-backward
    round trip; a failed step drops the point from deeper frames only.
    Returns the same track dicts as build_tracks."""
    fg0 = frames[ref_idx]
    (kp0, _), _ = feats_ref[ref_idx]
    uv0 = np.array([k.pt for k in kp0], np.float32)
    grays = [cv2.cvtColor(f.img, cv2.COLOR_BGR2GRAY) for f in frames]
    lk = dict(winSize=(KLT_WIN, KLT_WIN), maxLevel=KLT_PYR,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30, 0.01))
    obs = {i: [(ref_idx, uv0[i])] for i in range(len(uv0))}
    n_drop = 0
    for sgn in (1, -1):
        chain = uv0.reshape(-1, 1, 2).copy()
        al = np.ones(len(uv0), bool)
        prev = ref_idx
        for step in range(1, len(frames)):
            j = ref_idx + sgn * step
            if not (0 <= j < len(frames)):
                break
            p2, st, _ = cv2.calcOpticalFlowPyrLK(grays[prev], grays[j],
                                                 chain, None, **lk)
            p1b, stb, _ = cv2.calcOpticalFlowPyrLK(grays[j], grays[prev],
                                                   p2, None, **lk)
            fb = np.linalg.norm((p1b - chain).reshape(-1, 2), axis=1)
            ok = al & (st.ravel() == 1) & (stb.ravel() == 1) \
                & (fb < KLT_FB_MAX)
            n_drop += int((al & ~ok).sum())
            for i in np.nonzero(ok)[0]:
                obs[i].append((j, p2[i, 0].copy()))
            chain, al, prev = p2, ok, j
    P, C = _proj_centers(frames, ref_idx)
    tracks = []
    for i, views in obs.items():
        anchors = [(j, uv) for j, uv in views if key_mask[j]]
        if len(anchors) < KLT_MIN_ANCHORS \
                or (max(j for j, _ in anchors)
                    - min(j for j, _ in anchors)) < KLT_ANCHOR_SPAN:
            continue
        X = _dlt([P[j] for j, _ in anchors],
                 [uv for _, uv in anchors])
        if X is None or not np.isfinite(X).all():
            continue
        rr = _ray0(X)
        es, angs = [], []
        for j, uv in anchors:
            p = P[j] @ np.r_[X, 1.0]
            if p[2] <= 0:
                es = None
                break
            es.append(np.linalg.norm(p[:2] / p[2] - uv))
            dj = (X - C[j]) / max(float(np.linalg.norm(X - C[j])), 1e-9)
            angs.append(np.degrees(np.arccos(np.clip(
                float(rr @ dj), -1, 1))))
        if es is None \
                or np.median(es) > KLT_REPROJ_MAX_PX \
                or max(es) > KLT_MAX_REPROJ_PX \
                or max(angs) < MIN_TRI_ANGLE_DEG:
            continue
        if not (DEPTH_MIN < X[2] < DEPTH_MAX):
            continue
        if depth_map is not None:
            gh, gw = depth_map.shape
            vi = int(np.clip(uv0[i][1] / DEPTH_CELL, 0, gh - 1))
            ui = int(np.clip(uv0[i][0] / DEPTH_CELL, 0, gw - 1))
            z_l = depth_map[vi, ui]
            if not np.isfinite(z_l) \
                    or abs(X[2] - z_l) > max(DEPTH_TOL * z_l,
                                             DEPTH_TOL_ABS):
                continue
        tracks.append(dict(u=(fg0.R_ec0 @ X).T + fg0.t_ec, X_ref=X,
                           uv_ref=uv0[i].astype(float), frame=ref_idx,
                           n_views=len(anchors), T_ge=fg0.T_ge_c))
    if verbose:
        vs = [t['n_views'] for t in tracks]
        print(f'  window ref f{ref_idx:02d}: {len(tracks)} tracks '
              f'(median anchors {int(np.median(vs)) if vs else 0}, '
              f'LK drops {n_drop})')
    return tracks


def _dyn_ok(fg, uv):
    """In-fov + dynamic-mask-free check for image points (annotation
    placeholder mask, same discipline as the repo's depth-edge path)."""
    h, w = fg.shape
    ok = (uv[:, 0] > 1) & (uv[:, 0] < w - 2) \
        & (uv[:, 1] > 1) & (uv[:, 1] < h - 2)
    if fg.dyn_mask is not None:
        ui = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        ok &= fg.dyn_mask[vi, ui] == 0
    return ok


# ---- lidar planes ----------------------------------------------------------

def facade_mask(fg, planes, cloud_tree=None):
    """Image mask of the facade-plane lidar inliers projected through the
    BASE pose (PBACalib's 'visual points derived from those planes' --
    matching is restricted to facade regions so tracks sit at the depths
    where triangulation is actually conditioned; the dilation absorbs the
    base-pose projection error of a coarse initialization)."""
    h, w = fg.shape
    mask = np.zeros((h, w), np.uint8)
    R, t = fg.R_ec0, fg.t_ec
    for n, d, inl in planes:
        pc = (inl - t) @ R.T
        z = pc[:, 2]
        ok = (z > DEPTH_MIN) & (z < FACADE_MASK_Z_MAX)
        uv = (fg.K @ pc[ok].T).T[:, :2] / z[ok, None]
        ui = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        mask[vi, ui] = 1
    if mask.any():
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (FACADE_MASK_DILATE, FACADE_MASK_DILATE)))
    return mask


def ground_plane_ego(p_e):
    """Ground plane (n, c) in reference-ego from the near-road band
    (same estimator as lines.contact_lines; the ground itself is NOT
    used as a constraint plane -- road crown biases its normal)."""
    band = ((p_e[:, 0] > -15.0) & (p_e[:, 0] < 70.0)
            & (np.abs(p_e[:, 1]) < 25.0)
            & (p_e[:, 2] > -1.5) & (p_e[:, 2] < 0.5))
    if int(band.sum()) < 200:
        return None
    C = p_e[band]
    c = np.median(C, axis=0)
    _, _, vh = np.linalg.svd(C - c, full_matrices=False)
    n = vh[-1]
    if n[2] < 0:
        n = -n
    return n, c


def extract_facades(frames, ref_idx, rng, debug=False):
    """Facade planes for the constraint set, extracted per frame from
    that frame's stacked cloud bridged to the REFERENCE ego (B =
    inv(T_ge_ref) T_ge_frame is the exact per-frame rigid bridge; a
    bridged frame cloud is internally coherent, unlike the raw union --
    the street curves, so a global RANSAC over all frames at once loses
    walls to tilt rejects). Duplicate walls across frames are merged
    (greedy, biggest first: |n.n|>0.994, |d|<1.0 m; normals averaged
    weighted by inlier count). Reuses the repo's vertical-facade RANSAC
    verbatim per frame (gates |n_z|<=0.06, |d|>=2.5 m). Returns
    (merged constraint planes, per-frame planes with inlier points for
    the mask, the bridged union cloud)."""
    T_ge_ref = frames[ref_idx].T_ge_c
    per_frame = []
    bridged = []
    for fg in frames:
        B = np.linalg.inv(T_ge_ref) @ fg.T_ge_c   # ego(frame) -> ego(ref)
        p = fg.stacked_ego_ref()
        p = p @ B[:3, :3].T + B[:3, 3]
        bridged.append(p)
        g = ground_plane_ego(p)
        if g is None:
            continue
        planes, _ = fl._fit_planes_ego(p, g[0], g[1], rng, debug=debug)
        per_frame.extend(planes)
    cloud = np.concatenate(bridged, axis=0)
    flat = sorted(per_frame, key=lambda t: -len(t[2]))
    merged = []                                   # [n, d, count]
    for n, d, inl in flat:
        for o in merged:
            if abs(float(n @ o[0])) > 0.994 and abs(d - o[1]) < 1.0:
                if n @ o[0] < 0:
                    n, d = -n, -d
                w1, w2 = o[2], len(inl)
                o[0] = o[0] * w1 + n * w2
                o[0] /= np.linalg.norm(o[0])
                o[1] = (o[1] * w1 + d * w2) / (w1 + w2)
                o[2] += w2
                break
        else:
            merged.append([n, d, len(inl)])
    merged = merged[:12]
    return [(o[0], o[1], o[2]) for o in merged], per_frame, cloud


# ---- membership (one-time, pose-independent, in the shared metric frame)

def plane_membership(points_e, cloud, tree, planes, band,
                     k=NB_K, debug=False):
    """Local-covariance membership test per triangulated point (the
    paper's l-nearest-neighbor patch): covariance of the k nearest lidar
    points, planarity lambda2/lambda3 >= PLANARITY_MIN, local normal
    aligned with the facade, and plane distance within the band. The
    cloud should be ground-filtered (GROUND_MARGIN) so patches stay on
    the facades. Returns (plane_idx [-1 none], dist, planarity,
    var_along_n weights)."""
    n_pts = len(points_e)
    plane_idx = np.full(n_pts, -1, np.int32)
    dists = np.full(n_pts, np.inf)
    plans = np.zeros(n_pts)
    vars_n = np.ones(n_pts)
    if not planes:
        return plane_idx, dists, plans, vars_n
    N = np.array([p[0] for p in planes])            # [k,3]
    D = np.array([p[1] for p in planes])            # [k]
    k = min(k, len(cloud))
    _, nb_all = tree.query(points_e, k=k)
    n_plan_ok = 0
    for j in range(n_pts):
        Q = cloud[nb_all[j]]
        qc = Q.mean(0)
        C = (Q - qc).T @ (Q - qc) / len(Q)
        evals, evecs = np.linalg.eigh(C)            # ascending
        lam_min, lam_mid = max(evals[0], 1e-9), evals[1]
        plan = lam_mid / lam_min
        if plan < PLANARITY_MIN:
            continue
        n_plan_ok += 1
        n_loc = evecs[:, 0]
        d_all = np.abs(points_e[j] @ N.T + D)       # [n_planes]
        best, bestd = -1, np.inf
        for pi in range(len(planes)):
            if d_all[pi] >= min(band, bestd):
                continue
            if abs(float(n_loc @ N[pi])) < NORMAL_ALIGN:
                continue
            best, bestd = pi, float(d_all[pi])
        if best < 0:
            continue
        plane_idx[j] = best
        dists[j] = bestd
        plans[j] = plan
        vn = max(float(n_loc @ C @ n_loc), 1e-4)
        vars_n[j] = vn
    if debug:
        dd = np.abs(points_e @ N.T + D).min(axis=1)
        print(f'   [membership dbg] k={k} plan-ok {n_plan_ok} / {n_pts}'
              f' | nearest-plane dist pct '
              f'{np.round(np.percentile(dd, [10, 50, 90]), 2)}')
    w = 1.0 / vars_n
    med = np.median(w[w > 0]) if (w > 0).any() else 1.0
    w = np.clip(w / max(med, 1e-9), 0.25, 4.0)
    return plane_idx, dists, plans, w


# ---- objective -------------------------------------------------------------

class PBAResidual:
    """Point-to-plane residuals of the constrained tracks under
    x=[rotvec_le(3), rotvec_ec(3)] (radians, apply_deltas convention),
    plus the weak prior rows. Weights fold in as sqrt(w).

    FRAMES (fixed 2026-09-29: the committed version fed scene-ego-frame
    planes to global-frame points -- never exposed while the funnel
    starved): planes N, D arrive in the SCENE-REFERENCE ego (where
    extract_facades / local_plane_membership fit them) and are converted
    to GLOBAL here via T_ge_ref = [R|t]: n_g = R n_e, d_g = d_e - n_g t.
    Evidence points stay global (per-track T_ge); the lidar conjugation
    M = R_ref Exp(d_le) R_ref^T acts about c0 = R_ref t_le + t_ref."""

    def __init__(self, tracks_u, track_plane, track_w, planes, T_ge_ref,
                 t_le, t_ec, T_ge_list, R_le0, R_ec0,
                 prior_sigma_deg=PRIOR_SIGMA_DEG):
        self.U = np.asarray(tracks_u, float)                # [n,3] per-
        # track ego (each track's own reference frame bridge T_ge below)
        self.pk = np.asarray(track_plane, np.int32)         # [n]
        self.sw = np.sqrt(np.asarray(track_w, float))       # [n]
        self.Rg, self.tg = T_ge_ref[:3, :3], T_ge_ref[:3, 3]
        Ne = np.array([p[0] for p in planes])               # [k,3] ego
        de = np.array([p[1] for p in planes])               # [k] ego
        self.N = Ne @ self.Rg.T                             # -> global
        self.D = de - self.N @ self.tg
        self.c0 = self.Rg @ t_le + self.tg                  # R_ref t_le
        self.t_le, self.t_ec = t_le, t_ec
        T = np.asarray(T_ge_list, float)                    # [n,4,4]
        if T.ndim == 2:          # one bridge shared by all (self-test)
            T = np.broadcast_to(T[None], (len(self.U), 4, 4)).copy()
        self.R_gc, self.t_gc = T[:, :3, :3], T[:, :3, 3]
        self.R_le0, self.R_ec0 = R_le0, R_ec0
        self.prior = 1.0 / np.radians(prior_sigma_deg)

    def set_association(self, pk, w, planes_ego):
        """Rebind the constrained subset (round-2 tight band): new pk/w
        plus per-track patch planes, converted ego->global exactly as
        in __init__. Call this instead of touching .pk/.N directly."""
        self.pk = np.asarray(pk, np.int32)
        self.sw = np.sqrt(np.asarray(w, float))
        Ne = np.array([p[0] for p in planes_ego])           # [k,3] ego
        de = np.array([p[1] for p in planes_ego])           # [k] ego
        self.N = Ne @ self.Rg.T                             # -> global
        self.D = de - self.N @ self.tg

    def raw(self, x):
        """Plane distances [n] at pose x (also the gate metric).

        SIGN CONVENTION (fixed 2026-09-29): recovery means
        apply_deltas recovers GT, i.e. x = -rotvec(Cn) for injection
        R0 = Cn @ R_gt. A drifted camera reconstructs the scene as
        Exp(+d_ec) applied about t_ec, and drifted lidar planes are
        M(d_le)-conjugated -- so UNDOING the drift rotates the camera
        evidence by +x_ec and the plane normals by M(-x_le) here; both
        sides then zero exactly at the extrinsic-recovering x."""
        R_le, R_ec = apply_deltas(self.R_le0, self.R_ec0, x)
        # camera side: hypothesized mount error applied to the evidence
        Ec = Rotation.from_rotvec(x[3:]).as_matrix()
        u_corr = (self.U - self.t_ec) @ Ec.T + self.t_ec
        p_corr = np.einsum('nij,nj->ni', self.R_gc, u_corr) + self.t_gc
        # lidar side: conjugated plane correction about c0
        El = Rotation.from_rotvec(-x[:3]).as_matrix()
        M = self.Rg @ El @ self.Rg.T
        Nk = self.N[self.pk]                                # [n,3]
        Dk = self.D[self.pk]                                # [n]
        n_c = Nk @ M          # M(x)-rotated normal: undo hypothesized
        # lidar error; (n_c, d_c) is the plane rotated about c0
        d_c = Dk + Nk @ self.c0 - (n_c * self.c0).sum(1)
        r = (n_c * p_corr).sum(1) + d_c
        return r, R_le, R_ec

    def cost(self, x, sel):
        r = self.raw(x)[0][sel] * self.sw[sel]
        return np.concatenate([r, x * self.prior])

    def solve(self, x0, sel, box_deg=BOX_DEG, f_scale=F_SCALE,
              max_nfev=200):
        B = np.radians(box_deg)
        return least_squares(self.cost, x0, args=(sel,),
                             bounds=(-B, B), loss='huber',
                             f_scale=f_scale, max_nfev=max_nfev)

    def gate(self, x, sel):
        """Holdout publish gate metric: (median |r|, p90 |r|, n)."""
        r = np.abs(self.raw(x)[0][sel])
        if len(r) < 20:
            return None, None, int(len(r))
        return float(np.median(r)), float(np.percentile(r, 90)), \
            int(len(r))

    def dof_sigma_deg(self, x, sel):
        """Posterior per-DoF sigma from the final linearization."""
        return _dof_sigma_deg(self.cost, self.raw, x, sel)


def _dof_sigma_deg(cost_fn, raw_fn, x, sel):
    """Posterior per-DoF sigma from the final linearization (J^T J over
    weighted residuals + prior rows, robust scale). Shared by
    PBAResidual and the FixedLidarPBA adapter."""
    eps = 1e-6
    r0 = cost_fn(x, sel)
    J = np.empty((len(r0), len(x)))
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        J[:, i] = (cost_fn(xp, sel) - cost_fn(xm, sel)) / (2 * eps)
    JTJ = J.T @ J
    med = np.median(np.abs(raw_fn(x)[0][sel]))
    sig = 1.4826 * max(med, 1e-3)
    try:
        cov = np.linalg.inv(JTJ) * sig ** 2
        return np.degrees(np.sqrt(np.clip(np.diag(cov), 0, None)))
    except np.linalg.LinAlgError:
        return None


class FixedLidarPBA:
    """cams-only adapter for the 'lidar mount factory-correct, camera
    mounts drifted' scenario: the state is x3 = rotvec_ec and the lidar
    correction stays at the identity (x[:3] pinned to 0). Same residual
    algebra, gates and Huber solve over the 3 free DoF."""

    def __init__(self, pb):
        self.pb = pb
        self.R_le0, self.R_ec0 = pb.R_le0, pb.R_ec0
        self.prior = pb.prior

    @property
    def pk(self):
        return self.pb.pk          # delegate: round-2 rebinds on pb

    @property
    def sw(self):
        return self.pb.sw

    def _full(self, x3):
        return np.r_[np.zeros(3), np.asarray(x3, float)]

    def raw(self, x3):
        return self.pb.raw(self._full(x3))

    def cost(self, x3, sel):
        r = self.raw(x3)[0][sel] * self.pb.sw[sel]
        return np.concatenate([r, np.asarray(x3, float) * self.prior])

    def solve(self, x3, sel, box_deg=BOX_DEG, f_scale=F_SCALE,
              max_nfev=200):
        B = np.radians(box_deg)
        return least_squares(self.cost, x3, args=(sel,), bounds=(-B, B),
                             loss='huber', f_scale=f_scale,
                             max_nfev=max_nfev)

    def gate(self, x3, sel):
        return self.pb.gate(self._full(x3), sel)

    def dof_sigma_deg(self, x3, sel):
        return _dof_sigma_deg(self.cost, self.raw, x3, sel)


# ---- residual algebra self-test --------------------------------------------

def self_test():
    """Synthetic check: residual(x = true errors) == 0, residual(0) > 0.
    Runs in <1 s; guards the correction algebra before any dataset run."""
    rng = np.random.default_rng(0)
    G = np.eye(4); G[:3, :3] = Rotation.from_rotvec(
        [0.01, -0.02, 0.03]).as_matrix(); G[:3, 3] = [0.5, -0.3, 1.8]
    t_le = np.array([0.0, 0.0, 1.8])
    t_ec = np.array([1.6, 0.0, 1.5])
    T_ge_c = np.eye(4)
    T_ge_c[:3, :3] = Rotation.from_rotvec([0.02, 0.01, -0.04]).as_matrix()
    T_ge_c[:3, 3] = [3.0, 1.0, 0.2]
    d_le = np.radians([0.4, -0.3, 0.5])
    d_ec = np.radians([-0.5, 0.2, -0.4])
    # true points on a true plane, in TRUE ego coords
    n_true = np.array([0.0, -1.0, 0.05]); n_true /= np.linalg.norm(n_true)
    d_true = -8.0
    pts_true = rng.normal(size=(400, 3)) * [10, 1, 5] + [20, 0, 2]
    pts_true += (d_true - n_true @ pts_true.T)[:, None] * n_true
    u_true = (T_ge_c[:3, :3].T @ (pts_true - T_ge_c[:3, 3]).T).T
    # camera evidence: Exp(d_ec) about t_ec applied to truth
    Ec = Rotation.from_rotvec(d_ec).as_matrix()
    u_est = (u_true - t_ec) @ Ec.T + t_ec
    # lidar evidence: M(d_le) about c0 = G t_le applied to truth
    Rg, tg = G[:3, :3], G[:3, 3]
    c0 = Rg @ t_le + tg
    M = Rg @ Rotation.from_rotvec(d_le).as_matrix() @ Rg.T
    p_est = pts_true @ M.T + (np.eye(3) - M) @ c0
    # fit the evidence plane where the pipeline fits it: in scene-ref
    # ego (PBAResidual converts back to global via T_ge_ref)
    p_est_e = (p_est - G[:3, 3]) @ G[:3, :3]
    c = p_est_e.mean(0)
    _, _, vh = np.linalg.svd(p_est_e - c, full_matrices=False)
    n_hat = vh[-1]; d_hat = -float(n_hat @ c)
    pb = PBAResidual(u_est, np.zeros(400, np.int32), np.ones(400),
                     [(n_hat, d_hat)], G, t_le, t_ec, T_ge_c,
                     np.eye(3), np.eye(3))
    r_true = pb.raw(np.concatenate([-d_le, -d_ec]))[0]
    r_zero = pb.raw(np.zeros(6))[0]
    assert np.abs(r_true).max() < 1e-6, \
        f'algebra broken (x=-d must zero): {r_true[:5]}'
    assert np.abs(r_zero).mean() > 0.01, 'no signal at zero'
    # the extrinsic-recovery state must also be the residual minimizer
    R_le_r, R_ec_r = apply_deltas(np.eye(3), np.eye(3),
                                  np.concatenate([-d_le, -d_ec]))
    assert (np.linalg.norm(R_le_r - Rotation.from_rotvec(-d_le).as_matrix())
            < 1e-12), 'apply_deltas sign drift'
    return float(np.abs(r_true).max()), float(np.abs(r_zero).mean())
