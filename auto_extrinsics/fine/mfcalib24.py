# -*- coding: utf-8 -*-
"""MFCalib (IROS 2024, arXiv:2409.00992) port: beam-divergence-inflated
depth-edge point-to-line objective.

PAPER MODEL (arXiv:2409.00992, beam model + Eq. 4-8)
  A LiDAR return at a depth discontinuity carries a systematic
  "edge inflation" bias: the beam footprint (radius e_i grown over range
  by the divergence angle theta) straddles the edge and the measured
  range mixes the two surfaces, so the measured point sits up to one
  beam radius off the true edge, E_i = e_i * V, a biased Gaussian with
  mean in [0, e_i]. The observation model removes the bias before
  projection and the residual is the SIGNED point-to-line distance to
  the local photometric edge line (q_i, n_i):

      z_i = n_i^T ( pi( T_LC (P_i - E_i) ) - q_i )              (Eq. 8)

  (q_i, n_i) come from the kappa nearest image Canny edge pixels
  (KD-tree, kappa = 5 -- reference buildVPnp / calcDirection).

PORT (faithful; every deviation documented):
  - e_i = r_i * tan(theta_div) / cos(alpha_i): range r_i, HDL-32E spec
    divergence theta_div = 1.29 mrad (the paper's Livox Avia constant
    does not transfer to this sensor), incidence angle alpha_i between
    beam and the local surface normal (PCA on the stacked cloud in the
    lidar frame -- pose-independent fixed evidence). The identifiable
    bias direction is beam-radial (range inflation); the correction
    subtracts e_i along the beam BEFORE projection, and the zero-mean
    tangential part survives as the per-point inflation corridor
    sigma_i = f * e_i / d_i (the paper's inflated Gaussian width).
  - Correspondence: kappa = 5 nearest Canny pixels (5x5 Gaussian,
    Canny(20, 60, L2), external contours >= 100 px kept whole --
    reference config Canny.gray_threshold/len_threshold); a match
    requires ALL kappa neighbors within the stage threshold
    (buildVPnp dis_check); line direction = PCA principal axis
    (calcDirection); residual = perpendicular component.
  - Stage schedule: dis_threshold 20 -> 2 px step 1 (reference main
    loop), re-association before every solve; the reference runs two
    solves per threshold, we run the second only if the first moved the
    state (runtime budget; logged per stage).
  - Rotation-only state x = [rotvec_le(3), rotvec_ec(3)] in radians,
    Exp left-multiplied (solver.apply_deltas): this rig's mounting
    error is rotation-dominant and the coarse stage fixes translation.
  - Evidence: depth-discontinuity samples of the DENSE stacked cloud
    (ev.depth_edge_samples_dense) instead of the reference's
    sphere-image Canny + min-depth tags -- same physical evidence
    (occluding contours, near or far side selectable), already densified
    (0.5 s of sweeps) per this repo's data layer.

Annotation-free: no GT boxes/poses are used as evidence; the
project_stacked dynamic-mask placeholder stays None.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from auto_extrinsics.fine.geometry import DEPTH_MIN, DEPTH_MAX
from auto_extrinsics.fine.solver import apply_deltas

THETA_DIV = 1.29e-3      # rad, Velodyne HDL-32E beam divergence (spec)
KAPPA = 5                # paper/reference neighbors defining the line
GRAY_THR = 20            # reference config: Canny.gray_threshold
LEN_THR = 100            # reference config: Canny.len_threshold
COS_MIN = 0.25           # incidence elongation clip (alpha <= 75.5 deg)
MAX_OFFSET_M = 1.5       # safety cap on the inflation magnitude
OFFSET_PULL = 1.0        # +1: subtract e_i along the beam (range-inflation
                         #     correction, pulls the point to the sensor)
STAGE_BOUNDS_DEG = 0.8   # per-DoF trust bound on the rotation state


# ---- image edges (reference edgeDetector) ---------------------------------
def canny_edge_pixels(img, gray_thr=GRAY_THR, len_thr=LEN_THR):
    """Reference edgeDetector: 5x5 Gaussian, Canny(thr, 3*thr, 3, L2),
    external contours kept whole when their length >= len_thr.
    Returns (uv [n,2] float64 xy, n_canny_raw, n_contours_kept)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, gray_thr, gray_thr * 3, 3, L2gradient=True)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    keep = np.zeros_like(canny)
    n_kept = 0
    for c in contours:
        if len(c) >= len_thr:
            pts = c.reshape(-1, 2)
            keep[pts[:, 1], pts[:, 0]] = 255
            n_kept += 1
    ys, xs = np.nonzero(keep)
    uv = np.stack([xs, ys], axis=1).astype(np.float64)
    return uv, int((canny > 0).sum()), n_kept


# ---- fixed per-frame evidence ----------------------------------------------
def frame_context(fg, samples, k_normal=12):
    """Pose-independent evidence for the sampled stacked points: 3D beam
    direction, range, incidence cosine and the divergence-inflation
    magnitude e_i (meters) and corridor sigma_i (px), all in the LIDAR
    frame / at the extraction depth (normals never re-fitted)."""
    idx = samples['p_idx']
    p = fg.stack_p[idx]
    tree = cKDTree(fg.stack_p)
    nbr = tree.query(p, k=k_normal, workers=-1)[1]
    nbrs = fg.stack_p[nbr]                      # (n, k, 3)
    mean = nbrs.mean(axis=1, keepdims=True)
    _, sv, vh = np.linalg.svd(nbrs - mean, full_matrices=False)
    normal = vh[:, 2, :]                        # smallest-variance axis
    r = np.linalg.norm(p, axis=1)
    u = p / np.maximum(r[:, None], 1e-9)
    cos_a = np.abs(np.sum(normal * u, axis=1))
    e_m = np.minimum(r * np.tan(THETA_DIV) / np.clip(cos_a, COS_MIN, None),
                     MAX_OFFSET_M)
    uv, z, ok = fg.project_stacked(fg.R_ec0, fg.R_le0, idx=idx)
    f_px = float(fg.K[0, 0])
    sigma_px = f_px * e_m / np.maximum(z, 0.5)
    return dict(p_l=p, u_hat=u, r=r, cos_a=cos_a, e_m=e_m,
                sigma_px=sigma_px, f_px=f_px)


# ---- projection (mirror of FrameGeom.project_stacked for offset points) ----
def project_points(fg, p_l, idx, R_ec, R_le):
    """project_stacked chain applied to (possibly beam-corrected) points;
    geometry.py is shared and must not grow method-specific entries."""
    sid = fg.stack_sid[idx]
    p_s = (R_le @ p_l.T).T + fg.t_le
    p_e = np.empty_like(p_s)
    for k, G in enumerate(fg.stack_G):
        mk = sid == k
        if mk.any():
            p_e[mk] = p_s[mk] @ G[:3, :3].T + G[:3, 3]
    p_c = (p_e - fg.t_ec) @ R_ec.T
    z = p_c[:, 2]
    uv = (fg.K @ p_c.T).T[:, :2] / np.maximum(z[:, None], 1e-9)
    h, w = fg.shape
    ok = ((z > DEPTH_MIN) & (z < DEPTH_MAX)
          & (uv[:, 0] > 0.5) & (uv[:, 0] < w - 1.5)
          & (uv[:, 1] > 0.5) & (uv[:, 1] < h - 1.5))
    return uv, z, ok


def point_to_line(fg, ctx, samples, R_ec, R_le, edge_tree, edge_uv,
                  kappa=KAPPA, gate_tau=None, offset_sign=0.0):
    """Paper Eq. 8 for one frame at one pose.

    offset_sign != 0 applies the divergence correction
    P_i - offset_sign * e_i * u_i before projection. gate_tau (px) is
    the stage association: all kappa nearest edge pixels within tau.
    Returns full-length arrays (n_samples): z (signed px, NaN off-frame)
    and matched (bool)."""
    idx = samples['p_idx']
    p_l = ctx['p_l']
    if offset_sign:
        p_l = p_l - offset_sign * ctx['e_m'][:, None] * ctx['u_hat']
    uv, z, ok = project_points(fg, p_l, idx, R_ec, R_le)
    n = len(idx)
    zout = np.full(n, np.nan)
    matched = np.zeros(n, bool)
    v = np.nonzero(ok)[0]
    if len(v) < kappa:
        return zout, matched
    d, nn = edge_tree.query(uv[v], k=kappa, workers=-1)
    q = edge_uv[nn]                             # (m, k, 2)
    qm = q.mean(axis=1, keepdims=True)
    _, sv, vt = np.linalg.svd(q - qm, full_matrices=False)
    dvec = vt[:, 0, :]                          # calcDirection
    nrm = np.stack([-dvec[:, 1], dvec[:, 0]], axis=1)
    diff = uv[v] - q[:, 0, :]                   # q_i = nearest edge pixel
    zv = np.sum(diff * nrm, axis=1)
    zout[v] = zv
    matched[v] = (d <= gate_tau).all(axis=1) if gate_tau is not None \
        else (d[:, -1] < np.inf)
    return zout, matched


# ---- discriminability statistics (falsification test) ----------------------
def pose_stats(frames, ctxs, samples, edges, R_ec, R_le,
               offset_sign=0.0, gate_tau=None):
    """Pooled point-to-line statistics over all frames at one pose, on the
    FIXED sample set (no pose-adaptive association gate): the honest
    landscape measure for the falsification test. offset_sign applies
    the radial divergence correction before projection. gate_tau adds
    the reference matched-set statistics (buildVPnp dis_check)."""
    absz, sig, whit, mabs, n_in, n_m = [], [], [], [], 0, 0
    for fg, ctx, s, (uv, tree) in zip(frames, ctxs, samples, edges):
        z, matched = point_to_line(fg, ctx, s, R_ec, R_le, tree, uv,
                                   gate_tau=gate_tau, offset_sign=offset_sign)
        v = np.isfinite(z)
        a = np.abs(z[v])
        absz.append(a)
        g = ctx['sigma_px'][v]
        sig.append(g)
        whit.append(np.minimum(a, 20.0) / np.maximum(g, 0.5))
        n_in += int(v.sum())
        n_m += int(matched[v].sum())
        mabs.append(a[matched[v]])
    a = np.concatenate(absz)
    g = np.concatenate(sig)
    m = np.concatenate(mabs) if n_m else np.array([np.nan])
    return dict(n=int(len(a)),
                med=float(np.median(a)),
                med20=float(np.median(np.minimum(a, 20.0))),
                frac5=float((a <= 5.0).mean()),
                frac10=float((a <= 10.0).mean()),
                frac_sig=float((a <= g).mean()),
                med_sig=float(np.median(g)),
                med_whitened=float(np.median(np.concatenate(whit))),
                frac_match=n_m / max(n_in, 1),
                med_match=float(np.median(m)))


# ---- stage objective + published schedule ----------------------------------
def make_residual_fn(frames, ctxs, samples, edges, base_R_le, base_R_ec,
                     tau, offset_sign):
    """Whitened signed point-to-line residuals of matched points, pooled
    over frames; association frozen per call (rebuilt between solves,
    reference semantics)."""
    def fn(x):
        R_le, R_ec = apply_deltas(base_R_le, base_R_ec, x)
        rs = []
        for fg, ctx, s, (uv, tree) in zip(frames, ctxs, samples, edges):
            z, matched = point_to_line(fg, ctx, s, R_ec, R_le, tree, uv,
                                       gate_tau=tau,
                                       offset_sign=offset_sign)
            sig = ctx['sigma_px']
            rs.append(np.where(matched & np.isfinite(z), z / sig, 0.0))
        return np.concatenate(rs)
    return fn


def match_count(frames, ctxs, samples, edges, base_R_le, base_R_ec, x,
                tau, offset_sign):
    R_le, R_ec = apply_deltas(base_R_le, base_R_ec, x)
    n = 0
    for fg, ctx, s, (uv, tree) in zip(frames, ctxs, samples, edges):
        _, matched = point_to_line(fg, ctx, s, R_ec, R_le, tree, uv,
                                   gate_tau=tau, offset_sign=offset_sign)
        n += int(matched.sum())
    return n


def run_schedule(frames, ctxs, samples, edges, base_R_le, base_R_ec, x0,
                 offset_sign=OFFSET_PULL, thresholds=None, max_nfev=40,
                 verbose=True):
    """Reference main loop: dis_threshold 20 -> 2 step 1, solve per
    threshold with fresh association; second solve per threshold only if
    the first moved the state (documented deviation). Bounds are the
    rotation-only trust region; every stage is logged."""
    if thresholds is None:
        thresholds = list(range(20, 1, -1))
    x = np.asarray(x0, float).copy()
    lo = x0 - np.radians(STAGE_BOUNDS_DEG)
    up = x0 + np.radians(STAGE_BOUNDS_DEG)
    history = []
    for tau in thresholds:
        for attempt in (1, 2):
            n_match = match_count(frames, ctxs, samples, edges,
                                  base_R_le, base_R_ec, x, tau,
                                  offset_sign)
            if n_match < 12:
                if verbose:
                    print(f'  stage tau={tau:2d}: only {n_match} matches '
                          '-- skip')
                history.append(dict(tau=tau, attempt=attempt, n_match=n_match,
                                    skipped=True))
                break
            fn = make_residual_fn(frames, ctxs, samples, edges,
                                  base_R_le, base_R_ec, tau, offset_sign)
            c0 = float(np.sqrt(np.sum(fn(x) ** 2)))
            res = least_squares(fn, x, bounds=(lo, up), loss='soft_l1',
                                f_scale=1.0, max_nfev=max_nfev)
            dx = float(np.linalg.norm(res.x - x))
            x = res.x
            c1 = float(np.sqrt(np.sum(fn(x) ** 2)))
            if verbose:
                print(f'  stage tau={tau:2d} a{attempt}: match {n_match} '
                      f'cost {c0:.1f}->{c1:.1f} |dx|={np.degrees(dx):.3f} deg'
                      + (' BOUND' if bool(np.any(np.isclose(
                          x, lo, atol=1e-6))
                          or np.any(np.isclose(x, up, atol=1e-6))) else ''))
            history.append(dict(tau=tau, attempt=attempt, n_match=n_match,
                                cost0=c0, cost1=c1, dx_deg=np.degrees(dx),
                                bound_hit=bool(
                                    np.any(np.isclose(x, lo, atol=1e-6))
                                    or np.any(np.isclose(x, up,
                                                         atol=1e-6))),
                                skipped=False))
            if dx < 1e-6:
                break
    return x, history


# ---- holdout publish gate ---------------------------------------------------
def holdout_metrics(frames, ctxs, samples, edges, base_R_le, base_R_ec, x,
                    tau, offset_sign):
    """Odd-frame holdout at fresh association (tau-gated matched set):
    median |z| (px, unwhitened) and matched count. Evaluated at the
    initial and final states only -- the publish decision."""
    R_le, R_ec = apply_deltas(base_R_le, base_R_ec, x)
    med, n = [], 0
    for fg, ctx, s, (uv, tree) in zip(frames, ctxs, samples, edges):
        z, matched = point_to_line(fg, ctx, s, R_ec, R_le, tree, uv,
                                   gate_tau=tau, offset_sign=offset_sign)
        m = matched & np.isfinite(z)
        if m.any():
            med.append(np.abs(z[m]))
            n += int(m.sum())
    if not med:
        return None, n
    return float(np.median(np.concatenate(med))), n
