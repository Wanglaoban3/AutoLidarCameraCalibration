# -*- coding: utf-8 -*-
"""GRIL-Calib family port (Roh et al., RA-L 2024 / GRIL-Calib++; arXiv
2312.14374): LiDAR INTENSITY-EDGE points vs the image edge distance field,
robust direct alignment + a light ground-plane rotation prior.

What is ported here
  1. INTENSITY-EDGE selection in the LIDAR domain (the step that makes GRIL
     work): the stacked cloud's calibrated intensity is splatted into a
     virtual PERSPECTIVE intensity image (base-pose camera intrinsics at
     half resolution, ~0.03 rad/px -> ~0.10 m/px at 3 m, 0.33 m/px at 30 m),
     holes from beam sparsity are inpainted, Canny runs on the render, and
     Canny pixels are back-projected through the z-buffer to 3D lidar
     points of the stacked cloud. The selection is PHYSICAL (lidar
     intensity + geometry; the camera image is never read), extracted ONCE
     at the base pose and FROZEN for the whole optimization -- no
     re-location, no image-proximity filtering (points picked for being
     near image edges are near SOME edge at every pose and the basin
     flattens; measured repeatedly in this repo).
  2. Objective: per-point residual = value of the strong-TEED distance
     field (ev.build_dt_map, reused) at the projected point; OUT-OF-FRAME /
     depth-range / dynamic-mask invalid = CONSTANT BARRIER 30.0, never
     dropped (reference semantics: a valid-only metric lets the optimizer
     herd points out of frame and read the leftovers as improvement).
  3. Priors: a WEAK quadratic pull keeping the measured stacked-cloud
     ground normal (frozen, ego frame, fit as in ev.marking_points) fixed
     under the current lidar rotation -- self-consistent (constraint is
     R_le @ n_l == R_le0 @ n_l with n_l frozen, so the road-crown bias of
     the measured normal cancels; it regularizes, it does not anchor
     roll/pitch to gravity) -- plus the reference's 0.08-per-DoF prior
     toward the init.
  4. scipy least_squares over x = [rotvec_le, rotvec_ec] (rad, Exp(x)
     left-multiplied via solver.apply_deltas), huber f_scale 2.5,
     bounds +-1.5 deg, numeric Jacobian, max_nfev 100.

Gate (reference discipline, implemented in the runner): odd-index frames
are holdout; publish only if the holdout median AND p90 of the FULL
residual vector (barrier included) improve and the valid fraction stays
>= 0.8x and no DoF sits on its bound.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from auto_extrinsics.fine import evidence as ev
from auto_extrinsics.fine.geometry import DEPTH_MIN, DEPTH_MAX
from auto_extrinsics.fine.solver import apply_deltas

# ---- virtual intensity render ----------------------------------------------
VIRT_DIV = 2        # virtual render at 1/2 the real camera resolution
CANNY_LO = 40.0     # classic 1:2 thresholds on the 0-255 normalized render
CANNY_HI = 120.0
INPAINT_R = 3       # px, hole fill radius (beam-sparse regions of the render)
BLUR_K = 3          # px, Gaussian smoothing before Canny
INT_PCT = (2.0, 98.0)   # robust intensity normalization percentiles
MAX_EDGE_PTS = 1200     # per frame, uniform stride over the edge-pixel set
SUP_CELL = 8            # virtual px, spatial decorrelation of edge pixels

# ---- objective --------------------------------------------------------------
BARRIER = 30.0          # residual for out-of-frame / invalid points
PRIOR_W = 0.08          # per-DoF prior toward the init (reference value)
BOUND_DEG = 1.5         # optimizer box, per DoF
GROUND_W = 0.05         # ground-normal prior weight (x focal): 1 deg of
                        # ground-normal tilt costs ~1.1 residual units --
                        # orders weaker than the data term, a soft
                        # regularizer, deliberately not an anchor


# ---- frozen evidence: intensity-edge points of the stacked cloud -------------
def extract_intensity_edges(fg, res_div=VIRT_DIV, canny_lo=CANNY_LO,
                            canny_hi=CANNY_HI, max_points=MAX_EDGE_PTS,
                            sup_cell=SUP_CELL, range_max=None,
                            hole_guard=0):
    """Canny-edge pixels of the virtual intensity render, back-projected to
    stacked-cloud point indices. Runs at the BASE pose; the returned ids are
    FROZEN evidence (pose-independent: they index physical lidar returns
    selected by lidar intensity only).

    range_max: optional camera-frame depth cap on the RENDERED points (m).
    Sparse far-field returns smear the render and Canny fires on inpainted
    noise there; capping the render is a physical, image-blind selection.

    hole_guard: Canny pixels with an unfilled (no-return) pixel within this
    radius are discarded -- their contrast comes from inpainting fill, not
    measured intensity.

    Returns dict(ids [m] into stack_p, n_splat, n_canny, n_kept)."""
    p_e = fg.stacked_ego_ref()
    # base ego(cam time) -> camera: invert the base cam extrinsics only
    # (stacked_ego_ref already lives in the reference ego frame)
    p_c = (p_e - fg.t_ec) @ fg.R_ec0.T
    z = p_c[:, 2]
    uv = (fg.K @ p_c.T).T[:, :2] / np.maximum(z[:, None], 1e-9)
    hv, wv = fg.shape[0] // res_div, fg.shape[1] // res_div
    u = uv[:, 0] / res_div
    v = uv[:, 1] / res_div
    good = (z > DEPTH_MIN) & (z < DEPTH_MAX) \
        & (u >= 0) & (u < wv - 1) & (v >= 0) & (v < hv - 1)
    if range_max is not None:
        good &= z < range_max
    idx = np.flatnonzero(good)
    out = dict(ids=np.array([], np.int64), n_splat=int(len(idx)),
               n_canny=0, n_kept=0)
    if len(idx) < 100:
        return out
    xi = np.floor(u[idx]).astype(np.int64)
    yi = np.floor(v[idx]).astype(np.int64)
    pix = yi * wv + xi
    order = np.argsort(z[idx])                 # near points written last
    pix_o, idx_o = pix[order], idx[order]
    ibuf = np.full(wv * hv, -1, np.int64)
    ibuf[pix_o] = idx_o                        # duplicate px: nearest wins
    img = np.zeros(wv * hv, np.uint8)
    lo_i, hi_i = np.percentile(fg.stack_intensity[idx], INT_PCT)
    val = np.clip((fg.stack_intensity[idx_o] - lo_i)
                  / max(hi_i - lo_i, 1e-6), 0.0, 1.0)
    img[pix_o] = (val * 255.0).astype(np.uint8)
    img = img.reshape(hv, wv)
    hole = (ibuf < 0).reshape(hv, wv).astype(np.uint8) * 255
    img = cv2.inpaint(img, hole, INPAINT_R, cv2.INPAINT_TELEA)
    img = cv2.GaussianBlur(img, (BLUR_K, BLUR_K), 0)
    edges = cv2.Canny(img, canny_lo, canny_hi) > 0
    if hole_guard > 0:
        near_hole = cv2.dilate(hole, np.ones(
            (2 * hole_guard + 1, 2 * hole_guard + 1), np.uint8)) > 0
        edges &= ~near_hole
    ids = ibuf.reshape(hv, wv)[edges]
    ids = ids[ids >= 0]
    out['n_canny'] = int(len(ids))
    if len(ids) == 0:
        return out
    # decorrelate: one point per sup_cell virtual cell (adjacent Canny
    # pixels are near-duplicate measurements of one physical edge), then a
    # uniform stride cap -- selection stays image-blind end to end
    ey, ex = np.nonzero(edges)
    sel2 = np.arange(len(ids))
    key = (ey // sup_cell) * 4096 + (ex // sup_cell)
    _, u_idx = np.unique(key[sel2], return_index=True)
    ids = ids[u_idx]
    if len(ids) > max_points:
        ids = ids[::int(np.ceil(len(ids) / max_points))].copy()
    out['ids'] = ids.astype(np.int64)
    out['n_kept'] = int(len(ids))
    return out


def inframe_stats(fg, ids, R_ec, R_le):
    """(n_valid, fraction) of edge points passing the base projection gates
    (depth range + in real-camera frustum + not on the dynamic mask)."""
    if len(ids) == 0:
        return 0, 0.0
    _, _, ok = fg.project_stacked(R_ec, R_le, idx=ids)
    return int(ok.sum()), float(ok.mean())


# ---- objective ---------------------------------------------------------------
def frame_residuals(fg, ids, R_ec, R_le):
    """DT values at the projected edge points; invalid -> BARRIER (kept)."""
    r = np.full(len(ids), BARRIER)
    if len(ids) == 0:
        return r
    uv, z, ok = fg.project_stacked(R_ec, R_le, idx=ids)
    if ok.any():
        d = ev.bilinear(fg.dt_map, uv[ok, 0], uv[ok, 1])
        r[ok] = np.where(np.isfinite(d), d, BARRIER)
    return r


def score_pose(frames_ids, R_le, R_ec):
    """(median, p90, valid fraction) of the FULL residual vector over the
    given (frame, ids) pairs -- barrier included, reference score()
    semantics; the valid fraction guards the degenerate all-barrier pose."""
    vals, n_valid, n_all = [], 0, 0
    for fg, ids in frames_ids:
        r = frame_residuals(fg, ids, R_ec, R_le)
        vals.append(r)
        n_valid += int((r < BARRIER).sum())
        n_all += len(r)
    v = np.concatenate(vals) if vals else np.array([BARRIER])
    return (float(np.median(v)), float(np.percentile(v, 90)),
            n_valid / max(n_all, 1))


# ---- priors --------------------------------------------------------------------
def ground_normal_ego(frames, tol=0.10):
    """Unit ground normal (reference ego frame) of the stacked cloud, fit
    exactly as ev.marking_points fits its plane: forward band -> SVD ->
    inlier refit; unit-average over frames. Frozen evidence."""
    normals = []
    for fg in frames:
        p_e = fg.stacked_ego_ref()
        band = ((p_e[:, 0] > 4.0) & (p_e[:, 0] < 30.0)
                & (np.abs(p_e[:, 1]) < 25.0)
                & (p_e[:, 2] > -1.5) & (p_e[:, 2] < 0.5))
        if int(band.sum()) < 100:
            continue
        C = p_e[band]
        c = np.median(C, axis=0)
        _, _, vh = np.linalg.svd(C - c, full_matrices=False)
        n = vh[-1]
        inl = band & (np.abs((p_e - c) @ n) < tol)
        if int(inl.sum()) < 100:
            continue
        C = p_e[inl]
        c = np.median(C, axis=0)
        _, _, vh = np.linalg.svd(C - c, full_matrices=False)
        n = vh[-1]
        if n[2] < 0:
            n = -n
        normals.append(n)
    if not normals:
        return None
    n = np.mean(normals, axis=0)
    n /= np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    return n


def ground_prior(x, R_le0, n_e0, w_px):
    """Residuals penalizing lidar rotation that TILTS the frozen measured
    ground plane. With n_l = R_le0^T n_e0 frozen, the constraint is
    (R_le @ n_l) == n_e0 == Exp(x_le) @ n_e0: self-consistent (both sides
    use the SAME measured plane, so a crown-biased normal cannot inject a
    bias), and rotation about the normal itself is unpenalized (projected
    out). ~2 effective DoF = lidar roll/pitch w.r.t. the road plane."""
    d = Rotation.from_rotvec(x[:3]).as_matrix() @ R_le0 @ n_e0 - n_e0
    d = d - (d @ n_e0) * n_e0
    return w_px * d


def make_objective(train_pairs, R_le0, R_ec0, n_e0, focal_px,
                   bound_rad=None, prior_w=PRIOR_W, ground_w=GROUND_W):
    """Pooled residual vector for least_squares over x (rad, 6): data terms
    on the train (even-index) frames + init prior + ground prior."""
    bound_rad = np.radians(BOUND_DEG) if bound_rad is None else bound_rad
    w_ground = ground_w * focal_px

    def fn(x):
        R_le, R_ec = apply_deltas(R_le0, R_ec0, x)
        rs = [frame_residuals(fg, ids, R_ec, R_le) for fg, ids in train_pairs]
        rs.append(prior_w * x / bound_rad)
        if n_e0 is not None:
            rs.append(ground_prior(x, R_le0, n_e0, w_ground))
        return np.concatenate(rs)
    return fn
