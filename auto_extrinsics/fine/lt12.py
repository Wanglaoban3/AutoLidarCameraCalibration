# -*- coding: utf-8 -*-
"""Levinson & Thrun (RSS 2012) depth-edge / image-edge correlation objective.

Port of the core objective of "Automatic Online Calibration of Cameras and
Lasers": the range sensor's depth-DISCONTINUITY points must land on
photometric edges in the image. Per frame, over the projected pool of
laser points,

    rho_f = (1/n_f) * sum_i az[i] * bz[i]        (over in-frame i)

with az the FROZEN laser-edge indicator (1 = depth-discontinuity point,
0 otherwise) z-scored ONCE over the pool, and bz the image gradient
magnitude z-scored ONCE over the whole image (the paper normalizes each
edge map to zero mean / unit variance once, then correlates over the
overlap; both normalizations are therefore POSE-INDEPENDENT -- the only
pose dependence is where the points project). The pooled score is the
point-count-weighted mean of per-frame values over the split being
evaluated, i.e. one correlation over the POOLED in-frame point set
(never a subset filtered by "currently near an edge" -- that kills the
basin).

Normalization choice (documented per protocol): image-level z-score of
the Gaussian-blurred gradient map + pool-level z-score of the frozen
indicator. An earlier variant re-z-scored both signals over the
in-frame samples at the CURRENT pose; that makes the baseline
pose-adaptive and the optimizer can raise the score by rotating
textured regions out of frame (observed: train corr 0.004 -> 0.062 at a
0.57 deg drift while holdout degraded -- rejected by the gate at GT,
and at real_coarse the same pull moved poses AWAY from truth). The
Gaussian smoothing (sigma 2 px) is the paper's own smoothing of both
edge maps, and it widens the photometric ridge so points a few px off
the contour (6 px evidence cells) still register.

Deviations from the paper, forced by this data/repo (documented):
  - The paper works on an organized dense range image; HDL-32E sweeps
    arrive unorganized, so discontinuity points come from
    auto_extrinsics.fine.evidence.depth_edge_samples_dense on the
    stacked cloud, NEAR side (the foreground point owns the photometric
    silhouette; measured in this repo: far-side points sit ~17 px from
    any strong edge, which zeroes the correlation at GT), extracted
    ONCE at the base pose.
  - The paper correlates the full overlap of two dense images; here the
    pool is the frozen edge set UNION a fixed 20k random subsample
    (seed 0) drawn from the points that project IN-FRAME at the base
    pose (single CAM_FRONT sees ~10% of the 360-deg cloud; a blind draw
    would spend 90% of the pool out of frame). Frozen base-pose
    evidence, same discipline as the edge set.
  - Gradient ascent (paper) -> bounded Nelder-Mead in a +-1.5 deg box
    per DoF (protocol), 2-stage (coarse simplex then polish).
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from auto_extrinsics.fine import evidence as ev
from auto_extrinsics.fine.solver import apply_deltas

BOX_DEG = 1.5           # +- box per DoF (protocol)
EDGE_CAP = 2500         # depth-edge samples per frame (evidence default)
POOL_RANDOM = 20000     # random pool points per frame (protocol: ~20k, seed 0)
MIN_POINTS = 32         # in-frame points below which a frame contributes 0
VALID_FRAC_MIN = 0.8    # gate: trial valid fraction >= 0.8 x init
BOUND_MARGIN_DEG = 0.01  # gate: a DoF within this of the box counts as bound-hit
GATE_EPS = 1e-4         # gate: holdout corr must improve by more than this
BLUR_SIGMA = 2.0        # px, Gaussian smoothing of the gradient map (paper)


def edge_map_z(frame):
    """Pose-independent camera-edge field: Gaussian-blurred Sobel magnitude,
    ROBUSTLY z-scored over the whole image (computed ONCE per frame).

    Choice documented per protocol: median/MAD centering + clipping
    instead of mean/std. The Sobel magnitude is heavy-tailed -- on this
    log the upper image half is textured brick facades whose outliers
    inflate the mean/std ~10x, pressing everything else toward 0; per
    frame the depth-edge points then sit AT the image-average gradient
    (measured: holdout pooled corr at true GT was NEGATIVE and fell
    further toward GT along the init->GT path, i.e. half the frames
    voted the wrong way and no common-mode basin can exist). Median/MAD
    centers the field on the dominant smooth background (road), so
    depth-edge points at the correct pose read clearly positive, and the
    clip bounds the facade outliers so the score cannot be raised by
    rotating textured image regions into the pool."""
    g = gaussian_filter(frame.gm, BLUR_SIGMA)
    med = float(np.median(g))
    mad = 1.4826 * float(np.median(np.abs(g - med)))
    return np.clip((g - med) / max(mad, 1e-9), -3.0, 5.0)


def build_pool(frame, edge, R_ec, R_le, n_random=POOL_RANDOM, seed=0):
    """Fixed correlation pool for one frame: ~20k random stacked-cloud
    indices drawn from the IN-FRAME-at-base-pose points (seed 0), UNION
    the frozen depth-edge indices. Returns (idx [m] into stack_p,
    az [m]) with az the pool-z-scored frozen indicator."""
    rng = np.random.default_rng(seed)
    _, _, ok_base = frame.project_stacked(R_ec, R_le)
    cand = np.flatnonzero(ok_base)
    rand = rng.choice(cand, size=min(n_random, len(cand)), replace=False)
    idx = np.union1d(rand, np.asarray(edge['p_idx'], np.int64))
    ind = np.zeros(len(idx), np.float64)
    pos = np.searchsorted(idx, np.asarray(edge['p_idx'], np.int64))
    ind[pos] = 1.0
    az = (ind - ind.mean()) / (ind.std() + 1e-9)
    return idx, az


def frame_corr(frame, R_ec, R_le, idx, az, bz_map):
    """(pooled-product, n_valid, n_pool) for one frame at one pose: mean
    product of the two PRE-NORMALIZED fields over the frame's in-frame
    pool points."""
    uv, z, ok = frame.project_stacked(R_ec, R_le, idx=idx)
    n_valid = int(ok.sum())
    if n_valid < MIN_POINTS:
        return 0.0, n_valid, len(idx)
    b = ev.bilinear(bz_map, uv[ok, 0], uv[ok, 1])
    return float(az[ok] @ b) / n_valid, n_valid, len(idx)


def pooled(frames, R_le, R_ec, pools):
    """Pooled score over the given frames: n-weighted mean of per-frame
    mean-products (one correlation over the pooled in-frame set), plus
    the pooled valid fraction. Computed over the FULL pool -- never a
    subset selected by current edge proximity."""
    num = 0.0
    n_valid = 0
    n_all = 0
    for fg, (idx, az) in zip(frames, pools):
        c, k, na = frame_corr(fg, R_ec, R_le, idx, az, fg.gm_z)
        num += c * k
        n_valid += k
        n_all += na
    return num / max(n_valid, 1), n_valid / max(n_all, 1)


def make_objective(frames, pools):
    """Negated pooled train score as f(x_deg [6]) for Nelder-Mead."""

    def fn(x_deg):
        R_le, R_ec = apply_deltas(frames[0].R_le0, frames[0].R_ec0,
                                  np.radians(np.asarray(x_deg, float)))
        c, _ = pooled(frames, R_le, R_ec, pools)
        return -c
    return fn


def _initial_simplex(x0, step_deg):
    """Explicit simplex (scipy's default uses 5% of a zero start = ~0)."""
    simplex = [np.array(x0, float)]
    for i in range(len(x0)):
        p = np.array(x0, float)
        p[i] += step_deg
        simplex.append(p)
    return np.array(simplex)


def optimize(fn, box_deg=BOX_DEG, coarse_step_deg=0.4, maxfev=(700, 400)):
    """2-stage bounded Nelder-Mead over x in DEGREES (6 DoF). Returns x."""
    bounds = [(-box_deg, box_deg)] * 6
    r1 = minimize(fn, np.zeros(6), method='Nelder-Mead', bounds=bounds,
                  options=dict(initial_simplex=_initial_simplex(
                      np.zeros(6), coarse_step_deg),
                      maxfev=maxfev[0], xatol=1e-3, fatol=1e-7,
                      adaptive=True))
    r2 = minimize(fn, r1.x, method='Nelder-Mead', bounds=bounds,
                  options=dict(maxfev=maxfev[1], xatol=2e-4, fatol=1e-8,
                               adaptive=True))
    return np.asarray(r2.x, float)


def publish_gate(frames_hold, pools_hold, R_le0, R_ec0, x_deg):
    """AutoLidarCameraCalibration publish discipline on the holdout frames:
    (a) pooled holdout correlation improves by more than GATE_EPS AND a
    majority (median) of the individual holdout frames improve by more
    than GATE_EPS, (b) pooled valid fraction >= 0.8 x init (no herding
    points out of frame), (c) no DoF sits on the +-1.5 deg box.

    The per-frame majority is a documented strengthening: train and
    holdout frames of ONE log share the scene, so a scene-level texture
    bias (the optimizer sliding the rig until frozen edge points sit on
    building facades) shifts the POOLED holdout correlation on both
    splits alike -- observed: a publish with +0.0038 pooled holdout
    while both sensors moved ~0.15 deg AWAY from GT. A genuine
    mounting-rotation correction reduces the same rigid error in every
    frame, so it must improve most frames individually. Returns
    (accepted, metrics dict)."""
    x_deg = np.asarray(x_deg, float)
    R_le_t, R_ec_t = apply_deltas(R_le0, R_ec0, np.radians(x_deg))
    c0, vf0 = pooled(frames_hold, R_le0, R_ec0, pools_hold)
    c1, vf1 = pooled(frames_hold, R_le_t, R_ec_t, pools_hold)
    d_frames = []
    for fg, pool in zip(frames_hold, pools_hold):
        b0, _, _ = frame_corr(fg, R_ec0, R_le0, pool[0], pool[1], fg.gm_z)
        b1, _, _ = frame_corr(fg, R_ec_t, R_le_t, pool[0], pool[1], fg.gm_z)
        d_frames.append(b1 - b0)
    med_delta = float(np.median(d_frames))
    on_bound = bool(np.any(np.abs(x_deg) > BOX_DEG - BOUND_MARGIN_DEG))
    metrics = dict(corr_init=c0, corr_final=c1,
                   valid_frac_init=vf0, valid_frac_final=vf1,
                   holdout_frame_delta=[float(d) for d in d_frames],
                   holdout_median_delta=med_delta,
                   on_bound=on_bound,
                   improved=bool(c1 > c0 + GATE_EPS
                                 and med_delta > GATE_EPS),
                   vf_ok=bool(vf1 >= VALID_FRAC_MIN * vf0))
    accepted = bool(metrics['improved'] and metrics['vf_ok']
                    and not on_bound)
    return accepted, metrics
