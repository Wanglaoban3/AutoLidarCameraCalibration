# -*- coding: utf-8 -*-
"""Image evidence: TEED wrapper, bilinear sampling, normal-search residuals.

Residual (both scales): for each depth-edge sample, the SIGNED offset along
its image normal to the strongest gradient-magnitude peak within a search
window (bilinear + quadratic interpolation), gated by the TEED probability
map -- the pattern proven in extrinsic_recovery/camera_vp_calib
.subpixel_refine, lifted to cross-modal. TEED only SELECTS structure;
localization precision comes from the image gradient.

The window is annealed wide->narrow by the caller: the offset-to-feature
distance is well-behaved at any width, while a blurred-edge-map chamfer
(residual = 1 - DT) has NO basin with sparse samples -- the optimizer can
slide samples onto arbitrary dark ridges and cost keeps dropping (observed:
poses drift 80 deg while cost monotonically decreases). Window annealing
replaces the DT phase; peaks must be interior to the window and not too
close to its rim, otherwise the sample is dropped (weight 0).
"""
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

TEED_PROB_GATE = 0.2
GRAD_HALF_WIN = 4.0     # px, search along the normal
GRAD_STEP = 0.5         # px


# ---- TEED ----------------------------------------------------------------
def load_teed(device):
    import torch
    root = os.environ.get('AUTOEX_TEED_ROOT', r'H:\projects\TEED-main')
    sys.path.insert(0, root)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        from ted import TED
    finally:
        os.chdir(cwd)
    model = TED().to(device)
    model.load_state_dict(torch.load(
        os.path.join(root, 'checkpoints', 'BIPED', '7', '7_model.pth'),
        map_location=device))
    model.eval()
    return model


def teed_prob(model, img_bgr, device):
    """TEED probability map at native resolution (min-max normalized, same
    as camera_vp_calib.teed_edges -- downstream gates are tuned to this)."""
    import torch
    h, w = img_bgr.shape[:2]
    x = img_bgr.astype(np.float32)
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        x = cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x -= np.array([104.007, 116.669, 122.679], dtype=np.float32)
    x = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(device)
    with torch.no_grad():
        fused = torch.sigmoid(model(x)[-1])[0, 0].cpu().numpy()
    prob = (fused - fused.min()) / (np.ptp(fused) + 1e-8)
    return prob[:h, :w]


def gradient_magnitude(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


# ---- sampling --------------------------------------------------------------
def bilinear(img, x, y):
    """Vectorized bilinear sample; out-of-bounds -> 0."""
    h, w = img.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    v = ((1 - fy) * (1 - fx) * img[y0c, x0c]
         + (1 - fy) * fx * img[y0c, x0c + 1]
         + fy * (1 - fx) * img[y0c + 1, x0c]
         + fy * fx * img[y0c + 1, x0c + 1])
    return v * ok


def build_soft_dt(prob, sigma):
    """Gaussian-blurred TEED edge map -- used by verification/visualization
    overlays, NOT by the optimizer (see module docstring)."""
    return gaussian_filter((prob > 0.25).astype(np.float32), sigma)


# ---- normal-search residuals ------------------------------------------------
NEAR_FRAC = 0.6         # 'strong ridge' = >= this fraction of window max

def locate_ridges(frame, samples_uv, normals, half_win, t_max=None):
    """ICP association (fixed per pass): each depth-edge sample locks to the
    NEAREST STRONG ridge along its normal, searched ONE-SIDED in
    [0.5px, min(0.85*half_win, 0.85*t_max)] when t_max is given.

    One-sidedness is the geometric meaning, not a heuristic: the target is
    the photometric edge BETWEEN the sample and its own background point.
    Two-sided search lets samples lock onto their own object's interior
    texture -- at GT calibration that produced a 4-5 px biased offset field
    that the solver absorbed as a spurious 2 deg COMMON-MODE rig rotation
    (zero-noise check: both sensors drifting identically). TEED probability
    at the located ridge gates the association.

    t_max: per-sample px separation to the background point (see
    depth_edge_samples); None = symmetric window (px metrics at a converged
    pose use this).

    Returns (targets [n,2], weights [n]; weight 0 = dropped)."""
    u, v = samples_uv[:, 0], samples_uv[:, 1]
    offs = np.arange(-half_win, half_win + 1e-9, GRAD_STEP)
    pos_u = u[None, :] + normals[None, :, 0] * offs[:, None]
    pos_v = v[None, :] + normals[None, :, 1] * offs[:, None]
    g = bilinear(frame.gm, pos_u, pos_v)                    # (n_off, n_pts)
    n = g.shape[1]
    idx = np.arange(n)
    lo = np.full(n, 0.5 if t_max is not None else -half_win)
    hi = np.full(n, half_win)
    if t_max is not None:
        hi = np.minimum(hi, 0.85 * np.maximum(t_max, 1.0))
    band = (offs[:, None] >= lo[None, :]) & (offs[:, None] <= hi[None, :])
    gmax = np.where(band, g, 0.0).max(axis=0)
    strong = (g >= NEAR_FRAC * gmax[None, :]) & band
    o_abs = np.abs(offs)[:, None]
    k = np.argmin(np.where(strong, o_abs, np.inf), axis=0)
    has_strong = strong.any(axis=0)
    gk = g[k, idx]
    lo_i = np.clip(k - 1, 0, len(offs) - 1)
    hi_i = np.clip(k + 1, 0, len(offs) - 1)
    den = g[lo_i, idx] - 2 * gk + g[hi_i, idx]
    den_safe = np.where(np.abs(den) > 1e-9, den, 1.0)
    delta = np.clip(0.5 * (g[lo_i, idx] - g[hi_i, idx]) / den_safe, -1, 1)
    delta = np.where(np.abs(den) > 1e-9, delta, 0.0)
    t_star = offs[k] + delta * GRAD_STEP
    in_band = (t_star >= lo) & (t_star <= hi)
    t_star = np.where(in_band, t_star, lo)
    pk_u = u + normals[:, 0] * t_star
    pk_v = v + normals[:, 1] * t_star
    prob_pk = bilinear(frame.teed_prob, pk_u, pk_v)
    good = has_strong & (gk >= 30.0) & in_band & (prob_pk >= TEED_PROB_GATE)
    targets = np.stack([pk_u, pk_v], axis=1)
    return targets, np.where(good, prob_pk, 0.0)


def offset_to_targets(samples_uv, normals, targets):
    """Smooth point-to-target distance along the normal (ICP residual)."""
    d = samples_uv - targets
    return d[:, 0] * normals[:, 0] + d[:, 1] * normals[:, 1]


def grad_residuals(frame, samples_uv, normals, half_win=GRAD_HALF_WIN):
    """Measurement-style residual: locate ridges NOW and return the signed
    offset. Used for px-quality metrics at a fixed pose (where assignment is
    unambiguous), not as a solver objective (see locate_ridges)."""
    targets, w = locate_ridges(frame, samples_uv, normals, half_win)
    res = offset_to_targets(samples_uv, normals, targets)
    return res * (w > 0), w


# ---- depth-edge samples -----------------------------------------------------
def depth_edge_samples(frame, p_l, T_lc, max_samples, rng,
                       nbr_radius_px=30.0, jump_thr=0.5, jump_rel=0.06,
                       min_nbrs=3, min_sep_px=3.0):
    """Depth-discontinuity samples under SPARSE beams (~3k points in the
    camera FOV per 32-line sweep).

    Occluding edges are adjacent in ANGLE, far in RANGE, so the neighborhood
    must be searched in the IMAGE plane: a point pair (near, far) with
    depth jump > jump_thr + jump_rel*z within nbr_radius_px marks an edge.
    The sample is the BACKGROUND point CLOSEST to the near point -- its
    projection abuts the photometric contour, so at correct calibration its
    image offset to that contour is ~0 (beam width). Anchoring the
    FOREGROUND point instead bakes the occlusion parallax (tens of px: a
    ground point 10 m vs a wall base 14 m projects ~50 px below the
    contact line) into the target -- a systematic the solver can only
    absorb as a spurious rig rotation (observed as 2-3 deg common-mode
    drift at GT in the zero-noise check).
    The image normal points from the background sample TOWARD the contour
    (i.e. toward the near point) -- what the one-sided association search
    walks along (locate_ridges with t_max).

    Returns dict(p_idx [n] into p_l, normals [n,2], t_max [n] px)."""
    from scipy.spatial import cKDTree
    uv, z, ok = frame.project(p_l, T_lc)
    valid = np.nonzero(ok)[0]
    if len(valid) < 50:
        return dict(p_idx=np.array([], np.int64), normals=np.zeros((0, 2)),
                    t_max=np.zeros(0))
    zv, uvv = z[valid], uv[valid]
    tree = cKDTree(uvv)
    nbr_lists = tree.query_ball_point(uvv, nbr_radius_px)
    e_idx, e_nrm, e_tmax = [], [], []
    for i, nbrs in enumerate(nbr_lists):
        if len(nbrs) < min_nbrs:
            continue
        nb = np.asarray(nbrs)
        dz = zv[nb] - zv[i]
        cand = nb[dz > jump_thr + jump_rel * zv[i]]
        if len(cand) == 0:
            continue
        # background point CLOSEST to the near point = closest to the
        # occluding contour (argmax dz would pick the farthest wall point
        # and maximize parallax)
        seps = np.linalg.norm(uvv[cand] - uvv[i], axis=1)
        j = cand[int(np.argmin(seps))]
        sep = float(seps[int(np.argmin(seps))])
        if not (min_sep_px < sep < 3.0 * nbr_radius_px):
            continue
        n2 = uvv[i] - uvv[j]          # from background sample toward contour
        e_idx.append(valid[j])
        e_nrm.append(n2 / sep)
        e_tmax.append(sep)
    e_idx = np.asarray(e_idx, np.int64)
    e_nrm = np.asarray(e_nrm).reshape(-1, 2)
    e_tmax = np.asarray(e_tmax)
    if len(e_idx) > max_samples:
        keep = rng.choice(len(e_idx), max_samples, replace=False)
        e_idx, e_nrm, e_tmax = e_idx[keep], e_nrm[keep], e_tmax[keep]
    sel_px = np.stack([uv[e_idx, 0], uv[e_idx, 1]], axis=1) \
        if len(e_idx) else np.zeros((0, 2))
    return dict(p_idx=e_idx, normals=e_nrm, t_max=e_tmax,
                pixel=np.round(sel_px).astype(int))


def project_samples(frame, p_l, T_lc, sample):
    """uv of the sampled points (current pose) + validity."""
    p = p_l[sample['p_idx']]
    uv, z, ok = frame.project(p, T_lc)
    return uv, ok


# ---- depth-edge samples on the stacked cloud --------------------------------
def depth_edge_samples_dense(frame, R_ec, R_le, cell_px=6.0, jump_thr=0.5,
                             jump_rel=0.06, max_samples=2500, rng=None,
                             min_sep_px=3.0, near_side=False):
    """Depth-discontinuity samples from the DENSE stacked cloud (~10x the
    key sweep), extracted at the BASE pose (fixed evidence).

    Vectorized z-buffer variant of depth_edge_samples: points are binned
    into cell_px image cells (min depth wins), and a far cell whose depth
    exceeds a neighbor cell's by jump_thr + jump_rel*z marks an occluding
    boundary. The sample is the far cell's point (background side -- see
    depth_edge_samples for why), with the normal toward the near cell.
    Rasterization needs fill, which only the stacked cloud has; running
    this on a single sparse sweep leaves mostly holes.

    near_side=True returns the NEAR cell's point instead (normals flipped):
    the foreground object OWNS the photometric silhouette, so for
    distance-field alignment the near side is the evidence -- far-side
    points sit on bare road/wall behind the contour, ~17 px median from
    any strong TEED edge (measured), which makes the DT objective flat.

    Returns dict(p_idx into the STACKED cloud, normals [n,2], t_max [n])."""
    uv, z, ok = frame.project_stacked(R_ec, R_le)
    valid = np.nonzero(ok & np.isfinite(z))[0]
    if len(valid) < 200:
        return dict(p_idx=np.array([], np.int64), normals=np.zeros((0, 2)),
                    t_max=np.zeros(0))
    h, w = frame.shape
    cw, ch = int(np.ceil(w / cell_px)), int(np.ceil(h / cell_px))
    ci = np.clip((uv[valid, 0] / cell_px).astype(np.int64), 0, cw - 1)
    cj = np.clip((uv[valid, 1] / cell_px).astype(np.int64), 0, ch - 1)
    cell = cj * cw + ci
    zbuf = np.full(cw * ch, np.inf)
    pidx = np.full(cw * ch, -1, np.int64)
    order = np.argsort(z[valid])          # nearest writes last -> wins
    np.minimum.at(zbuf, cell[order], z[valid][order])
    hit = np.isfinite(zbuf)
    # representative point per cell: nearest in-grid point of that cell
    first = np.zeros(cw * ch, np.int64)
    first[cell[order]] = valid[order]     # later (nearer) overwrite
    zbuf2 = zbuf.reshape(ch, cw)

    best_far, best_near = [], []
    # far cell at (r, c) vs NEAR neighbors on rings of 1..3 cells: the
    # baseline (far-to-near px distance) sets the one-sided search reach
    # (band = [0.5, 0.85*t_max]), so it must span the expected POSE error
    # (~9 px per 0.4 deg), not just the cell size -- adjacent-cell-only
    # pairs clip the band below the true contour and associations collapse
    # onto nearer texture (observed: real_coarse drifted 0.36 -> 0.55 deg).
    # shifted[r, c] = zbuf2[r + dr, c + dc]; cells without that neighbor
    # stay inf (their comparison yields NaN -> False).
    with np.errstate(invalid='ignore'):
        for radius in (1, 2, 3):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue          # ring only
                    shifted = np.full_like(zbuf2, np.inf)
                    r0s, r1s = max(0, -dr), min(ch, ch - dr)
                    c0s, c1s = max(0, -dc), min(cw, cw - dc)
                    if r1s > r0s and c1s > c0s:
                        shifted[r0s:r1s, c0s:c1s] = \
                            zbuf2[r0s + dr:r1s + dr,
                                  c0s + dc:c1s + dc]
                    jump = (np.isfinite(zbuf2) & np.isfinite(shifted)
                            & ((zbuf2 - shifted)
                               > (jump_thr + jump_rel * shifted)))
                    rr, cc = np.nonzero(jump)
                    best_far.append(rr * cw + cc)
                    best_near.append((rr + dr) * cw + (cc + dc))
    far_cells = np.concatenate(best_far)
    near_cells = np.concatenate(best_near)
    m = hit[far_cells] & hit[near_cells]
    far_cells, near_cells = far_cells[m], near_cells[m]
    i_far, i_near = first[far_cells], first[near_cells]
    if len(i_far) == 0:
        return dict(p_idx=np.array([], np.int64), normals=np.zeros((0, 2)),
                    t_max=np.zeros(0))
    sep = np.linalg.norm(uv[i_far] - uv[i_near], axis=1)
    good = sep > min_sep_px
    i_far, i_near, sep = i_far[good], i_near[good], sep[good]
    if len(i_far) == 0:
        return dict(p_idx=np.array([], np.int64), normals=np.zeros((0, 2)),
                    t_max=np.zeros(0))
    # keep ONE pair per far point. Far side: the WIDEST baseline (max
    # reach for the one-sided band; nearest-strong-ridge still stops at
    # the first ridge). Near side: the SHORTEST -- the ring-1 near point
    # hugs the silhouette, while a ring-3 near point sits a dozen px
    # inside the object, away from the photometric edge.
    order = np.argsort(sep if near_side else -sep)
    _, first_occ = np.unique(i_far[order], return_index=True)
    keep = order[first_occ]
    i_far, i_near, sep = i_far[keep], i_near[keep], sep[keep]
    # spatial decorrelation: adjacent cells on the same physical contour
    # are near-duplicate measurements sharing one systematic bias (wall
    # texture, paint stripes) -- in the pooled residual that bias behaves
    # like N independent votes and walks the solution off GT (observed:
    # 0.1 -> 0.35 deg drift in the zero-noise check). One sample per
    # SUP px image cell keeps contours represented but independent.
    sup = 16
    key = ((uv[i_far, 1] // sup).astype(np.int64) * 4096
           + (uv[i_far, 0] // sup).astype(np.int64))
    _, s_idx = np.unique(key, return_index=True)
    s_idx.sort()
    i_far, i_near, sep = i_far[s_idx], i_near[s_idx], sep[s_idx]
    n2 = uv[i_near] - uv[i_far]
    n2 /= np.maximum(sep[:, None], 1e-9)
    if near_side:
        i_near, i_far = i_far, i_near
        n2 = -n2
        # re-key the per-point selection on the NEAR points: widest
        # baseline + 16 px decorrelation were deduped against far cells
        order = np.argsort(-sep)
        _, first_occ = np.unique(i_near[order], return_index=True)
        keep = order[first_occ]
        i_far, i_near, sep, n2 = (i_far[keep], i_near[keep], sep[keep],
                                  n2[keep])
        sup = 16
        key = ((uv[i_near, 1] // sup).astype(np.int64) * 4096
               + (uv[i_near, 0] // sup).astype(np.int64))
        _, s_idx = np.unique(key, return_index=True)
        s_idx.sort()
        i_near, n2, sep = i_near[s_idx], n2[s_idx], sep[s_idx]
    e_idx, e_nrm, e_tmax = i_near, n2, sep
    if max_samples and len(e_idx) > max_samples:
        keep = (rng or np.random.default_rng(0)).choice(
            len(e_idx), max_samples, replace=False)
        e_idx, e_nrm, e_tmax = e_idx[keep], e_nrm[keep], e_tmax[keep]
    return dict(p_idx=e_idx.astype(np.int64), normals=e_nrm, t_max=e_tmax)


def project_stacked_samples(frame, R_ec, R_le, sample):
    """uv of stacked-cloud samples (current pose) + validity."""
    uv, z, ok = frame.project_stacked(R_ec, R_le, idx=sample['p_idx'])
    return uv, ok


# ---- ground-wall contact interval evidence ----------------------------------
SOFTMAX_BETA = 0.05      # ridge centroid sharpness (per gradient unit)
INT_STEP = 0.5           # px, sampling step along the pair segment

def interval_residuals(frame, uv_far, normals, sep, weight_gate=True):
    """Soft CONTAINMENT residual for occluding-edge pairs.

    Geometry: the far point is the background return closest to an
    occluding contour, the near point is the foreground return on the
    other side. The photometric contact ridge necessarily projects
    BETWEEN them -- a constraint that is satisfied identically at the
    correct pose (zero loss, no bias injected: unlike an equality
    residual it cannot confuse beam-gap geometry with pose error), and
    is violated COHERENTLY under a rig rotation (ridges exit on one
    side), which is exactly what a fine-stage solver needs.

    The ridge position is the softmax centroid of gradient magnitude
    along the segment (smooth in the pose, unlike argmax); the residual
    is the distance by which the centroid exits [0, sep] along the
    normal (far -> near).

    Returns (residuals [n], weights [n]; weight 0 = dropped)."""
    n = len(uv_far)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    n_steps = int(np.ceil(sep.max() / INT_STEP)) + 1
    ts = np.arange(n_steps) * INT_STEP
    pos_u = uv_far[None, :, 0] + normals[None, :, 0] * ts[:, None]
    pos_v = uv_far[None, :, 1] + normals[None, :, 1] * ts[:, None]
    g = bilinear(frame.gm, pos_u, pos_v)                 # (n_steps, n)
    in_band = ts[:, None] <= sep[None, :]
    g = np.where(in_band, g, 0.0)
    z = SOFTMAX_BETA * g
    z = np.where(in_band, z, -np.inf)
    z = z - z.max(axis=0, keepdims=True)
    w = np.exp(z)
    w = np.where(in_band & (w > 1e-6), w, 0.0)
    wsum = w.sum(axis=0)
    ok = wsum > 1e-3
    t_c = np.where(ok, (w * ts[:, None]).sum(axis=0) / np.maximum(wsum, 1e-9),
                   0.5 * sep)
    if weight_gate:
        cu = uv_far[:, 0] + normals[:, 0] * t_c
        cv = uv_far[:, 1] + normals[:, 1] * t_c
        pk = bilinear(frame.teed_prob, cu, cv)
        ok &= pk >= TEED_PROB_GATE
    r = np.maximum(0.0, -t_c) + np.maximum(0.0, t_c - sep)
    return np.where(ok, r, 0.0), np.where(ok, 1.0, 0.0)


# ---- road-marking evidence (DIAGNOSTIC stream) ------------------------------
# Why markings and not depth edges as the OPTIMIZER objective: ground-wall
# contact pairs inherit a vertical beam-spacing bias (32 lines -> ~5.3 px
# between rings; the last ground return and first wall return each sit up to
# one ring gap from the true contact line, with a NONZERO mean that depends
# on range/geometry). Any scheme anchoring one side to the photometric
# contour is biased and the optimizer absorbs it as spurious rig pitch --
# observed as common-mode drift in the zero-noise GT check. Paint-rim points
# sit on the marking BOUNDARY, where the beam-phase offset is zero-mean.
# Recipe from AutoLidarCameraCalibration (teed_stacked_refinement /
# intensity_bev_contour); lifted to the joint (R_ec, R_le) parametrization.
MARK_INTENSITY_PCT = 90.0   # intensity percentile among ground returns
MARK_MIN_AREA = 8           # BEV cells per connected marking region
MARK_BEV_RES = 0.10         # m per BEV cell
DT_EDGE_PCT = 95.0          # TEED percentile for the distance-field edges
GROUND_PLANE_TOL = 0.10     # m, inlier band around the fitted ground plane


def build_dt_map(frame, percentile=DT_EDGE_PCT):
    """L2 distance transform of the strong-TEED edge set (px). The
    objective samples this bilinearly at re-projected marking points --
    smooth cones, so numeric-Jacobian LM works (an UNBLURRED field; the
    earlier blurred soft-DT chamfer had no basin at all)."""
    thr = float(np.percentile(frame.teed_prob, percentile))
    return cv2.distanceTransform(
        (frame.teed_prob <= thr).astype(np.uint8), cv2.DIST_L2, 3)


def marking_points(frame, intensity, max_points=400,
                   int_pct=MARK_INTENSITY_PCT, front_only=True):
    """Indices (into the stacked cloud) of high-intensity GROUND returns on
    the rim of BEV marking regions, in reference-ego coordinates at the
    BASE lidar rotation (fixed evidence, like the dynamic mask).

    front_only: the reference recipe candidates the FULL circle around the
    vehicle because it refines with 6 surround cameras; with a single
    front camera, keep only the VISIBLE ground band -- ground closer than
    ~4 m projects below the vertical FOV (camera ~1.5 m up, ~20 deg down
    half-angle), and ground farther than 30 m yields no intensity-
    separable paint anyway (far paint p99 intensity = 27, HDL-32E).
    Evidence must be selectable WITHOUT the image (physics: paint
    intensity), otherwise the optimizer has no true pull -- DT proximity
    filters select points that are near SOME edge at every pose and kill
    the basin (measured: flat at both GT and coarse)."""
    p_e = frame.stacked_ego_ref()
    if front_only:
        x_min, x_max = 4.0, 30.0
        y_abs = 25.0
    else:
        # full circle around the vehicle (the reference recipe candidates
        # the full disk because its refinement uses 6 surround cameras;
        # each camera then projects the subset it can see)
        x_min, x_max = -60.0, 70.0
        y_abs = 40.0
    band = ((p_e[:, 0] > x_min) & (p_e[:, 0] < x_max)
            & (np.abs(p_e[:, 1]) < y_abs)
            & (p_e[:, 2] > -1.5) & (p_e[:, 2] < 0.5))
    if int(band.sum()) < 100:
        return np.array([], np.int64)
    C = p_e[band]
    c = np.median(C, axis=0)
    _, _, vh = np.linalg.svd(C - c, full_matrices=False)
    n = vh[-1]
    if n[2] < 0:
        n = -n
    ground = band & (np.abs((p_e - c) @ n) < GROUND_PLANE_TOL)
    if int(ground.sum()) < 100:
        return np.array([], np.int64)
    cutoff = float(np.percentile(intensity[ground], int_pct))
    high = ground & (intensity >= cutoff)

    x0, y0 = (-60.0, -40.0) if not front_only else (-5.0, -40.0)
    x_w = 130.0 if not front_only else 75.0
    res = MARK_BEV_RES
    gx = ((p_e[:, 0] - x0) / res).astype(np.int32)
    gy = ((p_e[:, 1] - y0) / res).astype(np.int32)
    in_grid = high & (gx >= 0) & (gx < int(x_w / res)) \
        & (gy >= 0) & (gy < int(80.0 / res))
    if not in_grid.any():
        return np.array([], np.int64)
    occ = np.zeros((int(80.0 / res), int(x_w / res)), np.uint8)
    occ[gy[in_grid], gx[in_grid]] = 255
    # join the repeated returns of one painted region, keep sizable regions,
    # then keep only their rim cells (boundary = where the photometric
    # contour of the paint is)
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8)
    markings = np.zeros_like(closed)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= MARK_MIN_AREA:
            markings[labels == label] = 255
    rim = cv2.morphologyEx(markings, cv2.MORPH_GRADIENT,
                           np.ones((3, 3), np.uint8))
    sel = np.zeros(len(p_e), bool)
    sel[in_grid] = rim[gy[in_grid], gx[in_grid]] > 0
    ids = np.flatnonzero(sel)
    if len(ids) > max_points:
        ids = ids[::int(np.ceil(len(ids) / max_points))].copy()
    return ids
