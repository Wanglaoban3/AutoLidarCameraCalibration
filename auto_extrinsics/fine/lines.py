# -*- coding: utf-8 -*-
"""Contact-line evidence: 3D ground/facade intersection lines vs TEED ridges.

ANNOTATION-FREE (on-vehicle constraint): nothing here reads nuScenes
sample_annotation. Dynamic objects are not masked explicitly -- spurious
"facades" (car rows, near-sensor junk) are rejected by physical gates:
verticality |n_z| <= FACADE_MAX_NZ, plane offset >= FACADE_MIN_DIST,
ground-contact span >= MIN_LINE_LEN_3D. (GT-box masking was tried and
removed: boxes do not exist in a real run. The image-space dyn_mask still
used by the legacy depth-edge path is likewise box-based -- placeholder
until a segmenter or geometric heuristic replaces it.)

Why lines as the 3D side: a point-pair residual inherits the beam spacing
as per-pair uncertainty (~5 px vertical between 32 scan rings -- every
equality formulation either injects that as bias or, formulated as
containment, loses all signal when the pair extraction co-moves with the
pose). A contact LINE is the exact geometric intersection of the ground
plane with a facade plane -- no beam-gap term -- and it is extracted ONCE
at the base pose: pose errors move its PROJECTION, never the line itself
(the self-confirmation trap that killed re-extracted point pairs).

Why ridge search and not discrete image-line matching (LSD): LSD does not
emit the wall-base junction on this data -- its nearest segment to a
visibly perfect wall base was an unrelated diagonal line ~10 px away
(probe_lines, 2026-09-28; the segment list there had 319 entries, none on
that base). Instead each projected 3D-line sample searches the gradient
ridge along the line's projected normal within a tight symmetric window
(evidence.locate_ridges): the association is a property of the image,
subpixel, and the window bounds the aliasing reach. Targets are FROZEN
per pass; holdout lines gate publication.

Sample clipping: the physical line runs far past the image edges, so the
sampled segment is clipped to the base-pose frustum (longest visible run,
resampled uniformly) -- visibility is re-checked per pose in matching, but
the MARGIN (5 px) absorbs the <=1 deg projection shifts between passes.

Recipe lineage: GALIBL/Galibr-style ground-plane + line features
(referenced by AutoLidarCameraCalibration for validation), lifted to the
joint (R_le, R_ec) refinement role with the image-line counterpart
replaced by ridge search.
"""
import numpy as np

from auto_extrinsics.fine.evidence import locate_ridges

LINE_SAMPLES = 60         # sample points per 3D contact line
FACADE_MAX = 5            # vertical planes kept per frame
FACADE_MIN_INLIERS = 500  # stacked points per facade plane
FACADE_RANSAC_ITERS = 500 # 3-point draws (scored on a 15k subsample)
FACADE_SCORE_SUB = 15000  # pool subsample for RANSAC voting
FACADE_MAX_NZ = 0.06      # facades are vertical; junk near-sensor planes
FACADE_MIN_DIST = 2.5     # m, plane offset from sensor (ego-adjacent junk)
MIN_LINE_LEN_3D = 4.0     # m, contact segment must span this much
LINE_MIN_GOOD = 0.5       # fraction of samples that must lock a ridge


def _fit_planes_ego(p_e, ground_n, ground_c, rng, ok_mask=None, debug=False):
    """RANSAC vertical facades among non-ground stacked points (reference
    ego frame). ok_mask: points EXCLUDED from fitting (None = all points).
    Spans are measured over the FULL physical inliers, so walls partially
    occluded by parked cars still yield long contact lines; camera
    visibility is re-checked per candidate pose.
    Returns (planes, stats) -- stats: per-iteration diagnostics when
    debug=True."""
    stats = []
    up = np.array([0., 0., 1.])
    rel = p_e - ground_c
    dist_g = np.abs(rel @ ground_n)
    # exclude ego-adjacent returns: the sensor mast / vehicle body form the
    # DENSEST vertical planes in the scan (observed: |d| ~ 0.6-1.3 m, normals
    # along +/-x) -- they out-vote real facades and burn all FACADE_MAX
    # slots. No facade is ever within 4 m of the sensor.
    rng_2d = np.hypot(p_e[:, 0], p_e[:, 1])
    facade_pool = (dist_g > 0.3) & (rng_2d > 4.0) \
        & (p_e[:, 0] > 1.0) & (p_e[:, 0] < 70.0) \
        & (np.abs(p_e[:, 1]) < 40.0) & (p_e[:, 2] > -1.0) & (p_e[:, 2] < 25.0)
    if ok_mask is not None:
        facade_pool &= ok_mask
    P = p_e[facade_pool]
    if len(P) < FACADE_MIN_INLIERS:
        stats.append(dict(reason=f'pool {len(P)} < {FACADE_MIN_INLIERS}'))
        return [], stats
    planes = []
    remaining = P
    for _ in range(FACADE_MAX):
        m = len(remaining)
        if m < FACADE_MIN_INLIERS:
            break
        # score RANSAC draws on a fixed subsample: a 3-point draw lands on
        # ONE facade with p ~ (facade fraction)^3 (~1.6%/draw against a
        # 100k-pt mixed pool), so 120 full-pool draws miss walls that 500
        # subsampled draws find; the refit below uses the full pool
        S = remaining if m <= FACADE_SCORE_SUB else \
            remaining[rng.choice(m, FACADE_SCORE_SUB, replace=False)]
        best_n, best_in = None, np.array([], int)
        for _ in range(FACADE_RANSAC_ITERS):
            i0, i1, i2 = rng.choice(len(S), 3, replace=False)
            p0, p1, p2 = S[i0], S[i1], S[i2]
            n = np.cross(p1 - p0, p2 - p0)
            ln = np.linalg.norm(n)
            if ln < 1e-6:
                continue
            n /= ln
            if abs(n @ up) > FACADE_MAX_NZ:  # near-vertical only
                continue
            d = np.abs((S - p0) @ n)
            inl = np.nonzero(d < 0.15)[0]
            if len(inl) > len(best_in):
                best_n, best_in = n, inl
        if best_n is None or len(best_in) < 100:
            stats.append(dict(reason=f'best votes {len(best_in)} < 100'))
            break
        Q = remaining[np.abs((remaining - S[best_in].mean(0)) @ best_n)
                      < 0.15]
        if len(Q) < FACADE_MIN_INLIERS:
            stats.append(dict(reason=f'full-pool inliers {len(Q)} < '
                                    f'{FACADE_MIN_INLIERS} '
                                    f'(subsample votes {len(best_in)})'))
            break
        c = Q.mean(0)
        _, _, vh = np.linalg.svd(Q - c, full_matrices=False)
        n = vh[-1]
        if abs(n @ up) > FACADE_MAX_NZ:
            stats.append(dict(reason=f'tilt {abs(n @ up):.3f}', n=n))
            continue
        d_signed = -float(n @ c)
        if abs(d_signed) < FACADE_MIN_DIST:
            stats.append(dict(reason=f'offset {abs(d_signed):.1f}m', n=n))
            continue
        d = np.abs((remaining - c) @ n)
        best_in = np.nonzero(d < 0.15)[0]
        if len(best_in) < FACADE_MIN_INLIERS:
            stats.append(dict(reason='refit inliers low', n=n))
            break
        stats.append(dict(reason='ok', n=n, d=d_signed,
                          inliers=int(len(best_in))))
        planes.append((n, d_signed, remaining[best_in].copy()))
        remaining = np.delete(remaining, best_in, axis=0)
    return planes, stats


def contact_lines(frame, rng, debug=False):
    """3D contact lines (ground x facade) for one frame, at the BASE pose.
    Annotation-free: no GT boxes, no image-space dyn mask on the pool.

    debug=True stores frame.line_debug: one dict per candidate plane with
    its reject reason (or 'ok') -- probe-only, no stdout noise.

    Returns list of dicts with p_l [LINE_SAMPLES, 3] sample points in the
    KEY sweep's lidar frame (re-projectable at any candidate R_le). Also
    stores the frame's ground plane."""
    p_e = frame.stacked_ego_ref()
    band = ((p_e[:, 0] > -15.0) & (p_e[:, 0] < 70.0)
            & (np.abs(p_e[:, 1]) < 25.0)
            & (p_e[:, 2] > -1.5) & (p_e[:, 2] < 0.5))
    if int(band.sum()) < 200:
        return []
    C = p_e[band]
    c = np.median(C, axis=0)
    _, _, vh = np.linalg.svd(C - c, full_matrices=False)
    n_g = vh[-1]
    if n_g[2] < 0:
        n_g = -n_g
    d_g = -float(n_g @ c)
    frame.ground_plane = (n_g, d_g)

    planes, stats = _fit_planes_ego(p_e, n_g, c, rng, debug=debug)
    dbg = list(stats)
    lines = []
    for n_f, d_f, inl in planes:
        info = dict(n=n_f, d=d_f, inliers=len(inl), reason='')
        dbg.append(info)
        dir_l = np.cross(n_f, n_g)
        ln = np.linalg.norm(dir_l)
        if ln < 1e-6:
            info['reason'] = 'degenerate direction'
            continue
        dir_l /= ln
        # point on both planes: solve [n_f; n_g; dir] x = [-d_f; -d_g; 0]
        A = np.stack([n_f, n_g, dir_l])
        b = np.array([-d_f, -d_g, 0.0])
        try:
            p0 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            info['reason'] = 'solve fail'
            continue
        t = (inl - p0) @ dir_l
        t0, t1 = np.percentile(t, [4, 96])
        info['span'] = float(t1 - t0)
        if t1 - t0 < MIN_LINE_LEN_3D:
            info['reason'] = f'span {t1 - t0:.1f}m < {MIN_LINE_LEN_3D}'
            continue
        ts = np.linspace(t0, t1, LINE_SAMPLES)

        def to_lidar(q):
            """ref-ego -> key-sweep lidar (inverse of project_stacked's
            p_s = R_le @ p_l + t_le; p_e = G @ p_s)."""
            G = frame.stack_G[-1]
            p = (np.linalg.inv(G) @ np.vstack(
                [q.T, np.ones(len(q))])).T[:, :3]
            return (p - frame.t_le) @ frame.R_le0

        p_l = to_lidar(p0[None, :] + ts[:, None] * dir_l[None, :])
        # clip to the base-pose frustum: only the visible run can ever
        # match image evidence, and uniform sampling over the WHOLE line
        # dilutes it below the in-fov gate (observed: 35/60 on a 40 m wall)
        T_lc0 = frame.T_lc(frame.R_ec0, frame.R_le0)
        pc = (T_lc0[:3, :3] @ p_l.T).T + T_lc0[:3, 3]
        zz = pc[:, 2]
        uv = (frame.K @ pc.T).T[:, :2] / np.maximum(zz[:, None], 1e-9)
        h, w = frame.shape
        vis = (zz > 1.5) & (uv[:, 0] > 5) & (uv[:, 0] < w - 6) \
            & (uv[:, 1] > 5) & (uv[:, 1] < h - 6)
        runs, s = [], None
        for i, v_ in enumerate(vis):
            if v_ and s is None:
                s = i
            elif not v_ and s is not None:
                runs.append((s, i))
                s = None
        if s is not None:
            runs.append((s, len(vis)))
        if not runs:
            info['reason'] = 'not in camera fov'
            continue
        i0, i1 = max(runs, key=lambda r: r[1] - r[0])
        info['visible'] = int(i1 - i0)
        if i1 - i0 < 12:
            info['reason'] = f'longest visible run {i1 - i0} < 12 samples'
            continue
        ts_v = np.linspace(ts[i0], ts[i1 - 1], LINE_SAMPLES)
        lines.append(dict(p_l=to_lidar(
            p0[None, :] + ts_v[:, None] * dir_l[None, :])))
    if debug:
        frame.line_debug = dbg
    frame.contact_lines = lines
    return lines


# ---- ridge association (frozen per pass) ------------------------------------

def project_line(frame, p_l, R_ec, R_le):
    """uv [n,2] and depth of contact-line samples at the candidate pose."""
    T_lc = frame.T_lc(R_ec, R_le)
    pc = (T_lc[:3, :3] @ p_l.T).T + T_lc[:3, 3]
    z = pc[:, 2]
    uv = (frame.K @ pc.T).T[:, :2] / np.maximum(z[:, None], 1e-9)
    return uv, z


def line_samples_px(frame, line, R_ec, R_le):
    """(uv, normals, ok): in-fov mask, plus per-sample search normals --
    the in-image perpendicular of the projected line direction, constant
    along the line to first order (the ridge-search axis)."""
    uv, z = project_line(frame, line['p_l'], R_ec, R_le)
    h, w = frame.shape
    ok = (z > 1.5) & (z < 80) & (uv[:, 0] > 2) & (uv[:, 0] < w - 3) \
        & (uv[:, 1] > 2) & (uv[:, 1] < h - 3)
    if ok.sum() < 6:
        return uv, None, ok
    d = uv[ok][-1] - uv[ok][0]
    nn = np.hypot(d[0], d[1])
    if nn < 1e-6:
        return uv, None, ok
    nvec = np.array([-d[1], d[0]]) / nn
    normals = np.broadcast_to(nvec, (len(uv), 2)).copy()
    return uv, normals, ok


def associate_lines(frames, lines_per_frame, R_ec, R_le, half_win):
    """Frozen ridge association at the pass pose: per line, locate ridge
    targets along frozen normals; the entry carries everything the smooth
    residual needs. Lines whose association is too weak are dropped
    (None) -- association QUALITY gates, never residual size (a residual
    gate would delete exactly the lines that disagree, i.e. the evidence)."""
    assoc = []
    for fg, lines in zip(frames, lines_per_frame):
        per = []
        for line in lines:
            uv, normals, ok = line_samples_px(fg, line, R_ec, R_le)
            if normals is None or ok.sum() < max(10, LINE_MIN_GOOD * len(uv)):
                per.append(None)
                continue
            targets, w = locate_ridges(fg, uv[ok], normals[ok], half_win)
            if (w > 0).sum() < max(8, LINE_MIN_GOOD * int(ok.sum())):
                per.append(None)
                continue
            per.append(dict(frame=fg, p_l=line['p_l'],
                            gate=bool(line.get('gate', False)),
                            idx=np.nonzero(ok)[0], normals=normals,
                            targets=targets, w=w))
        assoc.append(per)
    return assoc


def frozen_residuals(entry, R_ec, R_le):
    """Signed px offsets of CURRENT projections to the frozen ridge targets
    along frozen normals; weights from association (0 = dropped)."""
    uv, _ = project_line(entry['frame'], entry['p_l'], R_ec, R_le)
    idx = entry['idx']
    d = uv[idx] - entry['targets']
    nm = entry['normals'][idx]
    res = d[:, 0] * nm[:, 0] + d[:, 1] * nm[:, 1]
    return res, entry['w']


def line_px(frame, line, R_ec, R_le, half_win):
    """Measurement-style quality of one line at a fixed pose: ridge
    targets re-located fresh (assignment unambiguous at a fixed pose).
    Returns (median |offset| px, n_locked) or (None, 0)."""
    uv, normals, ok = line_samples_px(frame, line, R_ec, R_le)
    if normals is None or ok.sum() < max(10, LINE_MIN_GOOD * len(uv)):
        return None, 0
    targets, w = locate_ridges(frame, uv[ok], normals[ok], half_win)
    good = w > 0
    if good.sum() < max(8, LINE_MIN_GOOD * int(ok.sum())):
        return None, int(good.sum())
    d = uv[ok][good] - targets[good]
    nm = normals[ok][good]
    res = d[:, 0] * nm[:, 0] + d[:, 1] * nm[:, 1]
    return float(np.median(np.abs(res))), int(good.sum())
