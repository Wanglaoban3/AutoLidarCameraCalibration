# -*- coding: utf-8 -*-
"""TLC-Calib port (Lu et al.): targetless LiDAR-camera calibration by
line-feature matching, lifted to the joint (R_le, R_ec) rotation-only
parametrization of this repo.

Variant A (faithful core): line-to-line. 3D lines = ground x facade
contact lines of the stacked cloud (lines.py, extracted ONCE at the base
pose); 2D lines = LSD segments of the image (HoughLinesP fallback if the
OpenCV build rejects createLineSegmentDetector). Matching uses POSE-STABLE
descriptors only -- line direction angle (mod 180), distance of the 2D
segment midpoint to the projected 3D polyline, projected length -- plus a
TTL-style best/second-best uniqueness ratio and greedy 1-1 exclusivity.
Residual = per-sample point-to-corresponding-line px distance, where the
2D segment acts as its INFINITE support inside a frozen span window,
minimized by the repo's robust LM over x = [rotvec_le(3), rotvec_ec(3)].

Variant B (paper's fallback family, documented): point-to-line. When LSD
segments only partially cover a projected 3D line (the measured situation
on this data -- LSD does not emit complete wall-base lines), each 3D line
SAMPLE is matched to its nearest direction-gated segment (infinite
support residual as in A, but assignment is per sample). This relaxes the
whole-line correspondence while keeping the same support geometry.

Association discipline (the lesson that killed run_m1 --stage lines):
matching happens EXACTLY ONCE, at the base pose, and is FROZEN for the
whole run. There is NO re-matching between outer passes, so no sliding
pose can re-lock its own evidence: a coherent slide moves projections OFF
the frozen 2D supports, and the holdout gate -- computed against the same
frozen supports (odd frames + 30% of lines never fitted) -- sees it. The
gate is therefore built from quantities that cannot co-move with the pose.
Per-sample validity can only SHRINK (span window + a frozen 30 px
displacement guard); it can never re-target.
"""
import numpy as np

from auto_extrinsics.fine import lines as fl

# ---- 2D segment extraction --------------------------------------------------
SEG_MIN_LEN_PX = 40.0     # shorter segments are texture noise (measured pool)
HOUGH_THRESHOLD = 60
HOUGH_MAX_GAP = 8

# ---- matching gates (applied at the base pose, then frozen) ------------------
MATCH_DIST_PX = 15.0      # max seg-midpoint -> projected-polyline distance
MATCH_ANG_DEG = 10.0      # max direction difference, line-level (variant A)
POINT_ANG_DEG = 12.0      # direction gate for per-sample matching (variant B)
POINT_UNIQ_RATIO = 0.7    # per-sample best/second distance ratio gate
PT_STEP = 2               # variant B: match every PT_STEP-th line sample
UNIQ_RATIO = 0.75         # line-level best/second cost ratio (variant A)
MIN_PROJ_SAMPLES = 12     # visible samples needed to attempt matching
MIN_PROJ_LEN_PX = 40.0    # projected 3D-line length needed to attempt matching
MIN_GROUP_SAMPLES = 3     # variant B: samples per (line, seg) group
COLLINEAR_STD_PX = 2.0    # group consistency: signed base offsets along one
                          # straight support must be near-constant (a rigid
                          # counterpart line is collinear with the projected
                          # 3D line; texture lines crossing it are not)

# ---- frozen residual supports ------------------------------------------------
SPAN_MARGIN_PX = 15.0     # along-segment validity margin (absorbs pose shift)
MOVE_GUARD_PX = 30.0      # frozen per-sample displacement guard
BOUND_DEG = 1.0           # per-DoF trust bound for the publish gate
MIN_GATE_RES = 20         # holdout residuals needed for a valid gate metric


# ---- 2D segments -------------------------------------------------------------
def image_segments(fg):
    """2D line segments of the frame image. LSD when the OpenCV build
    provides a working createLineSegmentDetector, HoughLinesP on Canny
    otherwise (the caller reports which method ran). Returns
    (segs, method, lsd_error) with seg dicts p0, d, n, ang (mod pi),
    length, mid; lsd_error records why LSD was rejected if it was."""
    import cv2
    gray = cv2.cvtColor(fg.img, cv2.COLOR_BGR2GRAY)
    raw, method, err = None, 'none', ''
    try:
        lsd = cv2.createLineSegmentDetector()
        out = lsd.detect(gray)[0]
        if out is not None:
            raw = np.asarray(out).reshape(-1, 4)
            method = 'lsd'
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
        raw = None
    if raw is None:
        edges = cv2.Canny(gray, 40, 120, apertureSize=3)
        found = cv2.HoughLinesP(edges, 1, np.pi / 360,
                                threshold=HOUGH_THRESHOLD,
                                minLineLength=int(SEG_MIN_LEN_PX),
                                maxLineGap=HOUGH_MAX_GAP)
        method = 'hough'
        if found is not None:
            raw = np.asarray(found).reshape(-1, 4)
    segs = []
    if raw is None:
        return segs, method, err
    x1, y1, x2, y2 = (raw[:, k].astype(float) for k in range(4))
    length = np.hypot(x2 - x1, y2 - y1)
    for ax, ay, bx, by, L in zip(x1, y1, x2, y2, length):
        if L < SEG_MIN_LEN_PX:
            continue
        p0 = np.array([ax, ay])
        d = np.array([bx - ax, by - ay]) / L
        segs.append(dict(p0=p0, d=d, n=np.array([-d[1], d[0]]),
                         ang=float(np.arctan2(d[1], d[0])) % np.pi,
                         length=float(L), mid=p0 + 0.5 * L * d))
    return segs, method, err


def seg_arrays(segs):
    """Vectorized per-frame segment arrays for point-to-segment queries."""
    if not segs:
        return None
    return dict(P0=np.stack([s['p0'] for s in segs]),
                D=np.stack([s['d'] for s in segs]),
                LEN=np.array([s['length'] for s in segs]),
                ANG=np.array([s['ang'] for s in segs]))


def _pt_seg_dist(p, arr):
    """Distance of one point [2] to every segment (finite support)."""
    q = p[None, :] - arr['P0']                       # (m, 2)
    t = np.clip((q * arr['D']).sum(1), 0.0, arr['LEN'])
    proj = arr['P0'] + t[:, None] * arr['D']
    return np.linalg.norm(p[None, :] - proj, axis=1)


def _ang_diff(a, b):
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


# ---- projection --------------------------------------------------------------
def project_uv(fg, p_l, R_ec, R_le):
    """uv and in-fov mask of 3D line samples at a candidate pose."""
    uv, z = fl.project_line(fg, p_l, R_ec, R_le)
    h, w = fg.shape
    ok = (z > 1.5) & (z < 80) & (uv[:, 0] > 2) & (uv[:, 0] < w - 3) \
        & (uv[:, 1] > 2) & (uv[:, 1] < h - 3)
    return uv, ok


# ---- frozen entries ------------------------------------------------------------
def _freeze(fg, line, idx, uv, seg, gate):
    """Build one frozen entry from kept sample indices at the base pose.
    Carries everything the smooth residual needs; targets never move.
    Collinearity gate: the group's signed offsets to the support at the
    base pose must be near-constant (std <= COLLINEAR_STD_PX) -- this is a
    POSE-STABLE descriptor (computed once at the base pose) that rejects
    texture segments crossing the projected line, which pure distance and
    direction gates admit (measured: without it the GT holdout median is
    ~6 px and the solver slides both sensors ~6 deg onto junk supports)."""
    idx = np.asarray(idx, np.int64)
    half = 0.5 * seg['length'] + SPAN_MARGIN_PX
    s = (uv - seg['mid']) @ seg['d']
    keep = idx[np.abs(s[idx]) <= half]
    if len(keep) < MIN_GROUP_SAMPLES:
        return None
    r0 = (uv[keep] - seg['p0']) @ seg['n']
    std0 = float(np.std(r0))
    if std0 > COLLINEAR_STD_PX:
        return None
    return dict(frame=fg, p_l=line['p_l'], idx=keep,
                uv_base=uv[keep].copy(), p0=seg['p0'].copy(),
                d=seg['d'].copy(), n=seg['n'].copy(),
                mid=seg['mid'].copy(), half=half, gate=bool(gate),
                n_seg=float(seg['length']), std0=std0,
                r0_med=float(np.median(r0)),
                span=float(s[keep].max() - s[keep].min()))


def entry_residual(e, R_ec, R_le):
    """Signed px offsets of the CURRENT projections to the FROZEN 2D support
    (infinite line of the matched segment). Validity can only shrink:
    frozen span window + frozen displacement guard."""
    uv, _ = fl.project_line(e['frame'], e['p_l'], R_ec, R_le)
    u = uv[e['idx']]
    r = (u - e['p0']) @ e['n']
    s = np.abs((u - e['mid']) @ e['d'])
    disp = np.linalg.norm(u - e['uv_base'], axis=1)
    valid = (s <= e['half']) & (disp <= MOVE_GUARD_PX)
    return np.where(valid, r, 0.0), valid.astype(float)


# ---- variant A: line-to-line ---------------------------------------------------
def match_line2line(frames, lines_per_frame, R_ec, R_le):
    """Strict TLC-Calib matching, computed ONCE at the base pose and frozen:
    per 3D line, the best LSD segment under (angle, distance) gates with a
    best/second uniqueness ratio; greedy 1-1 exclusivity on segments.
    Returns (entries [flat], per_frame counts)."""
    pairs = []
    for fi, (fg, lines) in enumerate(zip(frames, lines_per_frame)):
        for li, line in enumerate(lines):
            uv, ok = project_uv(fg, line['p_l'], R_ec, R_le)
            if ok.sum() < MIN_PROJ_SAMPLES:
                continue
            uvo = uv[ok]
            e2 = uvo[-1] - uvo[0]
            if np.hypot(e2[0], e2[1]) < MIN_PROJ_LEN_PX:
                continue
            ang3 = float(np.arctan2(e2[1], e2[0]) % np.pi)
            cands = []
            for si, s in enumerate(fg.tlc_segs):
                if _ang_diff(ang3, s['ang']) > np.radians(MATCH_ANG_DEG):
                    continue
                dmid = float(np.min(np.hypot(*(uvo - s['mid']).T)))
                if dmid > MATCH_DIST_PX:
                    continue
                cost = dmid / MATCH_DIST_PX \
                    + _ang_diff(ang3, s['ang']) / np.radians(MATCH_ANG_DEG)
                cands.append((cost, dmid, si))
            if not cands:
                continue
            cands.sort()
            if len(cands) > 1 and cands[0][0] > UNIQ_RATIO * cands[1][0]:
                continue
            pairs.append(dict(fi=fi, li=li, cost=cands[0][0],
                              dmid=cands[0][1], si=cands[0][2]))
    # greedy exclusivity: closest match claims its segment first
    claimed = set()
    used_lines = set()
    for pr in sorted(pairs, key=lambda p: p['dmid']):
        key = (pr['fi'], pr['si'])
        lkey = (pr['fi'], pr['li'])
        if key in claimed or lkey in used_lines:
            continue
        claimed.add(key)
        used_lines.add(lkey)
    entries = []
    per_frame = [0] * len(frames)
    for pr in sorted(pairs, key=lambda p: p['dmid']):
        lkey = (pr['fi'], pr['li'])
        if lkey not in used_lines:
            continue
        fg = frames[pr['fi']]
        line = lines_per_frame[pr['fi']][pr['li']]
        uv, ok = project_uv(fg, line['p_l'], R_ec, R_le)
        e = _freeze(fg, line, np.nonzero(ok)[0], uv,
                    fg.tlc_segs[pr['si']], line.get('gate', False))
        if e is None or len(e['idx']) < 8:
            continue
        e['dmid'] = pr['dmid']
        entries.append(e)
        per_frame[pr['fi']] += 1
    return entries, per_frame


# ---- variant B: point-to-line ---------------------------------------------------
def match_point2line(frames, lines_per_frame, R_ec, R_le):
    """Fallback matching, computed ONCE at the base pose and frozen: each
    3D-line sample takes its nearest direction-gated segment (infinite
    support), best/second ratio gate. One entry per (line, seg) group."""
    entries = []
    per_frame = [0] * len(frames)
    n_samples = 0
    for fi, (fg, lines) in enumerate(zip(frames, lines_per_frame)):
        arr = fg.tlc_seg_arr
        if arr is None:
            continue
        for line in lines:
            uv, ok = project_uv(fg, line['p_l'], R_ec, R_le)
            if ok.sum() < MIN_PROJ_SAMPLES:
                continue
            uvo = uv[ok]
            e2 = uvo[-1] - uvo[0]
            if np.hypot(e2[0], e2[1]) < MIN_PROJ_LEN_PX:
                continue
            ang3 = float(np.arctan2(e2[1], e2[0]) % np.pi)
            assign = {}
            for i in np.nonzero(ok)[0][::PT_STEP]:
                dist = _pt_seg_dist(uv[i], arr)
                ang_ok = np.array([_ang_diff(ang3, a) for a in arr['ANG']]) \
                    <= np.radians(POINT_ANG_DEG)
                cand = np.nonzero((dist <= MATCH_DIST_PX) & ang_ok)[0]
                if len(cand) == 0:
                    continue
                order = cand[np.argsort(dist[cand])]
                if len(order) > 1 and dist[order[0]] > POINT_UNIQ_RATIO * dist[order[1]]:
                    continue
                assign.setdefault(int(order[0]), []).append(i)
            for si, idxs in assign.items():
                e = _freeze(fg, line, idxs, uv, fg.tlc_segs[si],
                            line.get('gate', False))
                if e is None:
                    continue
                entries.append(e)
                n_samples += len(e['idx'])
                per_frame[fi] += 1
    return entries, per_frame, n_samples


# ---- objective and gate ---------------------------------------------------------
def make_obj_fn(entries, R_le0, R_ec0):
    """LM objective: pooled frozen point-to-support px residuals of TRAIN
    entries (holdout excluded), for solver.apply_deltas convention."""

    def fn(x):
        from auto_extrinsics.fine.solver import apply_deltas
        R_le, R_ec = apply_deltas(R_le0, R_ec0, x)
        rs, ws = [], []
        for e in entries:
            if e['gate']:
                continue
            r, w = entry_residual(e, R_ec, R_le)
            rs.append(r)
            ws.append(w)
        return np.concatenate(rs), np.concatenate(ws)
    return fn


def gate_metrics(entries, R_ec, R_le):
    """Holdout publish gate against FROZEN supports (cannot co-move with the
    pose): med = median |offset| px over holdout residuals; bias = norm of
    the mean SIGNED offset vector (median alone is insensitive to a
    coherent slide; the signed mean exposes it). Returns (med, bias, n)."""
    acc_r, acc_v = [], []
    for e in entries:
        if not e['gate']:
            continue
        r, w = entry_residual(e, R_ec, R_le)
        m = w > 0
        if not m.any():
            continue
        acc_r.append(np.abs(r[m]))
        acc_v.append(r[m][:, None] * e['n'][None, :])
    if not acc_r:
        return None, None, 0
    v = np.concatenate(acc_r)
    if len(v) < MIN_GATE_RES:
        return None, None, int(len(v))
    vv = np.concatenate(acc_v)
    bias = float(np.hypot(vv[:, 0].mean(), vv[:, 1].mean()))
    return float(np.median(v)), bias, int(len(v))
