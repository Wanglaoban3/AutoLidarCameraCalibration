# -*- coding: utf-8 -*-
"""Mutual-information (NID) refinement -- the classic targetless LiDAR-
camera fine registration (Pandey et al. 2011 MI; Taylor & Nieto 2016
NID), promoted here to the primary fine family after six
correspondence-based families failed on one of three horns:

  - image-selected evidence (pick points near edges) has NO pull: the
    points are near SOME edge at every pose, the basin flattens;
  - geometry-selected evidence (all depth jumps) is junk-dominated:
    canopy/texture jumps carry a flat-DT gradient that outweighs the
    informative minority;
  - discrete matching (lines, ridges, planes) re-locks adaptively, so a
    coherent slide leaves every re-locked gate metric unchanged.

MI uses NO correspondence: every stacked point with a calibrated
intensity votes into a joint (intensity, camera-gray) histogram, and the
true pose maximizes the shared information between the modalities.
Nothing about the objective cares whether the scene has paint, walls or
edges, so it generalizes across clips by construction -- which is the
M1 acceptance (all 10 v1.0-mini scenes, ~1 deg injected coarse error).

Estimator notes (measured on the first probe, scripts/probe_nid.py):
raw [0,255] binning of HDL-32E intensity crushes 99% of the points into
the lowest 3 bins (ground p99 = 27), the joint histogram degenerates and
the MI surface is flat noise (argmax of the 1-D sweeps wanders over the
whole +-1.5 deg box). EQUAL-FREQUENCY bin edges from the fixed point
subsample -- pose-independent, computed once -- spread the marginals and
restore a usable surface. Powell, not gradient descent: MI estimate
noise makes finite differences meaningless.

Mechanics: fixed per-frame point subsample (extracted once -- the stack
never moves in lidar coordinates), bilinear gray sampling at the
re-projection, 16x16 joint histogram, ~150k votes per evaluation,
bounded Powell, tiny quadratic prior. Publish gate lives in the runner:
holdout-frame MI must improve, valid fraction must not collapse, no
bound hit."""
import cv2
import numpy as np

from auto_extrinsics.fine import evidence as ev
from auto_extrinsics.fine.solver import apply_deltas

NID_BINS = 16        # joint-histogram bins per modality (equal-frequency)
NID_PTS = 25000      # stacked points voting per frame
NID_PRIOR = 0.01     # nats per deg^2 -- barely-there 6-DoF regularizer


def _edges(vals, bins):
    """Equal-frequency bin edges from the fixed subsample: interior
    quantiles, so both marginals spread over all bins regardless of the
    sensor's intensity scale (HDL-32E lives in 0-60, not 0-255)."""
    q = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    e = np.quantile(vals, q)
    return np.unique(e)


class NIDRefiner:
    """Fixed-evidence MI objective over a frame list."""

    def __init__(self, frames, n_pts=NID_PTS, bins=NID_BINS, seed=0):
        self.bins = bins
        rng = np.random.default_rng(seed)
        self.frs = []
        for fg in frames:
            n = len(fg.stack_p)
            if n > n_pts:
                idx = np.sort(rng.choice(n, n_pts, replace=False))
            else:
                idx = np.arange(n)
            fg.nid_idx = idx
            inten = np.clip(fg.stack_intensity[idx], 0.0, None)
            fg.nid_gray = cv2.cvtColor(fg.img, cv2.COLOR_BGR2GRAY).astype(
                np.float64)
            gray = fg.nid_gray.ravel()
            fg.nid_ie = np.clip(np.searchsorted(
                _edges(inten, bins), inten), 0, bins - 1)
            fg.nid_ge_edges = _edges(gray, bins)
            self.frs.append(fg)

    def mi(self, frs, R_le, R_ec):
        """(mutual information in nats, valid fraction) at the given pose."""
        b = self.bins
        h = np.zeros(b * b, np.float64)
        n_ok = n_all = 0
        for fg in frs:
            uv, z, ok = fg.project_stacked(R_ec, R_le, idx=fg.nid_idx)
            n_all += len(uv)
            n_ok += int(ok.sum())
            if not ok.any():
                continue
            g = ev.bilinear(fg.nid_gray, uv[ok, 0], uv[ok, 1])
            gi = np.clip(np.searchsorted(fg.nid_ge_edges, g), 0, b - 1)
            h += np.bincount(fg.nid_ie[ok] * b + gi, minlength=b * b)
        tot = h.sum()
        vf = n_ok / max(n_all, 1)
        if tot < 200:
            return 0.0, vf
        p = (h / tot).reshape(b, b)
        pi = p.sum(axis=1, keepdims=True)
        pj = p.sum(axis=0, keepdims=True)
        indep = pi @ pj
        nz = p > 0
        return float((p[nz] * np.log(p[nz] / indep[nz])).sum()), vf

    def cost(self, x_deg, frs, R_le0, R_ec0):
        R_le, R_ec = apply_deltas(R_le0, R_ec0, np.radians(x_deg))
        mi, _ = self.mi(frs, R_le, R_ec)
        return -mi + NID_PRIOR * float(x_deg @ x_deg)

    def run(self, train, hold, R_le0, R_ec0, half_box=1.2, maxiter=12):
        """Bounded Powell from the base pose. Returns (x_deg, info dict
        with train/holdout MI and valid fractions at both poses)."""
        from scipy.optimize import minimize
        x0 = np.zeros(6)
        mi0_tr, vf0_tr = self.mi(train, R_le0, R_ec0)
        mi0_ho, vf0_ho = self.mi(hold, R_le0, R_ec0)

        def f(x_deg):
            return self.cost(x_deg, train, R_le0, R_ec0)

        res = minimize(f, x0, method='Powell',
                       bounds=[(-half_box, half_box)] * 6,
                       options=dict(maxiter=maxiter, xtol=1e-3, ftol=1e-7))
        x = np.clip(res.x, -half_box, half_box)
        R_le_t, R_ec_t = apply_deltas(R_le0, R_ec0, np.radians(x))
        mi1_tr, vf1_tr = self.mi(train, R_le_t, R_ec_t)
        mi1_ho, vf1_ho = self.mi(hold, R_le_t, R_ec_t)
        info = dict(mi_train=(mi0_tr, mi1_tr), mi_hold=(mi0_ho, mi1_ho),
                    vf_train=(vf0_tr, vf1_tr), vf_hold=(vf0_ho, vf1_ho),
                    nit=res.nit, nfev=res.nfev, success=bool(res.success))
        return x, info
