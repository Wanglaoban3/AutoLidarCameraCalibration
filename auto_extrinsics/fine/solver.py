# -*- coding: utf-8 -*-
"""Joint SO(3)xSO(3) Levenberg-Marquardt over the 6 mounting-rotation DoF.

State x = [delta_le (3), delta_ec (3)] as rotation vectors; current poses
R = Exp(delta) @ R_base. Residuals are pooled over all frames (one rigid
mounting error per sensor, per run). The Jacobian is central finite
differences of the full residual pipeline -- robust to the subpixel search
and soft gates inside the residual function.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def apply_deltas(R_le_base, R_ec_base, x):
    R_le = Rotation.from_rotvec(x[:3]).as_matrix() @ R_le_base
    R_ec = Rotation.from_rotvec(x[3:]).as_matrix() @ R_ec_base
    return R_le, R_ec


def _huber(r, w, delta):
    a = np.abs(r)
    scale = np.where(a <= delta, 0.5 * r * r,
                     delta * (a - 0.5 * delta))
    return float(np.sum(w * scale)), r * np.where(a <= delta, 1.0,
                                                  delta / np.maximum(a, 1e-9))


class LMSolver:
    def __init__(self, residual_fn, x0, huber=2.0, fd_step=1e-4,
                 prior_sigma_deg=0.3, focal_px=1266.0):
        """residual_fn(x) -> (r [n], w [n]); w==0 entries are dropped.

        prior_sigma_deg: weak MAP prior pulling x toward 0 (the coarse
        estimate is a trusted initialization). It makes JtWJ strictly
        positive definite -- without it the normal-flow residuals are rank
        deficient, GN steps explode along the null space, and the trust
        region rescale crushes the useful component (LM stalls at ~0.002
        deg/iter, observed)."""
        self.fn = residual_fn
        self.x = x0.astype(float).copy()
        self.huber = huber
        self.fd = fd_step
        self.prior_w = (None if prior_sigma_deg is None else
                        1.0 / (prior_sigma_deg * focal_px * np.pi / 180) ** 2)
        self.focal_px = focal_px
        self.history = []

    def _eval(self, x):
        r, w = self.fn(x)
        if self.prior_w is not None:
            r = np.concatenate([r, x * self.focal_px])
            w = np.concatenate([w, np.full(len(x), self.prior_w)])
        return r, w

    def run(self, max_iter=15, tol=1e-8, verbose=True, max_step_deg=1.5):
        r, w = self._eval(self.x)
        m = w > 0
        if m.sum() < 12:
            raise RuntimeError(f'only {int(m.sum())} valid residuals at '
                               'solver start -- evidence or pose broken')
        cost, rw = _huber(r[m], w[m], self.huber)
        lam = 1e-3
        n = len(self.x)
        max_step = np.radians(max_step_deg)
        JtWJ = Jtw = sigma = None
        for it in range(max_iter):
            cols = []
            for i in range(n):
                xp = self.x.copy(); xp[i] += self.fd
                xm = self.x.copy(); xm[i] -= self.fd
                rp, _ = self._eval(xp)
                rm, _ = self._eval(xm)
                cols.append((rp[m] - rm[m]) / (2 * self.fd))
            J = np.stack(cols, axis=1)
            JtWJ = J.T @ (w[m][:, None] * J)
            Jtw = J.T @ (w[m] * rw)
            improved = False
            rel = 0.0
            for _ in range(6):
                A = JtWJ + lam * np.diag(np.maximum(np.diag(JtWJ), 1e-9))
                try:
                    dx = np.linalg.solve(A, -Jtw)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                # trust region: sparse-evidence Gauss-Newton steps can be
                # wild; cap the per-iteration rotation magnitude per sensor
                nrm = max(np.linalg.norm(dx[:3]), np.linalg.norm(dx[3:]))
                if nrm > max_step:
                    dx = dx * (max_step / nrm)
                xt = self.x + dx
                rt, wt = self._eval(xt)
                mt = wt > 0
                cost_t, rwt = _huber(rt[mt], wt[mt], self.huber)
                if np.isfinite(cost_t) and cost_t < cost:
                    self.x = xt
                    rel = (cost - cost_t) / max(cost, 1e-9)
                    self.history.append(cost_t)
                    if verbose:
                        print(f'  lm it{it}: cost {cost:.4f} -> {cost_t:.4f} '
                              f'|dx|={np.degrees(np.linalg.norm(dx[:3])):.3f}/'
                              f'{np.degrees(np.linalg.norm(dx[3:])):.3f} deg '
                              f'lam={lam:.1e}')
                    r, w, m, rw, cost = rt, wt, mt, rwt, cost_t
                    lam = max(lam * 0.3, 1e-9)
                    improved = True
                    break
                lam *= 10
            if not improved or rel < tol:
                break
        # per-DoF uncertainty from the final linearization (information
        # gating input, spec 4.5): sigma_i = sqrt(diag(inv(JtWJ))) scaled by
        # a robust residual std. The 2n guard keeps data-only statistics
        # (the appended prior pseudo-residuals must not inflate the count).
        dof_sigma = None
        if JtWJ is not None and m.sum() > 2 * n:
            med = np.median(np.abs(r[m]))
            sig = 1.4826 * max(med, 1e-3)
            cov = np.linalg.inv(JtWJ) * sig ** 2
            dof_sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
        return self.x, dof_sigma, cost
