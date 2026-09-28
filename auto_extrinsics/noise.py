# -*- coding: utf-8 -*-
"""Mounting-noise injection (rotation only, translation stays at factory GT).

The mounting error is FIXED per run: sampled once and applied to every
frame/sweep -- the online-calibration model used throughout this repo.
Monte-Carlo protocols vary the draw across runs via the seed only.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def sample_mounting_noise(rng, mag_deg, with_yaw=False):
    """Cn = Rz(psi) @ Ry(theta) @ Rx(phi), angles ~ U(-mag, +mag) deg.

    Matches extrinsic-xyz order of the existing scripts
    (Rotation.from_euler('xyz', [...]) == Rz @ Ry @ Rx)."""
    a = [np.radians(rng.uniform(-mag_deg, mag_deg)) for _ in
         ((0, 1, 2) if with_yaw else (0, 1))]
    if not with_yaw:
        a.append(0.0)
    return Rotation.from_euler('xyz', a).as_matrix()


def sim_coarse_init(R_gt, rng, yaw_err, tilt_resid_deg=0.5):
    """Orientation AFTER a good gravity-anchored coarse pass: tilt noise
    removed down to a small residual, yaw error untouched.

    R_init = Rz(yaw_err) @ R_eps @ R_gt with R_eps a small random tilt --
    the state the fine stage actually receives in the full cascade."""
    eps = Rotation.from_euler('xyz', [
        np.radians(rng.uniform(-tilt_resid_deg, tilt_resid_deg)),
        np.radians(rng.uniform(-tilt_resid_deg, tilt_resid_deg)),
        0.0]).as_matrix()
    yaw = Rotation.from_euler('z', yaw_err).as_matrix()
    return yaw @ eps @ R_gt


def residual_rotvec_deg(R_est, R_gt):
    """Small-angle error rotvec in the EGO frame, degrees -- avoids the
    camera-mounting euler singularity (roll~=-90) noted in the coarse
    pipeline's lessons; components read as (roll, pitch, yaw) errors about
    the ego axes for small residuals."""
    C = R_est @ R_gt.T
    return np.degrees(Rotation.from_matrix(C).as_rotvec())


def geodesic_deg(R_est, R_gt):
    return float(np.linalg.norm(residual_rotvec_deg(R_est, R_gt)))
