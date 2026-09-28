# -*- coding: utf-8 -*-
"""Projection chain and per-frame geometry for the cross-modal fine stage.

Chain (spec section 2): lidar -> ego -> global -> ego(cam time) -> cam.
Per-frame geometry is cached; a pose update only needs the two current
rotation estimates to rebuild the single lidar->cam transform.

Dynamic-object masking uses nuScenes annotation boxes (global frame),
projected once with the BASE (coarse) poses -- masks are fixed evidence.
"""
import cv2
import numpy as np

DILATE_DYNAMIC = 15      # px, safety margin around projected boxes
DEPTH_MIN, DEPTH_MAX = 1.5, 80.0


def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class FrameGeom:
    def __init__(self, name, K, R_ec, t_ec, ego_c, R_le, t_le, ego_l,
                 img_shape):
        self.name = name
        self.K = K
        self.R_ec0, self.t_ec = R_ec.copy(), t_ec   # base (coarse) pose
        self.R_le0, self.t_le = R_le.copy(), t_le
        self.T_ge_c = ego_c                          # ego->global, cam time
        self.T_ge_l = ego_l                          # ego->global, lidar time
        self.shape = img_shape
        # evidence, filled by the runner: TEED prob map, gradient magnitude,
        # dynamic-object mask, soft-DT maps per sigma
        self.teed_prob = None
        self.gm = None
        self.dyn_mask = None
        self.soft_dt = {}

    # ---- transforms --------------------------------------------------------
    def T_lc(self, R_ec, R_le):
        """lidar->cam for the CURRENT pose estimates."""
        return np.linalg.inv(_T(R_ec, self.t_ec)) \
            @ np.linalg.inv(self.T_ge_c) @ self.T_ge_l @ _T(R_le, self.t_le)

    def T_gc_base(self):
        """global->cam with BASE camera pose (for masks/visualization)."""
        return np.linalg.inv(_T(self.R_ec0, self.t_ec)) \
            @ np.linalg.inv(self.T_ge_c)

    # ---- stacked sweeps ------------------------------------------------------
    def set_stacked(self, sweeps):
        """Stacked evidence cloud from ~0.5 s of sweeps, each bridged to the
        reference ego frame (ego at CAMERA time) through its OWN ego pose --
        densifies the cloud ~10x and removes the inter-sweep motion (the
        single-sweep 35 ms camera offset is bridged the same way).

        sweeps: list of dicts p_l [n,3] (raw points, that sweep's LIDAR
        frame -- R_le is being estimated so nothing is pre-rotated) and
        T_eg [4,4] (sweep ego -> global). G_s = inv(T_ge_c) @ T_eg is the
        fixed sweep-ego -> reference-ego bridge. Optional per-point
        'intensity' is concatenated in the same order."""
        self.stack_p = np.concatenate([s['p_l'] for s in sweeps])
        self.stack_sid = np.concatenate(
            [np.full(len(s['p_l']), k, np.int32)
             for k, s in enumerate(sweeps)])
        self.stack_G = np.stack(
            [np.linalg.inv(self.T_ge_c) @ s['T_eg'] for s in sweeps])
        if 'intensity' in sweeps[0]:
            self.stack_intensity = np.concatenate(
                [s['intensity'] for s in sweeps])

    def mark_intensity(self):
        return self.stack_intensity

    def project_stacked(self, R_ec, R_le, idx=None):
        """Project stacked points (all, or the given index subset) under
        current rotation estimates:
        p_ego_ref = G_s @ (R_le @ p_l + t_le); p_cam = R_ec^T (p_ego_ref -
        t_ec). Returns (uv, z, ok) like project()."""
        p_all = self.stack_p if idx is None else self.stack_p[idx]
        sid = self.stack_sid if idx is None else self.stack_sid[idx]
        p_s = (R_le @ p_all.T).T + self.t_le
        p_e = np.empty_like(p_s)
        for k, G in enumerate(self.stack_G):
            m = sid == k
            if m.any():
                p_e[m] = p_s[m] @ G[:3, :3].T + G[:3, 3]
        p_c = (p_e - self.t_ec) @ R_ec.T
        z = p_c[:, 2]
        uv = (self.K @ p_c.T).T[:, :2] / np.maximum(z[:, None], 1e-9)
        h, w = self.shape
        ok = (z > DEPTH_MIN) & (z < DEPTH_MAX)
        ok &= (uv[:, 0] > 0.5) & (uv[:, 0] < w - 1.5) \
            & (uv[:, 1] > 0.5) & (uv[:, 1] < h - 1.5)
        if self.dyn_mask is not None:
            ui = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
            vi = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
            ok &= self.dyn_mask[vi, ui] == 0
        return uv, z, ok

    def stacked_ego_ref(self):
        """Stacked points in the reference ego frame at the BASE lidar
        rotation (fixed evidence for extraction; not re-projected)."""
        p_s = (self.R_le0 @ self.stack_p.T).T + self.t_le
        p_e = np.empty_like(p_s)
        for k, G in enumerate(self.stack_G):
            m = self.stack_sid == k
            p_e[m] = p_s[m] @ G[:3, :3].T + G[:3, 3]
        return p_e

    # ---- projection --------------------------------------------------------
    def project(self, p_l, T_lc):
        """Returns (uv, z, ok); ok = depth range + in-bounds + not dynamic."""
        pc = (T_lc[:3, :3] @ p_l.T).T + T_lc[:3, 3]
        z = pc[:, 2]
        uv = (self.K @ pc.T).T[:, :2] / np.maximum(z[:, None], 1e-9)
        h, w = self.shape
        ok = (z > DEPTH_MIN) & (z < DEPTH_MAX)
        ok &= (uv[:, 0] > 0.5) & (uv[:, 0] < w - 1.5) \
            & (uv[:, 1] > 0.5) & (uv[:, 1] < h - 1.5)
        if self.dyn_mask is not None:
            ui = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
            vi = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
            ok &= self.dyn_mask[vi, ui] == 0
        return uv, z, ok


def build_dynamic_mask(frame, boxes):
    """boxes: list of dicts with 'T_bg' (4x4, box->global) and 'size' (3,)."""
    if not boxes:
        return None
    h, w = frame.shape
    mask = np.zeros((h, w), np.uint8)
    T_cg = frame.T_gc_base()
    for b in boxes:
        half = np.asarray(b['size'], float) / 2.0
        local = np.array([[a * half[0], c * half[1], e * half[2]]
                          for a in (-1, 1) for c in (-1, 1) for e in (-1, 1)])
        corners_g = local @ b['T_bg'][:3, :3].T + b['T_bg'][:3, 3]
        corners_c = (T_cg[:3, :3] @ corners_g.T).T + T_cg[:3, 3]
        z = corners_c[:, 2]
        if (z <= 0.2).any():
            continue
        uv = (frame.K @ corners_c.T).T[:, :2] / z[:, None]
        poly = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [cv2.convexHull(poly)], 1)
    if mask.any():
        return cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATE_DYNAMIC, DILATE_DYNAMIC)))
    return None
