#!/usr/bin/env python3
"""Small, ROS-free MFCalib-style LiDAR-camera calibration runner.

The implementation follows the public MFCalib pipeline: Canny image edges,
depth-discontinuity LiDAR edges, nearest edge matching, and coarse-to-fine
SE(3) least-squares refinement. It intentionally keeps the input contract
simple so it can be used with exported nuScenes or ordinary files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        points = np.load(path)
    elif path.suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        width = 5 if raw.size % 5 == 0 else 4 if raw.size % 4 == 0 else 3
        points = raw.reshape(-1, width)
    elif path.suffix == ".pcd":
        with path.open("rb") as f:
            data = f.read()
        header_end = data.find(b"DATA ")
        if header_end < 0:
            raise ValueError("PCD header has no DATA field")
        header_end = data.find(b"\n", header_end) + 1
        header = data[:header_end].decode("ascii", errors="ignore").splitlines()
        fields = next(x.split()[1:] for x in header if x.startswith("FIELDS "))
        count = int(next(x.split()[1] for x in header if x.startswith("POINTS ")))
        if next(x.split()[1] for x in header if x.startswith("DATA ")) == "ascii":
            points = np.loadtxt(data[header_end:].splitlines(), dtype=np.float64, max_rows=count)
        else:
            sizes = [int(x) for x in next(x.split()[1:] for x in header if x.startswith("SIZE "))]
            types = next(x.split()[1:] for x in header if x.startswith("TYPE "))
            if sizes != [4] * len(fields) or types != ["F"] * len(fields):
                raise ValueError("Only float32 PCD fields are supported")
            points = np.frombuffer(data[header_end:], dtype=np.float32, count=count * len(fields)).reshape(count, len(fields))
    else:
        raise ValueError(f"unsupported point-cloud format: {path}")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("point cloud must have at least x,y,z columns")
    return np.asarray(points[:, :3], dtype=np.float64)


def load_cloud(path: Path):
    if path.suffix == ".npy": raw = np.load(path)
    elif path.suffix == ".bin":
        data = np.fromfile(path, dtype=np.float32); width = 5 if data.size % 5 == 0 else 4 if data.size % 4 == 0 else 3; raw = data.reshape(-1, width)
    else:
        raw = load_points(path)
    raw = np.asarray(raw)
    return np.asarray(raw[:, :3], float), (np.asarray(raw[:, 3], float) if raw.shape[1] > 3 else np.zeros(len(raw)))


class OpenCVYamlLoader(yaml.SafeLoader):
    pass


def _opencv_matrix(loader, node):
    value = loader.construct_mapping(node, deep=True)
    return np.asarray(value["data"], dtype=float).reshape(int(value["rows"]), int(value["cols"]))


OpenCVYamlLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", _opencv_matrix)


def load_config(path: Path, camera_config: Path | None = None):
    cfg = yaml.load(path.read_text(), Loader=OpenCVYamlLoader)
    cam = cfg.get("camera", {})
    if camera_config:
        camera_cfg = yaml.load(camera_config.read_text(), Loader=OpenCVYamlLoader)
        cam = camera_cfg.get("camera", camera_cfg)
        if "CameraMat" in camera_cfg: cam["camera_matrix"] = camera_cfg["CameraMat"]
        if "DistCoeffs" in camera_cfg: cam["dist_coeffs"] = camera_cfg["DistCoeffs"]
    K = np.asarray(cam.get("camera_matrix", cam.get("K", cfg.get("CameraMat"))), dtype=float).reshape(3, 3)
    dist = np.asarray(cam.get("dist_coeffs", cam.get("dist", cfg.get("DistCoeffs", [0, 0, 0, 0, 0]))), dtype=float).reshape(-1)
    T = np.asarray(cfg.get("ExtrinsicMat", cfg.get("extrinsic", np.eye(4))), dtype=float).reshape(4, 4)
    return K, dist, T


def se3(x):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("ZYX", x[:3]).as_matrix()
    T[:3, 3] = x[3:]
    return T


def project(points, T, K, shape, dist=None):
    q = points @ T[:3, :3].T + T[:3, 3]
    good = q[:, 2] > 0.1
    if dist is not None and np.any(np.asarray(dist)):
        uv, _ = cv2.projectPoints(q[good].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.asarray(dist))
        uv = uv.reshape(-1, 2)
    else:
        uv = (q[good] @ K.T)
        uv = uv[:, :2] / uv[:, 2:3]
    h, w = shape[:2]
    good2 = (uv[:, 0] >= 1) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 1)
    return uv[good2], q[good][good2]


def image_edges(image, min_component=40):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edge = cv2.Canny(gray, 20, 60)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((edge > 0).astype(np.uint8), 8)
    keep = np.zeros_like(edge)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_component:
            keep[labels == i] = 255
    ys, xs = np.nonzero(keep)
    return np.column_stack([xs, ys]).astype(float), keep


def lidar_sphere_edges(points, intensity, angular_resolution, depth_jump, max_points):
    """Port lidarToSphere: intensity projection, Canny, then depth check."""
    radius = np.linalg.norm(points, axis=1)
    valid = radius > 0.1
    p, inten, radius = points[valid], intensity[valid], radius[valid]
    theta = np.arccos(np.clip(p[:, 2] / radius, -1, 1))
    phi = np.arctan2(p[:, 1], p[:, 0])
    h = max(64, int(np.ceil((theta.max() - theta.min()) / angular_resolution)) + 3)
    w = max(128, int(np.ceil(2 * np.pi / angular_resolution)) + 3)
    u = np.clip(((theta.max() - theta) / angular_resolution).astype(int), 0, h - 1)
    v = ((phi + np.pi) / angular_resolution).astype(int) % w
    depth = np.full((h, w), np.inf); count = np.zeros((h, w), np.int32); mean_i = np.zeros((h, w), np.float32)
    for i, (yy, xx) in enumerate(zip(u, v)):
        depth[yy, xx] = min(depth[yy, xx], radius[i]); mean_i[yy, xx] += inten[i]; count[yy, xx] += 1
    mean_i[count > 0] /= count[count > 0]
    norm = cv2.normalize(mean_i, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edge = cv2.Canny(cv2.GaussianBlur(norm, (5, 5), 0), 20, 60)
    selected = set()
    for yy, xx in zip(*np.nonzero(edge)):
        local = depth[max(0, yy - 2):yy + 3, max(0, xx - 2):xx + 3]
        finite = local[np.isfinite(local)]
        if len(finite) and (finite.max() - finite.min() > depth_jump or len(finite) == 1):
            selected.update(np.flatnonzero((u >= max(0, yy - 1)) & (u <= yy + 1) & (np.abs(v - xx) <= 1)).tolist())
    out = p[np.asarray(sorted(selected), dtype=int)] if selected else np.empty((0, 3))
    if len(out) > max_points: out = out[np.linspace(0, len(out) - 1, max_points).astype(int)]
    return out, edge


def voxel_plane_edges(points, voxel_size, ransac_threshold, min_points, max_planes, max_points):
    """Approximate initVoxel/LiDAREdgeExtraction/calcLine with NumPy RANSAC."""
    bins = np.floor(points / voxel_size).astype(np.int64)
    result = []
    for key in np.unique(bins, axis=0):
        cloud = points[np.all(bins == key, axis=1)]
        if len(cloud) < min_points: continue
        planes = []; remaining = cloud.copy()
        for _ in range(max_planes):
            if len(remaining) < min_points: break
            best = None
            rng = np.random.default_rng(int(np.dot(key, [73856093, 19349663, 83492791]) & 0xffffffff))
            for _ in range(80):
                tri = remaining[rng.choice(len(remaining), 3, replace=False)]
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); norm = np.linalg.norm(n)
                if norm < 1e-8: continue
                n /= norm; d = -n @ tri[0]; ids = np.abs(remaining @ n + d) < ransac_threshold
                if best is None or ids.sum() > best[0]: best = (ids.sum(), ids, n, d)
            if best is None or best[0] < min_points: break
            ids, n, d = best[1:]
            plane = remaining[ids]; planes.append((plane, n, plane.mean(0))); remaining = remaining[~ids]
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                a, n1, c1 = planes[i]; b, n2, c2 = planes[j]
                direction = np.cross(n1, n2); norm = np.linalg.norm(direction)
                if norm < 0.15: continue
                direction /= norm
                # Points close to both planes are the finite line-like support.
                support = np.vstack([a, b]); dist1 = np.abs((support - c1) @ n1); dist2 = np.abs((support - c2) @ n2)
                line = support[(dist1 < ransac_threshold * 2) & (dist2 < ransac_threshold * 2)]
                if len(line): result.append(line)
    out = np.concatenate(result) if result else np.empty((0, 3))
    if len(out) > max_points: out = out[np.linspace(0, len(out) - 1, max_points).astype(int)]
    return out


def lidar_edges(points, voxel=0.08, jump=0.35, max_points=30000):
    # Rasterize nearest range in the LiDAR azimuth/elevation image and select
    # points next to a strong range discontinuity.
    p = points[np.linalg.norm(points, axis=1) > 1.0]
    az = np.arctan2(p[:, 1], p[:, 0])
    el = np.arctan2(p[:, 2], np.linalg.norm(p[:, :2], axis=1))
    ix = np.floor(az / voxel).astype(int)
    iy = np.floor(el / voxel).astype(int)
    cells = {}
    for i, key in enumerate(zip(ix, iy)):
        r = np.linalg.norm(p[i])
        if key not in cells or r < cells[key][0]: cells[key] = (r, i)
    out = []
    for (x, y), (r, i) in cells.items():
        for nb in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if nb in cells and abs(r - cells[nb][0]) > jump:
                out.append(i); break
    result = p[np.unique(out)] if out else p
    if len(result) > max_points:
        result = result[np.linspace(0, len(result) - 1, max_points).astype(int)]
    return result


def run(args):
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(args.image)
    K, dist, T0 = load_config(args.config, args.camera_config)
    points, intensity = load_cloud(args.points)
    img_uv, edge_img = image_edges(image, args.min_component)
    discontinuity, sphere_edge = lidar_sphere_edges(points, intensity, args.angular_resolution, args.depth_jump, args.max_lidar_edges)
    voxel_edge = voxel_plane_edges(points, args.voxel_size, args.ransac_threshold, args.plane_min_points, args.max_planes, args.max_lidar_edges)
    lidar = np.concatenate([voxel_edge, discontinuity]) if len(discontinuity) else voxel_edge
    if len(lidar) == 0: lidar = lidar_edges(points, args.lidar_voxel, args.depth_jump, args.max_lidar_edges)
    tree = cKDTree(img_uv)
    residual_size = min(500, max(20, len(lidar)))

    def residual(x, threshold):
        uv, xyz = project(lidar, se3(x), K, image.shape, dist)
        if len(uv) == 0: return np.ones(residual_size) * 1e3
        d, nn = tree.query(uv, k=min(5, len(img_uv)), distance_upper_bound=threshold)
        if d.ndim == 1: d, nn = d[:, None], nn[:, None]
        values = []
        for i in range(len(uv)):
            finite = nn[i][np.isfinite(d[i])]
            if len(finite) == 0: values.append(threshold); continue
            q = img_uv[finite]
            if len(q) >= 2:
                _, _, vt = np.linalg.svd(q - q.mean(0), full_matrices=False)
                tangent = vt[0]; normal = np.array([-tangent[1], tangent[0]])
                values.append(abs(float((uv[i] - q[0]) @ normal)))
            else: values.append(float(d[i, 0]))
        values = np.minimum(np.asarray(values), threshold)
        if len(values) >= residual_size: return np.sort(values)[:residual_size].astype(np.float64)
        return np.pad(values, (0, residual_size - len(values)), constant_values=float(threshold)).astype(np.float64)

    x = np.r_[Rotation.from_matrix(T0[:3, :3]).as_euler("ZYX"), T0[:3, 3]]
    before = float(np.median(residual(x, args.match_threshold)))
    stage_history = []
    for threshold in args.thresholds:
        current = float(np.median(residual(x, threshold)))
        rotation_limit = np.deg2rad(args.max_rotation_update_deg)
        bound = np.r_[np.full(3, rotation_limit), np.full(3, args.max_translation_update_m)]
        fit = least_squares(
            lambda z: residual(z, threshold), x, loss="soft_l1",
            f_scale=max(1.0, threshold / 3), max_nfev=args.max_nfev,
            bounds=(x - bound, x + bound),
        )
        candidate = float(np.median(residual(fit.x, threshold)))
        accepted = bool(fit.success and candidate < current - args.min_stage_improvement_px)
        if accepted: x = fit.x
        stage_history.append({"threshold_px": threshold, "before_median_px": current, "candidate_median_px": candidate, "accepted": accepted, "optimizer_success": bool(fit.success)})
    after = float(np.median(residual(x, args.match_threshold)))
    T = se3(x)
    uv, _ = project(lidar, T, K, image.shape, dist)
    vis = image.copy()
    vis[edge_img > 0] = (0, 180, 255)
    for u, v in uv.astype(int): cv2.circle(vis, (u, v), 1, (0, 255, 0), -1)
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "image_edges.png"), edge_img)
    cv2.imwrite(str(args.out / "lidar_sphere_edges.png"), sphere_edge)
    cv2.imwrite(str(args.out / "mfcalib_overlay.jpg"), vis)
    report = {"input_image": str(args.image), "input_points": str(args.points), "image_edges": int(len(img_uv)), "lidar_edges": int(len(lidar)), "lidar_sphere_discontinuity_edges": int(len(discontinuity)), "lidar_voxel_plane_edges": int(len(voxel_edge)), "median_pixel_error_before": before, "median_pixel_error_after": after, "stage_history": stage_history, "accepted_stages": int(sum(item["accepted"] for item in stage_history)), "extrinsic_lidar_to_camera": T.tolist()}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True); p.add_argument("--points", type=Path, required=True); p.add_argument("--config", type=Path, required=True); p.add_argument("--camera-config", type=Path); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-component", type=int, default=40); p.add_argument("--lidar-voxel", type=float, default=0.08); p.add_argument("--depth-jump", type=float, default=0.35); p.add_argument("--max-lidar-edges", type=int, default=30000)
    p.add_argument("--angular-resolution", type=float, default=0.003); p.add_argument("--voxel-size", type=float, default=1.0); p.add_argument("--ransac-threshold", type=float, default=0.02); p.add_argument("--plane-min-points", type=int, default=30); p.add_argument("--max-planes", type=int, default=8)
    p.add_argument("--match-threshold", type=float, default=20.0); p.add_argument("--thresholds", type=float, nargs="+", default=[20, 12, 8, 5, 3]); p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument("--max-rotation-update-deg", type=float, default=5.0); p.add_argument("--max-translation-update-m", type=float, default=0.30); p.add_argument("--min-stage-improvement-px", type=float, default=0.05)
    run(p.parse_args())
