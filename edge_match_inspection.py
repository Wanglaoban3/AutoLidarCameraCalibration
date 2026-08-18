"""Visualize LiDAR/image edge correspondences after ICP+hand-eye coarse calibration."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from nuscenes_edge_demo import (
    CAMERAS, draw_segments, load_frame, point_to_line_with_index, project, se3,
)


def closest_on_segment(point, segment):
    a, b = segment[:2], segment[2:]
    d = b - a
    t = np.clip(np.dot(point - a, d) / (np.dot(d, d) + 1e-9), 0.0, 1.0)
    return a + t * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--out", default="/workspace/results/edge_match_inspection")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--coarse-json", required=True)
    ap.add_argument("--noise-rpy-deg", type=float, nargs=3, default=[3.0, -3.0, 4.0])
    ap.add_argument("--noise-translation-m", type=float, nargs=3, default=[0.06, -0.04, 0.08])
    ap.add_argument("--max-match-px", type=float, default=14.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.coarse_json).read_text())
    coarse = np.asarray(report["estimated_body_correction"], dtype=np.float64)
    manual_body = se3(np.r_[np.deg2rad(args.noise_rpy_deg), args.noise_translation_m])
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    samples, token = [], scene["first_sample_token"]
    while token and len(samples) < args.frames:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]

    metrics = {"coarse_body_correction": coarse.tolist(), "cameras": {}}
    for camera in CAMERAS:
        camera_metrics = []
        for frame_id, sample in enumerate(samples):
            frame = load_frame(nusc, sample, camera, args.dataroot)
            T_manual = np.linalg.inv(frame["T_ego_camera"]) @ manual_body @ frame["T_ego_camera"]

            def transform_body(x):
                return np.linalg.inv(frame["T_ego_camera"]) @ se3(x) @ frame["T_ego_camera"] @ T_manual

            source, tangent = frame["pts"], frame["tangents"]
            uv, valid = project(source, transform_body(coarse), frame["K"], frame["shape"])
            distances, line_ids = point_to_line_with_index(
                uv[valid], frame["segments"], tangent[valid], max_angle_deg=40.0
            )
            accepted = np.zeros(len(source), dtype=bool)
            accepted[valid] = distances < args.max_match_px
            valid_ids = np.flatnonzero(valid)
            accepted_ids = np.flatnonzero(accepted)

            # Panel 1: all Canny/LSD image edges.
            image_edges = draw_segments(frame["image"], frame["segments"], (0, 210, 255), 2)
            # Panel 2: all projected LiDAR edge candidates (red), accepted (green).
            candidates = frame["image"].copy()
            for idx in valid_ids:
                cv2.circle(candidates, tuple(uv[idx].astype(int)), 2, (0, 0, 255), -1, cv2.LINE_AA)
            for idx in accepted_ids:
                cv2.circle(candidates, tuple(uv[idx].astype(int)), 3, (0, 255, 0), -1, cv2.LINE_AA)
            # Panel 3: accepted line association and residual connector.
            matches = draw_segments(frame["image"], frame["segments"][np.unique(line_ids[distances < args.max_match_px])], (0, 220, 255), 2)
            accepted_distances = []
            for idx in accepted_ids:
                local = np.flatnonzero(valid_ids == idx)
                if not len(local):
                    continue
                line = frame["segments"][line_ids[local[0]]]
                q = closest_on_segment(uv[idx], line)
                accepted_distances.append(float(np.linalg.norm(uv[idx] - q)))
                p = tuple(uv[idx].astype(int))
                cv2.circle(matches, p, 3, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.line(matches, p, tuple(q.astype(int)), (255, 0, 255), 1, cv2.LINE_AA)

            def label(image, text):
                cv2.rectangle(image, (0, 0), (min(900, image.shape[1]), 38), (0, 0, 0), -1)
                cv2.putText(image, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
                return image

            panels = [
                label(image_edges, "1 image Canny + LSD segments"),
                label(candidates, "2 LiDAR candidates: red=all, green=accepted"),
                label(matches, "3 accepted matches: magenta=point-to-line residual"),
            ]
            h = min(p.shape[0] for p in panels) // 2
            w = min(p.shape[1] for p in panels) // 2
            small = [cv2.resize(p, (w, h)) for p in panels]
            sheet = np.hstack([small[0], small[1], small[2]])
            cv2.imwrite(str(out / f"{camera}_{frame_id:02d}_matches.jpg"), sheet)
            cv2.imwrite(str(out / f"{camera}_{frame_id:02d}_image_edges.jpg"), panels[0])
            cv2.imwrite(str(out / f"{camera}_{frame_id:02d}_lidar_candidates.jpg"), panels[1])
            cv2.imwrite(str(out / f"{camera}_{frame_id:02d}_accepted_matches.jpg"), panels[2])
            camera_metrics.append({
                "frame": frame_id, "source_points": int(len(source)),
                "projected_candidates": int(valid.sum()), "accepted_matches": int(accepted.sum()),
                "acceptance_ratio": float(accepted.sum() / max(1, valid.sum())),
                "median_point_to_line_px": float(np.median(accepted_distances)) if accepted_distances else None,
                "p90_point_to_line_px": float(np.percentile(accepted_distances, 90)) if accepted_distances else None,
            })
        metrics["cameras"][camera] = camera_metrics
    (out / "report.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
