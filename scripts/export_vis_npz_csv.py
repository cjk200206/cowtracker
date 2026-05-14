#!/usr/bin/env python3
"""Export CoWTracker visualization NPZ trajectories to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more visualization .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for CSV outputs. Defaults to each NPZ's parent directory.",
    )
    parser.add_argument(
        "--point_limit",
        type=int,
        default=None,
        help="Optional maximum number of points to export per file.",
    )
    return parser.parse_args()


def load_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    arr = data[key]
    if arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0]
    return arr


def compute_step(track: np.ndarray) -> np.ndarray:
    step = np.full(track.shape[:2], np.nan, dtype=np.float32)
    if track.shape[0] > 1:
        step[1:] = np.linalg.norm(track[1:] - track[:-1], axis=-1)
    return step


def compute_epe(pred_track: np.ndarray, gt_track: np.ndarray | None) -> np.ndarray | None:
    if gt_track is None:
        return None
    return np.linalg.norm(pred_track - gt_track, axis=-1).astype(np.float32)


def export_one(npz_path: Path, output_dir: Path, point_limit: int | None) -> Path:
    data = np.load(npz_path)
    pred_track = load_array(data, "pred_track")
    pred_vis = load_array(data, "pred_visibility")
    pred_conf = load_array(data, "pred_confidence")
    gt_track = load_array(data, "gt_track")
    gt_vis = load_array(data, "gt_visibility")
    query_xy = load_array(data, "query_xy")

    if pred_track is None:
        raise ValueError(f"{npz_path} does not contain pred_track.")

    num_frames, num_points = pred_track.shape[:2]
    if point_limit is not None:
        num_points = min(num_points, int(point_limit))
        pred_track = pred_track[:, :num_points]
        if pred_vis is not None:
            pred_vis = pred_vis[:, :num_points]
        if pred_conf is not None:
            pred_conf = pred_conf[:, :num_points]
        if gt_track is not None:
            gt_track = gt_track[:, :num_points]
        if gt_vis is not None:
            gt_vis = gt_vis[:, :num_points]
        if query_xy is not None:
            query_xy = query_xy[:num_points]

    step = compute_step(pred_track)
    epe = compute_epe(pred_track, gt_track)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{npz_path.stem}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "point_id",
                "pred_x",
                "pred_y",
                "pred_visible",
                "pred_confidence",
                "gt_x",
                "gt_y",
                "gt_visible",
                "query_x",
                "query_y",
                "step_from_prev",
                "epe_to_gt",
            ]
        )

        for t in range(num_frames):
            for p in range(num_points):
                row = [
                    t,
                    p,
                    float(pred_track[t, p, 0]),
                    float(pred_track[t, p, 1]),
                    "" if pred_vis is None else int(bool(pred_vis[t, p])),
                    "" if pred_conf is None else float(pred_conf[t, p]),
                    "" if gt_track is None else float(gt_track[t, p, 0]),
                    "" if gt_track is None else float(gt_track[t, p, 1]),
                    "" if gt_vis is None else int(bool(gt_vis[t, p])),
                    "" if query_xy is None else float(query_xy[p, 0]),
                    "" if query_xy is None else float(query_xy[p, 1]),
                    "" if np.isnan(step[t, p]) else float(step[t, p]),
                    "" if epe is None else float(epe[t, p]),
                ]
                writer.writerow(row)

    return csv_path


def main():
    args = parse_args()
    for input_path in args.inputs:
        npz_path = Path(input_path).expanduser()
        out_dir = Path(args.output_dir).expanduser() if args.output_dir else npz_path.parent
        csv_path = export_one(npz_path, out_dir, args.point_limit)
        print(f"Exported {npz_path} -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
