#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize CoWTracker predictions from a trained checkpoint."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Force using only the first GPU for visualization

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cowtracker.datasets import cow_rgb_collate
from cowtracker.training.losses import sample_dense_predictions
from cowtracker.utils.visualization import TrackVisualizer
from scripts.train_cowtracker import (
    autocast_dtype,
    build_dataset,
    ensure_vggt_available,
    extract_state_dict,
    load_config,
    normalize_state_dict_keys,
)


FREEZE_CONFIG_KEYS = {
    "freeze_vggt",
    "freeze_aggregator",
    "freeze_feature_extractor",
    "freeze_tracking_head",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/train_rgb_tapformer_test.yaml")
    parser.add_argument("--output_dir", type=str, default="output/vis_cowtracker")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--full_sequence", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--window_len", type=int, default=None)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--num_points", type=int, default=128)
    parser.add_argument("--point_source", choices=("gt", "grid"), default="gt")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold", type=float, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--linewidth", type=int, default=2)
    parser.add_argument("--tracks_leave_trace", type=int, default=-1)
    parser.add_argument("--show_first_frame", type=int, default=0)
    parser.add_argument("--draw_gt", action="store_true")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--random_temporal_crop", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str | Path):
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt


def model_kwargs_from_config(cfg: dict) -> dict:
    model_cfg = dict(cfg.get("model", {}))
    for key in FREEZE_CONFIG_KEYS:
        model_cfg.pop(key, None)
    return model_cfg


def build_visualization_model(
    ckpt,
    cfg: dict,
    device: torch.device,
    use_online: bool = False,
    window_len: int | None = None,
    window_stride: int | None = None,
):
    ensure_vggt_available()
    from cowtracker.models.cowtracker import CoWTracker
    from cowtracker.models.cowtracker_online import CoWTrackerOnline

    model_cfg_source = ckpt.get("config", cfg) if isinstance(ckpt, dict) else cfg
    model_kwargs = model_kwargs_from_config(model_cfg_source)
    if use_online:
        resolved_window_len = int(window_len or cfg.get("data", {}).get("seq_len", 8))
        model = CoWTrackerOnline(
            window_len=resolved_window_len,
            window_stride=window_stride,
            **model_kwargs,
        ).to(device)
    else:
        model = CoWTracker(**model_kwargs).to(device)
    state_dict = extract_state_dict(ckpt)
    state_dict = normalize_state_dict_keys(state_dict)

    legacy_prefixes = [
        "tracking_head.feature_extractor.",
        "tracking_head.aggregator.",
        "tracking_head.fnet.",
    ]
    if any(k.startswith(prefix) for k in state_dict for prefix in legacy_prefixes):
        state_dict = CoWTracker._remap_legacy_state_dict(state_dict)
    if state_dict and all(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=False)
    print(
        "Loaded checkpoint "
        f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})",
        flush=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_visualization_dataset(cfg: dict, random_temporal_crop: bool, seq_len: int | None = None):
    vis_cfg = copy.deepcopy(cfg)
    vis_cfg.setdefault("data", {})
    vis_cfg["data"]["random_temporal_crop"] = bool(random_temporal_crop)
    if seq_len is not None:
        if int(seq_len) <= 0:
            raise ValueError("--seq_len must be a positive integer.")
        vis_cfg["data"]["seq_len"] = int(seq_len)
    return build_dataset(vis_cfg)


def full_sequence_len(dataset, sample_index: int) -> int:
    index = int(sample_index) % len(dataset)
    seq_path = dataset.seq_paths[index]
    frame_paths = dataset._frame_paths(seq_path / "raw")
    annot = np.load(seq_path / "annotations.npy", allow_pickle=True).item()
    traj = np.asarray(annot["target_points"])
    occluded = np.asarray(annot["occluded"])
    return int(min(len(frame_paths), traj.shape[1], occluded.shape[1]))


def load_one_batch(dataset, sample_index: int, num_workers: int):
    index = int(sample_index) % len(dataset)
    if num_workers > 0:
        loader = DataLoader(
            Subset(dataset, [index]),
            batch_size=1,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=cow_rgb_collate,
        )
        batch, gotit = next(iter(loader))
    else:
        sample, ok = dataset[index]
        batch, gotit = cow_rgb_collate([(sample, ok)])
    if not bool(gotit.all()):
        raise RuntimeError(f"Failed to load sample index {index}: gotit={gotit.tolist()}")
    return batch, index


def move_batch_to_device(batch, device: torch.device):
    batch.video = batch.video.to(device, non_blocking=True)
    batch.trajectory = batch.trajectory.to(device, non_blocking=True)
    batch.visibility = batch.visibility.to(device, non_blocking=True)
    batch.valid = batch.valid.to(device, non_blocking=True)
    return batch


def select_gt_points(batch, num_points: int):
    first_visible = batch.visibility[0, 0] > 0.5
    first_valid = batch.valid[0, 0] > 0.5
    point_mask = first_visible & first_valid
    selected = torch.nonzero(point_mask, as_tuple=False).flatten()
    if selected.numel() == 0:
        raise RuntimeError("No valid first-frame GT points are available for visualization.")
    selected = selected[: int(num_points)]
    query_xy = batch.trajectory[:, 0, selected]
    gt_tracks = batch.trajectory[:, :, selected]
    gt_visibility = batch.visibility[:, :, selected] > 0.5
    return query_xy, gt_tracks, gt_visibility


def select_grid_points(batch, num_points: int):
    _, _, _, height, width = batch.video.shape
    count = int(max(1, num_points))
    cols = int(np.ceil(np.sqrt(count * width / max(height, 1))))
    rows = int(np.ceil(count / max(cols, 1)))
    xs = torch.linspace(0, width - 1, steps=cols, device=batch.video.device)
    ys = torch.linspace(0, height - 1, steps=rows, device=batch.video.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    query_xy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[:count]
    return query_xy[None], None, None


def sample_sparse_predictions(predictions: dict, query_xy: torch.Tensor):
    pred_track = sample_dense_predictions(predictions["track"], query_xy)
    pred_vis = sample_dense_predictions(predictions["vis"][..., None], query_xy).squeeze(-1)
    pred_conf = sample_dense_predictions(predictions["conf"][..., None], query_xy).squeeze(-1)
    return pred_track, pred_vis, pred_conf


def output_stem(checkpoint: str | Path, batch, sample_index: int) -> str:
    ckpt_name = Path(checkpoint).stem
    seq_name = batch.seq_name[0] if isinstance(batch.seq_name, list) else str(batch.seq_name)
    return f"{ckpt_name}_{seq_name}_sample{sample_index:04d}"


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    ckpt = load_checkpoint(args.checkpoint)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = args.precision or str(cfg.get("train", {}).get("precision", "bf16")).lower()
    amp_dtype = autocast_dtype(precision)
    use_amp = amp_dtype is not None and device.type == "cuda"

    seq_len_override = args.seq_len
    dataset = build_visualization_dataset(
        cfg,
        random_temporal_crop=args.random_temporal_crop,
        seq_len=seq_len_override,
    )
    if args.full_sequence and seq_len_override is None:
        seq_len_override = full_sequence_len(dataset, args.sample_index)
        dataset = build_visualization_dataset(
            cfg,
            random_temporal_crop=args.random_temporal_crop,
            seq_len=seq_len_override,
        )
    batch, resolved_index = load_one_batch(dataset, args.sample_index, args.num_workers)
    batch = move_batch_to_device(batch, device)
    input_frames = int(batch.video.shape[1])

    use_online = bool(args.online or args.full_sequence or args.window_len is not None)
    model = build_visualization_model(
        ckpt,
        cfg,
        device,
        use_online=use_online,
        window_len=args.window_len,
        window_stride=args.window_stride,
    )
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
            predictions = model(batch.video, return_all_iters=False)

    if args.point_source == "gt":
        query_xy, gt_tracks, gt_visibility = select_gt_points(batch, args.num_points)
    else:
        query_xy, gt_tracks, gt_visibility = select_grid_points(batch, args.num_points)

    pred_track, pred_vis, pred_conf = sample_sparse_predictions(predictions, query_xy)
    pred_visibility = pred_vis > float(args.vis_threshold)
    if args.conf_threshold is not None:
        pred_visibility = pred_visibility & (pred_conf > float(args.conf_threshold))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.checkpoint, batch, resolved_index)

    visualizer = TrackVisualizer(
        save_dir=str(output_dir),
        fps=args.fps,
        linewidth=args.linewidth,
        tracks_leave_trace=args.tracks_leave_trace,
        show_first_frame=args.show_first_frame,
    )
    rendered = visualizer.visualize(
        batch.video,
        pred_track,
        pred_visibility,
        filename=f"{stem}.mp4",
        query_frame=0,
        gt_tracks=gt_tracks if args.draw_gt else None,
        gt_visibility=gt_visibility if args.draw_gt else None,
    )

    if args.save_npz:
        arrays = {
            "query_xy": query_xy.detach().cpu().numpy(),
            "pred_track": pred_track.detach().cpu().numpy(),
            "pred_visibility": pred_visibility.detach().cpu().numpy(),
            "pred_confidence": pred_conf.detach().cpu().numpy(),
        }
        if gt_tracks is not None:
            arrays["gt_track"] = gt_tracks.detach().cpu().numpy()
        if gt_visibility is not None:
            arrays["gt_visibility"] = gt_visibility.detach().cpu().numpy()
        np.savez_compressed(output_dir / f"{stem}.npz", **arrays)

    seq_name = batch.seq_name[0] if isinstance(batch.seq_name, list) else str(batch.seq_name)
    print(
        f"seq_name={seq_name} sample_index={resolved_index} "
        f"input_frames={input_frames} "
        f"show_first_frame={args.show_first_frame} "
        f"rendered_frames={rendered.shape[0]} "
        f"online={use_online} "
        f"window_len={getattr(model, 'window_len', None)} "
        f"window_stride={getattr(model, 'window_stride', None)} "
        f"num_windows={len(getattr(model, 'last_windows', []))}",
        flush=True,
    )
    print(f"Rendered frames: {tuple(rendered.shape)}", flush=True)
    print(f"Saved visualization under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
