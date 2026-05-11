# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Online/windowed CoWTracker for long first-frame anchored videos."""

from __future__ import annotations

import torch

from cowtracker.inference.windowed import WindowedInference
from cowtracker.models.cowtracker import CoWTracker


FREEZE_CONFIG_KEYS = {
    "freeze_vggt",
    "freeze_aggregator",
    "freeze_feature_extractor",
    "freeze_tracking_head",
}


class CoWTrackerOnline(CoWTracker):
    """CoWTracker with first-frame anchored windowed inference.

    This class inherits the regular CoWTracker modules directly. Long videos are
    processed as first-frame anchored windows with additional memory frames,
    matching the standalone CoWTrackerWindowed strategy while preserving the
    regular CoWTracker checkpoint layout.
    """

    def __init__(
        self,
        window_len: int = 8,
        window_stride: int | None = None,
        num_memory_frames: int = 10,
        merge_mode: str = "overwrite",
        **cow_tracker_kwargs,
    ) -> None:
        resolved_window_len = int(window_len)
        resolved_window_stride = (
            int(window_stride) if window_stride is not None else max(1, resolved_window_len // 2)
        )
        resolved_num_memory_frames = int(num_memory_frames)
        resolved_merge_mode = str(merge_mode).lower().strip()

        if resolved_window_len <= 0:
            raise ValueError("window_len must be a positive integer.")
        if resolved_window_stride <= 0:
            raise ValueError("window_stride must be a positive integer.")
        if resolved_window_stride > resolved_window_len:
            raise ValueError("window_stride must be <= window_len to avoid temporal gaps.")
        if resolved_num_memory_frames < 0:
            raise ValueError("num_memory_frames must be non-negative.")
        if resolved_merge_mode != "overwrite":
            raise ValueError("Only merge_mode='overwrite' is supported for now.")

        super().__init__(**cow_tracker_kwargs)
        self.window_len = resolved_window_len
        self.window_stride = resolved_window_stride
        self.num_memory_frames = resolved_num_memory_frames
        self.merge_mode = resolved_merge_mode
        self.windowed = WindowedInference(
            window_len=self.window_len,
            stride=self.window_stride,
            num_memory_frames=self.num_memory_frames,
        )
        self.last_windows: list[tuple[int, int]] = []
        print(
            "CoWTrackerOnline initialized: "
            f"window_len={self.window_len}, window_stride={self.window_stride}, "
            f"num_memory_frames={self.num_memory_frames}, merge_mode={self.merge_mode}"
        )

    def forward(
        self,
        video: torch.Tensor,
        queries: torch.Tensor = None,
        return_all_iters: bool = False,
    ) -> dict:
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"video must have shape [B,T,3,H,W] or [T,3,H,W], got {tuple(video.shape)}")

        b, total_frames, _, height, width = video.shape
        if total_frames <= self.window_len:
            self.last_windows = [(0, total_frames)]
            return self._forward_window_video(video, queries=queries, return_all_iters=return_all_iters)

        windows = self.compute_windows(total_frames)
        self.last_windows = windows

        images = video / 255.0
        first_frame = images[:, 0:1]
        accumulated = None
        iter_accumulated = None
        for window_idx, (start, end) in enumerate(windows):
            memory_indices = self.windowed.select_memory_frames(window_idx, start)
            frame_parts = [first_frame]
            if memory_indices:
                frame_parts.append(images[:, memory_indices])
            frame_parts.append(images[:, start:end])
            frames = torch.cat(frame_parts, dim=1)

            if not self.training:
                print(
                    f"Processing online window {window_idx + 1}/{len(windows)}: "
                    f"frames [{start}, {end})"
                )
                if memory_indices:
                    print(f"  Memory frames: {memory_indices}")

            features = self._extract_window_features(frames)
            first_frame_features = features[:, 0:1]
            num_memory = len(memory_indices)
            pred = self.tracking_head(
                features[:, 1:],
                image_size=(height, width),
                first_frame_features=first_frame_features,
                return_all_iters=return_all_iters,
            )
            window_pred = {
                "track": pred["track"][:, num_memory:],
                "vis": pred["vis"][:, num_memory:],
                "conf": pred["conf"][:, num_memory:],
            }
            if accumulated is None:
                accumulated = self._init_accumulated(window_pred, b, total_frames, height, width)
                if return_all_iters and "track_iters" in pred:
                    iter_accumulated = self._init_iter_accumulated(
                        pred,
                        b,
                        total_frames,
                        height,
                        width,
                    )

            self.windowed.merge_predictions(window_idx, start, end, window_pred, accumulated)
            if iter_accumulated is not None:
                window_iter_pred = {
                    "track_iters": [track[:, num_memory:] for track in pred["track_iters"]],
                    "vis_iters": [vis[:, num_memory:] for vis in pred["vis_iters"]],
                    "conf_iters": [conf[:, num_memory:] for conf in pred["conf_iters"]],
                }
                self._merge_window_iter_predictions(window_idx, start, end, window_iter_pred, iter_accumulated)

            if not self.training and torch.cuda.is_available():
                del features, pred
                torch.cuda.empty_cache()

        if iter_accumulated is not None:
            accumulated["track_iters"] = iter_accumulated["track_iters"]
            accumulated["vis_iters"] = iter_accumulated["vis_iters"]
            accumulated["conf_iters"] = iter_accumulated["conf_iters"]
        if not self.training:
            accumulated["images"] = video / 255.0
        return accumulated

    def _forward_window_video(
        self,
        video: torch.Tensor,
        queries: torch.Tensor = None,
        return_all_iters: bool = False,
    ) -> dict:
        return super().forward(video, queries=queries, return_all_iters=return_all_iters)

    def _extract_window_features(self, frames: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "vggt":
            tokens, patch_idx = self.aggregator(frames)
            return self.feature_extractor(tokens, frames, patch_idx)
        return self.feature_extractor(frames)

    def compute_windows(self, total_frames: int) -> list[tuple[int, int]]:
        return self.windowed.compute_windows(total_frames)

    @staticmethod
    def _init_accumulated(
        pred: dict,
        batch_size: int,
        total_frames: int,
        height: int,
        width: int,
    ) -> dict:
        return {
            "track": pred["track"].new_zeros((batch_size, total_frames, height, width, 2)),
            "vis": pred["vis"].new_zeros((batch_size, total_frames, height, width)),
            "conf": pred["conf"].new_zeros((batch_size, total_frames, height, width)),
        }

    @staticmethod
    def _init_iter_accumulated(
        pred: dict,
        batch_size: int,
        total_frames: int,
        height: int,
        width: int,
    ) -> dict:
        return {
            "track_iters": [
                track.new_zeros((batch_size, total_frames, height, width, 2))
                for track in pred["track_iters"]
            ],
            "vis_iters": [
                vis.new_zeros((batch_size, total_frames, height, width))
                for vis in pred["vis_iters"]
            ],
            "conf_iters": [
                conf.new_zeros((batch_size, total_frames, height, width))
                for conf in pred["conf_iters"]
            ],
        }

    def _merge_window_iter_predictions(
        self,
        window_idx: int,
        window_start: int,
        window_end: int,
        window_pred: dict,
        accumulated: dict,
    ) -> None:
        window_len = window_end - window_start
        start_offset = 0
        if window_idx > 0 and self.window_stride < self.window_len:
            overlap_len = min(self.window_len - self.window_stride, window_len)
            if overlap_len >= window_len:
                return
            start_offset = overlap_len

        for key in ("track_iters", "vis_iters", "conf_iters"):
            for dst, src in zip(accumulated[key], window_pred[key]):
                dst[:, window_start + start_offset : window_end] = src[:, start_offset:window_len]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = None,
        window_len: int = 8,
        window_stride: int | None = None,
        num_memory_frames: int = 10,
        merge_mode: str = "overwrite",
        device: str = "cuda",
        dtype=torch.bfloat16,
        **cow_tracker_kwargs,
    ):
        ckpt = cls._load_checkpoint(checkpoint_path)
        model_kwargs = dict(cow_tracker_kwargs)
        if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
            model_kwargs = {
                **_model_kwargs_from_config(ckpt["config"]),
                **model_kwargs,
            }

        model = cls(
            window_len=window_len,
            window_stride=window_stride,
            num_memory_frames=num_memory_frames,
            merge_mode=merge_mode,
            **model_kwargs,
        )
        state_dict = _extract_state_dict(ckpt)
        state_dict = _normalize_state_dict_keys(state_dict)

        legacy_prefixes = [
            "tracking_head.feature_extractor.",
            "tracking_head.aggregator.",
            "tracking_head.fnet.",
        ]
        if any(k.startswith(prefix) for k in state_dict for prefix in legacy_prefixes):
            print("Detected legacy checkpoint format, remapping keys...")
            state_dict = cls._remap_legacy_state_dict(state_dict)

        if all(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load message: {msg}")

        model = model.to(device).to(dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print("Model loaded successfully!")
        return model


def _model_kwargs_from_config(cfg: dict) -> dict:
    model_cfg = dict(cfg.get("model", {}))
    for key in FREEZE_CONFIG_KEYS:
        model_cfg.pop(key, None)
    return model_cfg


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    raise ValueError("Checkpoint must be a state_dict-like dict.")


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict
