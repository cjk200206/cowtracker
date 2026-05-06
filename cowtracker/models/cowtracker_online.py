# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Online/windowed CoWTracker for long first-frame anchored videos."""

from __future__ import annotations

import torch

from cowtracker.models.cowtracker import CoWTracker


FREEZE_CONFIG_KEYS = {
    "freeze_vggt",
    "freeze_aggregator",
    "freeze_feature_extractor",
    "freeze_tracking_head",
}


class CoWTrackerOnline(CoWTracker):
    """CoWTracker with TAPFormer-style first-frame anchored sliding windows.

    This class inherits the regular CoWTracker modules directly. For windows
    after the first one, it prepends global frame 0 as the anchor and writes the
    non-anchor predictions back into the full sequence output.
    """

    def __init__(
        self,
        window_len: int = 8,
        window_stride: int | None = None,
        merge_mode: str = "overwrite",
        **cow_tracker_kwargs,
    ) -> None:
        resolved_window_len = int(window_len)
        resolved_window_stride = (
            int(window_stride) if window_stride is not None else max(1, resolved_window_len // 2)
        )
        resolved_merge_mode = str(merge_mode).lower().strip()

        if resolved_window_len <= 0:
            raise ValueError("window_len must be a positive integer.")
        if resolved_window_stride <= 0:
            raise ValueError("window_stride must be a positive integer.")
        if resolved_window_stride > resolved_window_len:
            raise ValueError("window_stride must be <= window_len to avoid temporal gaps.")
        if resolved_merge_mode != "overwrite":
            raise ValueError("Only merge_mode='overwrite' is supported for now.")

        super().__init__(**cow_tracker_kwargs)
        self.window_len = resolved_window_len
        self.window_stride = resolved_window_stride
        self.merge_mode = resolved_merge_mode
        self.last_windows: list[tuple[int, int]] = []
        print(
            "CoWTrackerOnline initialized: "
            f"window_len={self.window_len}, window_stride={self.window_stride}, "
            f"merge_mode={self.merge_mode}"
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

        accumulated = None
        iter_accumulated = None
        for window_idx, (start, end) in enumerate(windows):
            if window_idx == 0:
                window_video = video[:, start:end]
                output_offset = 0
            else:
                window_video = torch.cat([video[:, 0:1], video[:, start:end]], dim=1)
                output_offset = 1

            if not self.training:
                print(
                    f"Processing online window {window_idx + 1}/{len(windows)}: "
                    f"frames [{start}, {end})"
                )

            pred = self._forward_window_video(
                window_video,
                queries=queries,
                return_all_iters=return_all_iters,
            )
            if accumulated is None:
                accumulated = self._init_accumulated(pred, b, total_frames, height, width)
                if return_all_iters and "track_iters" in pred:
                    iter_accumulated = self._init_iter_accumulated(
                        pred,
                        b,
                        total_frames,
                        height,
                        width,
                    )

            self._write_window_predictions(
                accumulated,
                pred,
                start=start,
                end=end,
                output_offset=output_offset,
            )
            if iter_accumulated is not None:
                self._write_window_iter_predictions(
                    iter_accumulated,
                    pred,
                    start=start,
                    end=end,
                    output_offset=output_offset,
                )

            if not self.training and torch.cuda.is_available():
                del pred
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

    def compute_windows(self, total_frames: int) -> list[tuple[int, int]]:
        if total_frames <= self.window_len:
            return [(0, total_frames)]

        windows = []
        start = 0
        while start < total_frames:
            end = min(start + self.window_len, total_frames)
            windows.append((start, end))
            if end == total_frames:
                break
            start += self.window_stride
        return windows

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

    @staticmethod
    def _write_window_predictions(
        accumulated: dict,
        pred: dict,
        start: int,
        end: int,
        output_offset: int,
    ) -> None:
        write_len = end - start
        for key in ("track", "vis", "conf"):
            accumulated[key][:, start:end] = pred[key][:, output_offset : output_offset + write_len]

    @staticmethod
    def _write_window_iter_predictions(
        accumulated: dict,
        pred: dict,
        start: int,
        end: int,
        output_offset: int,
    ) -> None:
        write_len = end - start
        for key in ("track_iters", "vis_iters", "conf_iters"):
            for dst, src in zip(accumulated[key], pred[key]):
                dst[:, start:end] = src[:, output_offset : output_offset + write_len]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = None,
        window_len: int = 8,
        window_stride: int | None = None,
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
