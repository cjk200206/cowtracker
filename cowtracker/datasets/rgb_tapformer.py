# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RGB-only TAPFormer/EventKubric style training dataset."""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclasses.dataclass(eq=False)
class CowRGBData:
    """Batchable RGB point-tracking sample."""

    video: torch.Tensor  # [T, 3, H, W], float32 in [0, 255]
    trajectory: torch.Tensor  # [T, N, 2], pixel xy
    visibility: torch.Tensor  # [T, N], 1 for visible
    valid: torch.Tensor  # [T, N], 1 for real sampled points
    seq_name: Optional[str] = None


def cow_rgb_collate(batch):
    """Collate samples while preserving the gotit mask."""
    samples, gotit = zip(*batch)
    return (
        CowRGBData(
            video=torch.stack([s.video for s in samples], dim=0),
            trajectory=torch.stack([s.trajectory for s in samples], dim=0),
            visibility=torch.stack([s.visibility for s in samples], dim=0),
            valid=torch.stack([s.valid for s in samples], dim=0),
            seq_name=[s.seq_name for s in samples],
        ),
        torch.tensor(gotit, dtype=torch.bool),
    )


class RGBTapFormerDataset(Dataset):
    """Read TAPFormer/EventKubric test split as ordinary RGB tracking data.

    Expected layout:
        data_root/test/<seq>/raw/rgba_blur_*.png
        data_root/test/<seq>/raw/rgba_*.png
        data_root/test/<seq>/annotations.npy

    ``annotations.npy`` must contain TAPVid-style ``target_points`` and
    ``occluded`` arrays with shape [N, T, ...].
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "test",
        crop_size: tuple[int, int] = (384, 512),
        seq_len: int = 24,
        traj_per_sample: int = 256,
        frame_source: str = "blur",
        use_augs: bool = False,
        random_temporal_crop: bool = True,
        choose_long_point: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.root_dir = self._resolve_split_root(self.data_root, split)
        self.crop_size = tuple(int(v) for v in crop_size)
        self.seq_len = int(seq_len)
        self.traj_per_sample = int(traj_per_sample)
        self.frame_source = str(frame_source).lower().strip()
        self.use_augs = bool(use_augs)
        self.random_temporal_crop = bool(random_temporal_crop)
        self.choose_long_point = bool(choose_long_point)

        if self.frame_source not in {"blur", "clear"}:
            raise ValueError("frame_source must be one of: blur, clear")

        self.pad_bounds = (0, 25)
        self.resize_lim = (0.75, 1.25)
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

        self.seq_paths = [
            path
            for path in sorted(self.root_dir.iterdir())
            if path.is_dir() and (path / "raw").is_dir() and (path / "annotations.npy").is_file()
        ]
        if not self.seq_paths:
            raise FileNotFoundError(
                f"No valid sequences found under {self.root_dir}. "
                "Expected <seq>/raw and <seq>/annotations.npy."
            )

    @staticmethod
    def _resolve_split_root(data_root: Path, split: str) -> Path:
        if (data_root / split).is_dir():
            return data_root / split
        return data_root

    def __len__(self) -> int:
        return len(self.seq_paths)

    def __getitem__(self, index: int):
        try:
            return self._getitem(index), True
        except Exception as exc:
            print(f"warning: failed to load sample {self.seq_paths[index]}: {exc}")
            return self._empty_sample(self.seq_paths[index].name), False

    def _getitem(self, index: int) -> CowRGBData:
        seq_path = self.seq_paths[index]
        frame_paths = self._frame_paths(seq_path / "raw")
        annot = np.load(seq_path / "annotations.npy", allow_pickle=True).item()
        traj = np.asarray(annot["target_points"], dtype=np.float32)[..., :2]
        occluded = np.asarray(annot["occluded"]).astype(bool)

        if traj.ndim != 3 or traj.shape[-1] != 2:
            raise ValueError(f"target_points must have shape [N, T, 2], got {traj.shape}")
        if occluded.shape[:2] != traj.shape[:2]:
            raise ValueError(f"occluded shape {occluded.shape} is incompatible with {traj.shape}")

        total = min(len(frame_paths), traj.shape[1], occluded.shape[1])
        if total < self.seq_len:
            raise ValueError(f"sequence has {total} usable frames, less than seq_len={self.seq_len}")

        start = 0
        if self.random_temporal_crop and total > self.seq_len:
            start = random.randint(0, total - self.seq_len)
        end = start + self.seq_len

        frames = [self._read_rgb(path) for path in frame_paths[start:end]]
        traj = np.transpose(traj[:, start:end], (1, 0, 2))
        visibility = np.transpose(~occluded[:, start:end], (1, 0))

        if self.use_augs:
            frames, traj = self._spatial_augment(frames, traj, visibility)
        else:
            frames, traj = self._center_crop(frames, traj)

        visibility = self._mask_out_of_bounds(visibility, traj)
        traj_t = torch.from_numpy(traj).float()
        vis_t = torch.from_numpy(visibility).bool()

        selected, valid_points = self._sample_first_frame_points(traj_t, vis_t)
        traj_t = traj_t[:, selected].float()
        vis_t = vis_t[:, selected].float()
        valid_t = valid_points[None].expand(self.seq_len, -1).float()

        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
        return CowRGBData(
            video=video,
            trajectory=traj_t,
            visibility=vis_t,
            valid=valid_t,
            seq_name=seq_path.name,
        )

    def _empty_sample(self, seq_name: str) -> CowRGBData:
        h, w = self.crop_size
        return CowRGBData(
            video=torch.zeros(self.seq_len, 3, h, w),
            trajectory=torch.zeros(self.seq_len, self.traj_per_sample, 2),
            visibility=torch.zeros(self.seq_len, self.traj_per_sample),
            valid=torch.zeros(self.seq_len, self.traj_per_sample),
            seq_name=seq_name,
        )

    def _frame_paths(self, raw_dir: Path) -> list[Path]:
        files = sorted(raw_dir.iterdir())
        if self.frame_source == "blur":
            frame_paths = [
                p for p in files if p.name.startswith("rgba_blur_") and p.suffix.lower() == ".png"
            ]
        else:
            frame_paths = [
                p
                for p in files
                if p.name.startswith("rgba_")
                and not p.name.startswith("rgba_blur_")
                and p.suffix.lower() == ".png"
            ]
        if not frame_paths:
            raise FileNotFoundError(f"No {self.frame_source} rgba frames found under {raw_dir}")
        return frame_paths

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        image = Image.open(path)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        return np.asarray(image)

    def _center_crop(self, frames: list[np.ndarray], traj: np.ndarray):
        h_crop, w_crop = self.crop_size
        h, w = frames[0].shape[:2]
        pad_y = max(0, h_crop - h)
        pad_x = max(0, w_crop - w)
        if pad_y > 0 or pad_x > 0:
            top = pad_y // 2
            left = pad_x // 2
            frames = [
                np.pad(
                    f,
                    ((top, pad_y - top), (left, pad_x - left), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                for f in frames
            ]
            traj = traj.copy()
            traj[..., 0] += left
            traj[..., 1] += top
            h, w = frames[0].shape[:2]

        y0 = max(0, (h - h_crop) // 2)
        x0 = max(0, (w - w_crop) // 2)
        frames = [f[y0 : y0 + h_crop, x0 : x0 + w_crop] for f in frames]
        traj = traj.copy()
        traj[..., 0] -= x0
        traj[..., 1] -= y0
        return frames, traj

    def _spatial_augment(self, frames: list[np.ndarray], traj: np.ndarray, visibility: np.ndarray):
        h_crop, w_crop = self.crop_size
        frames = [f.astype(np.float32) for f in frames]
        traj = traj.copy()

        pad_x0 = random.randint(*self.pad_bounds)
        pad_x1 = random.randint(*self.pad_bounds)
        pad_y0 = random.randint(*self.pad_bounds)
        pad_y1 = random.randint(*self.pad_bounds)
        frames = [
            np.pad(f, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)), mode="constant")
            for f in frames
        ]
        traj[..., 0] += pad_x0
        traj[..., 1] += pad_y0

        h, w = frames[0].shape[:2]
        scale = np.random.uniform(*self.resize_lim)
        scale_x = scale
        scale_y = scale
        scaled = []
        for i, frame in enumerate(frames):
            if i > 0:
                scale_x += np.random.uniform(-self.resize_delta, self.resize_delta)
                scale_y += np.random.uniform(-self.resize_delta, self.resize_delta)
            scale_x = float(np.clip(scale_x, 0.2, 2.0))
            scale_y = float(np.clip(scale_y, 0.2, 2.0))
            h_new = max(h_crop + 10, int(h * scale_y))
            w_new = max(w_crop + 10, int(w * scale_x))
            sx = (w_new - 1) / max(1.0, float(w - 1))
            sy = (h_new - 1) / max(1.0, float(h - 1))
            scaled.append(cv2.resize(frame, (w_new, h_new), interpolation=cv2.INTER_LINEAR))
            traj[i, :, 0] *= sx
            traj[i, :, 1] *= sy
        frames = scaled

        first_vis = visibility[0] > 0
        if np.any(first_vis):
            center_x = float(np.mean(traj[0, first_vis, 0]))
            center_y = float(np.mean(traj[0, first_vis, 1]))
        else:
            center_x = w_crop / 2.0
            center_y = h_crop / 2.0

        x0 = int(center_x - w_crop // 2)
        y0 = int(center_y - h_crop // 2)
        offset_x = 0
        offset_y = 0
        cropped = []
        for i, frame in enumerate(frames):
            if i > 0:
                offset_x = int(0.8 * offset_x + 0.2 * random.randint(-self.max_crop_offset, self.max_crop_offset))
                offset_y = int(0.8 * offset_y + 0.2 * random.randint(-self.max_crop_offset, self.max_crop_offset))
            x0 += offset_x
            y0 += offset_y
            h_new, w_new = frame.shape[:2]
            x = min(max(0, x0), max(0, w_new - w_crop))
            y = min(max(0, y0), max(0, h_new - h_crop))
            cropped.append(frame[y : y + h_crop, x : x + w_crop])
            traj[i, :, 0] -= x
            traj[i, :, 1] -= y
        frames = cropped

        if random.random() < self.h_flip_prob:
            frames = [f[:, ::-1] for f in frames]
            traj[..., 0] = w_crop - traj[..., 0]
        if random.random() < self.v_flip_prob:
            frames = [f[::-1] for f in frames]
            traj[..., 1] = h_crop - traj[..., 1]

        frames = [np.ascontiguousarray(np.clip(f, 0, 255).astype(np.uint8)) for f in frames]
        return frames, traj

    def _mask_out_of_bounds(self, visibility: np.ndarray, traj: np.ndarray) -> np.ndarray:
        h, w = self.crop_size
        visible = visibility.copy()
        visible[traj[..., 0] < 0] = False
        visible[traj[..., 0] > w - 1] = False
        visible[traj[..., 1] < 0] = False
        visible[traj[..., 1] > h - 1] = False
        finite = np.isfinite(traj).all(axis=-1)
        visible &= finite
        return visible

    def _sample_first_frame_points(self, traj: torch.Tensor, visibility: torch.Tensor):
        visible_inds = torch.nonzero(visibility[0], as_tuple=False).flatten()
        valid_points = torch.zeros(self.traj_per_sample, dtype=torch.bool)
        if visible_inds.numel() == 0:
            return torch.zeros(self.traj_per_sample, dtype=torch.long), valid_points

        if self.choose_long_point and visible_inds.numel() > 1:
            displacement = torch.linalg.vector_norm(
                traj[-1, visible_inds] - traj[0, visible_inds], dim=-1
            )
            if displacement.sum() > 1e-6:
                probs = displacement / displacement.sum()
            else:
                probs = torch.full_like(displacement, 1.0 / visible_inds.numel())
            real_count = min(visible_inds.numel(), self.traj_per_sample)
            sampled_pos = torch.multinomial(probs, real_count, replacement=False)
            selected = visible_inds[sampled_pos]
            valid_points[:real_count] = True
            if real_count < self.traj_per_sample:
                pad = selected[torch.arange(self.traj_per_sample - real_count) % real_count]
                selected = torch.cat([selected, pad], dim=0)
            return selected, valid_points

        order = torch.randperm(visible_inds.numel())
        real_count = min(visible_inds.numel(), self.traj_per_sample)
        selected = visible_inds[order[:real_count]]
        valid_points[:real_count] = True
        if real_count < self.traj_per_sample:
            pad = selected[torch.arange(self.traj_per_sample - real_count) % real_count]
            selected = torch.cat([selected, pad], dim=0)
        return selected, valid_points
