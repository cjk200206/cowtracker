# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Losses for first-frame anchored dense CoWTracker training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float) -> torch.Tensor:
    diff = x - y
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, dtype=x.dtype, device=x.device))
    linear = abs_diff - quadratic
    return 0.5 * quadratic**2 + delta * linear


def sample_dense_predictions(predictions: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
    """Sample dense [B, T, H, W, C] predictions at first-frame query xy points."""
    b, t, h, w, c = predictions.shape
    n = query_xy.shape[1]

    x = query_xy[..., 0]
    y = query_xy[..., 1]
    x_norm = 2.0 * x / max(w - 1, 1) - 1.0
    y_norm = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    grid = grid[:, None].expand(b, t, n, 2).reshape(b * t, n, 1, 2)

    dense = predictions.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    sampled = F.grid_sample(dense, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(-1).permute(0, 2, 1).reshape(b, t, n, c)


class CowTrackerDenseLoss(nn.Module):
    """Sparse supervised loss over dense CoWTracker outputs.

    The supervision anchor is always the GT coordinate at frame 0.
    """

    def __init__(
        self,
        coord_weight: float = 0.05,
        visibility_weight: float = 1.0,
        confidence_weight: float = 1.0,
        confidence_threshold: float = 12.0,
        use_huber: bool = True,
        huber_delta: float = 6.0,
    ) -> None:
        super().__init__()
        self.coord_weight = float(coord_weight)
        self.visibility_weight = float(visibility_weight)
        self.confidence_weight = float(confidence_weight)
        self.confidence_threshold = float(confidence_threshold)
        self.use_huber = bool(use_huber)
        self.huber_delta = float(huber_delta)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        gt_trajectory: torch.Tensor,
        gt_visibility: torch.Tensor,
        gt_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        track = predictions["track"]
        vis = predictions["vis"][..., None]
        conf = predictions["conf"][..., None]

        query_xy = gt_trajectory[:, 0].detach()
        pred_track = sample_dense_predictions(track, query_xy)
        pred_vis = sample_dense_predictions(vis, query_xy).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
        pred_conf = sample_dense_predictions(conf, query_xy).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)

        gt_visibility = gt_visibility.float()
        gt_valid = gt_valid.float()
        finite = torch.isfinite(gt_trajectory).all(dim=-1).float()
        valid = gt_valid * finite
        visible_valid = valid * gt_visibility

        if self.use_huber:
            coord_elem = _huber_loss(pred_track, gt_trajectory, self.huber_delta).mean(dim=-1)
        else:
            coord_elem = (pred_track - gt_trajectory).abs().mean(dim=-1)
        coord_loss = _masked_mean(coord_elem, visible_valid)

        vis_elem = F.binary_cross_entropy(pred_vis, gt_visibility, reduction="none")
        visibility_loss = _masked_mean(vis_elem, valid)

        err_sq = torch.sum((pred_track.detach() - gt_trajectory) ** 2, dim=-1)
        conf_target = (err_sq <= self.confidence_threshold**2).float()
        conf_elem = F.binary_cross_entropy(pred_conf, conf_target, reduction="none")
        confidence_loss = _masked_mean(conf_elem, visible_valid)

        loss = (
            self.coord_weight * coord_loss
            + self.visibility_weight * visibility_loss
            + self.confidence_weight * confidence_loss
        )
        with torch.no_grad():
            epe = torch.linalg.vector_norm(pred_track - gt_trajectory, dim=-1)
            final_epe = _masked_mean(epe, visible_valid)
            vis_acc = _masked_mean(((pred_vis > 0.5).float() == gt_visibility).float(), valid)

        return {
            "loss": loss,
            "coord_loss": coord_loss,
            "visibility_loss": visibility_loss,
            "confidence_loss": confidence_loss,
            "final_epe": final_epe,
            "vis_acc": vis_acc,
            "pred_track": pred_track,
            "pred_visibility": pred_vis,
            "pred_confidence": pred_conf,
        }
