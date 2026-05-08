# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CoWTracker: Simple version for short videos."""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from cowtracker.heads.tracking_head import CowTrackingHead
from cowtracker.layers.cotracker3_encoder import CoTracker3BasicEncoder


class CoWTracker(nn.Module, PyTorchModelHubMixin):
    """
    CoWTracker simple version: processes entire video at once.

    Suitable for: short videos / sufficient GPU memory.
    For long videos, use CoWTrackerWindowed instead.
    """

    # Backbone configuration
    IMG_SIZE = 518
    PATCH_SIZE = 14
    EMBED_DIM = 1024
    PATCH_EMBED = "dinov2_vitl14_reg"
    DEPTH = 24

    # Default HuggingFace repo for model loading
    DEFAULT_REPO_ID = "facebook/cowtracker"
    DEFAULT_FILENAME = "cowtracker_model.pth"

    def __init__(
        self,
        backbone_type: str = "vggt",
        features: int = 128,
        side_resnet_channels: int = 128,
        down_ratio: int = 2,
        cotracker_backbone_stride: int = None,
        warp_iters: int = 5,
        warp_vit_num_blocks: int = None,
        limit_flow: bool = False,
        max_flow_update_ratio: float = 0.15,
        max_flow_magnitude_ratio: float = 1.0,
    ):
        """
        Args:
            backbone_type: "vggt" for the original VGGT+DPT path, or
                "cotracker3" for a lightweight CoTracker3 BasicEncoder path.
            features: Number of DPT output features.
            side_resnet_channels: Number of ResNet side feature channels.
            down_ratio: Feature downsampling ratio.
            cotracker_backbone_stride: Optional stride for the cotracker3
                backbone. Defaults to down_ratio when unset.
            warp_iters: Number of Warping-based iterative refinement iterations.
            warp_vit_num_blocks: Number of transformer blocks (None = default).
            limit_flow: If True, bound residual flow updates in the tracking head.
            max_flow_update_ratio: Per-iteration flow update limit as a ratio of
                the tracking feature map's longer side.
            max_flow_magnitude_ratio: Accumulated flow limit as a ratio of the
                tracking feature map's longer side.
        """
        super().__init__()

        print("Initializing CoWTracker...")
        self.backbone_type = str(backbone_type).lower().strip()
        if self.backbone_type not in {"vggt", "cotracker3"}:
            raise ValueError("backbone_type must be one of: vggt, cotracker3")

        self.aggregator = None
        self.feature_extractor = None

        if self.backbone_type == "vggt":
            # Lazy imports keep the cotracker3 path independent from the VGGT submodule.
            import cowtracker.thirdparty  # noqa: F401 - sets up vggt path
            from vggt.models.aggregator import Aggregator
            from cowtracker.heads.feature_extractor import FeatureExtractor

            self.aggregator = Aggregator(
                img_size=self.IMG_SIZE,
                patch_size=self.PATCH_SIZE,
                embed_dim=self.EMBED_DIM,
                patch_embed=self.PATCH_EMBED,
                depth=self.DEPTH,
            )

            self.feature_extractor = FeatureExtractor(
                features=features,
                down_ratio=down_ratio,
                side_resnet_channels=side_resnet_channels,
            )
            tracking_feature_dim = self.feature_extractor.out_dim
            tracking_down_ratio = down_ratio
        else:
            tracking_down_ratio = (
                int(cotracker_backbone_stride)
                if cotracker_backbone_stride is not None
                else int(down_ratio)
            )
            self.feature_extractor = CoTracker3BasicEncoder(
                input_dim=3,
                output_dim=features,
                stride=tracking_down_ratio,
            )
            tracking_feature_dim = features

        # Tracking head: warping-based iterative refinement
        self.tracking_head = CowTrackingHead(
            feature_dim=tracking_feature_dim,
            down_ratio=tracking_down_ratio,
            warp_iters=warp_iters,
            warp_vit_num_blocks=warp_vit_num_blocks,
            limit_flow=limit_flow,
            max_flow_update_ratio=max_flow_update_ratio,
            max_flow_magnitude_ratio=max_flow_magnitude_ratio,
        )

        print(f"  - Backbone: {self.backbone_type}")
        if self.backbone_type == "vggt":
            print(f"  - Features: {features}, Side channels: {side_resnet_channels}")
        else:
            print(f"  - Features: {features}, Encoder stride: {tracking_down_ratio}")
        print(f"  - Warping-based iterative refinement iterations: {warp_iters}")

    def forward(
        self,
        video: torch.Tensor,
        queries: torch.Tensor = None,
        return_all_iters: bool = False,
    ) -> dict:
        """
        Forward pass for dense tracking.

        Args:
            video: Input video [B, S, 3, H, W] or [S, 3, H, W] in range [0, 255].
            queries: Optional query points (unused, for API compatibility).
            return_all_iters: If True, include per-warp-iteration dense predictions
                for training losses. Defaults to False to preserve inference API.

        Returns:
            dict with:
                - track: Dense tracks [B, S, H, W, 2].
                - vis: Visibility scores [B, S, H, W].
                - conf: Confidence scores [B, S, H, W].
        """
        # Normalize input
        images = video / 255.0
        if images.ndim == 4:
            images = images.unsqueeze(0)

        B, S, C, H, W = images.shape

        if self.backbone_type == "vggt":
            tokens, patch_idx = self.aggregator(images)
            features = self.feature_extractor(tokens, images, patch_idx)
        else:
            features = self.feature_extractor(images)

        # Run tracking
        predictions = self.tracking_head(
            features,
            image_size=(H, W),
            return_all_iters=return_all_iters,
        )

        if not self.training:
            predictions["images"] = images

        return predictions

    @staticmethod
    def _remap_legacy_state_dict(state_dict: dict) -> dict:
        """
        Remap legacy checkpoint keys to new model structure.

        Old structure:
            tracking_head.aggregator.* -> aggregator.*
            tracking_head.feature_extractor.* -> feature_extractor.dpt_head.*
            tracking_head.fnet.* -> feature_extractor.fnet.*
            tracking_head.* (rest) -> tracking_head.*

        Args:
            state_dict: Original state dict.

        Returns:
            Remapped state dict.
        """
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Remap tracking_head.aggregator -> aggregator
            if key.startswith("tracking_head.aggregator."):
                new_key = key.replace("tracking_head.aggregator.", "aggregator.")
            # Remap tracking_head.feature_extractor -> feature_extractor.dpt_head
            elif key.startswith("tracking_head.feature_extractor."):
                new_key = key.replace(
                    "tracking_head.feature_extractor.", "feature_extractor.dpt_head."
                )
            # Remap tracking_head.fnet -> feature_extractor.fnet
            elif key.startswith("tracking_head.fnet."):
                new_key = key.replace("tracking_head.fnet.", "feature_extractor.fnet.")

            new_state_dict[new_key] = value

        return new_state_dict

    @classmethod
    def _load_checkpoint(cls, checkpoint_path: str = None) -> dict:
        """
        Load checkpoint from local path or HuggingFace Hub.

        Args:
            checkpoint_path: Local file path to checkpoint.
                             If None, downloads from default HuggingFace repo.

        Returns:
            Loaded checkpoint dict.
        """
        import os

        if checkpoint_path is None:
            # Download from HuggingFace Hub (uses HF_TOKEN env var automatically)
            print(f"Downloading checkpoint from HuggingFace: {cls.DEFAULT_REPO_ID}/{cls.DEFAULT_FILENAME}")
            checkpoint_path = hf_hub_download(
                repo_id=cls.DEFAULT_REPO_ID,
                filename=cls.DEFAULT_FILENAME,
            )
            print(f"Downloaded to: {checkpoint_path}")
        else:
            checkpoint_path = os.path.expanduser(checkpoint_path)
            print(f"Loading checkpoint from local path: {checkpoint_path}")

        with open(checkpoint_path, "rb") as fp:
            ckpt = torch.load(fp, map_location="cpu")

        return ckpt

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = None,
        device: str = "cuda",
        dtype=torch.bfloat16,
    ):
        """
        Load model from checkpoint (local path or HuggingFace Hub).

        Args:
            checkpoint_path: Path to local checkpoint file.
                             If None, downloads from default HuggingFace repo.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Loaded model in eval mode.
        """
        ckpt = cls._load_checkpoint(checkpoint_path)
        model_kwargs = {}
        if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
            model_kwargs = dict(ckpt["config"].get("model", {}))
            for key in (
                "freeze_vggt",
                "freeze_aggregator",
                "freeze_feature_extractor",
                "freeze_tracking_head",
            ):
                model_kwargs.pop(key, None)

        model = cls(**model_kwargs)
        state_dict = ckpt.get("model", ckpt)

        # Remap legacy checkpoint keys if needed
        legacy_prefixes = ["tracking_head.feature_extractor.", "tracking_head.aggregator.", "tracking_head.fnet."]
        if any(k.startswith(p) for k in state_dict.keys() for p in legacy_prefixes):
            print("Detected legacy checkpoint format, remapping keys...")
            state_dict = cls._remap_legacy_state_dict(state_dict)

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load message: {msg}")

        model = model.to(device).to(dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print("Model loaded successfully!")
        return model
