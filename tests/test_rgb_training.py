import numpy as np
import torch
from PIL import Image

from cowtracker.datasets import RGBTapFormerDataset, cow_rgb_collate
from cowtracker.training.losses import CowTrackerDenseLoss
from scripts.train_cowtracker import freeze_vggt_patch_embedding, select_aggregator_state_dict


def _write_sample(root, num_frames=5, height=16, width=20, num_points=3):
    seq = root / "test" / "000000"
    raw = seq / "raw"
    raw.mkdir(parents=True)
    for t in range(num_frames):
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[..., :3] = t * 10
        image[..., 3] = 255
        Image.fromarray(image).save(raw / f"rgba_blur_{t:03d}.png")
        Image.fromarray(image).save(raw / f"rgba_{t:03d}.png")

    target_points = np.zeros((num_points, num_frames, 2), dtype=np.float32)
    for p in range(num_points):
        target_points[p, :, 0] = 3 + p + np.arange(num_frames)
        target_points[p, :, 1] = 4 + p
    occluded = np.zeros((num_points, num_frames), dtype=bool)
    np.save(seq / "annotations.npy", {"target_points": target_points, "occluded": occluded})


def test_rgb_tapformer_dataset_smoke(tmp_path):
    _write_sample(tmp_path)
    dataset = RGBTapFormerDataset(
        tmp_path,
        seq_len=4,
        traj_per_sample=5,
        crop_size=(12, 14),
        random_temporal_crop=False,
    )
    sample, gotit = dataset[0]
    assert gotit
    assert sample.video.shape == (4, 3, 12, 14)
    assert sample.trajectory.shape == (4, 5, 2)
    assert sample.visibility.shape == (4, 5)
    assert sample.valid.shape == (4, 5)
    assert sample.valid[:, :3].all()
    assert not sample.valid[:, 3:].any()

    batch, gotit = cow_rgb_collate([(sample, gotit)])
    assert gotit.tolist() == [True]
    assert batch.video.shape == (1, 4, 3, 12, 14)


def test_dense_loss_backward():
    b, t, h, w, n = 1, 3, 8, 10, 2
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dense_track = torch.stack([xx, yy], dim=-1).float()[None, None].repeat(b, t, 1, 1, 1)
    dense_track = dense_track.clone().requires_grad_(True)
    dense_vis = torch.full((b, t, h, w), 0.8, requires_grad=True)
    dense_conf = torch.full((b, t, h, w), 0.8, requires_grad=True)
    gt = torch.tensor([[[[2.0, 3.0], [5.0, 4.0]]] * t])
    visibility = torch.ones(b, t, n)
    valid = torch.ones(b, t, n)

    criterion = CowTrackerDenseLoss()
    out = criterion(
        {"track": dense_track, "vis": dense_vis, "conf": dense_conf},
        gt,
        visibility,
        valid,
    )
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert dense_track.grad is not None
    assert out["loss_coord"] is out["coord_loss"]
    assert out["loss_vis"] is out["visibility_loss"]
    assert out["loss_conf"] is out["confidence_loss"]


def test_dense_loss_supports_iteration_predictions():
    b, t, h, w, n = 1, 3, 8, 10, 2
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = torch.stack([xx, yy], dim=-1).float()[None, None].repeat(b, t, 1, 1, 1)
    iter_0 = base.clone().requires_grad_(True)
    iter_1 = (base + 0.5).clone().requires_grad_(True)
    dense_vis_0 = torch.full((b, t, h, w), 0.7, requires_grad=True)
    dense_vis_1 = torch.full((b, t, h, w), 0.8, requires_grad=True)
    dense_conf_0 = torch.full((b, t, h, w), 0.7, requires_grad=True)
    dense_conf_1 = torch.full((b, t, h, w), 0.8, requires_grad=True)
    gt = torch.tensor([[[[2.0, 3.0], [5.0, 4.0]]] * t])
    visibility = torch.ones(b, t, n)
    valid = torch.ones(b, t, n)

    criterion = CowTrackerDenseLoss(iter_gamma=0.5)
    out = criterion(
        {
            "track": iter_1,
            "vis": dense_vis_1,
            "conf": dense_conf_1,
            "track_iters": [iter_0, iter_1],
            "vis_iters": [dense_vis_0, dense_vis_1],
            "conf_iters": [dense_conf_0, dense_conf_1],
        },
        gt,
        visibility,
        valid,
    )
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert iter_0.grad is not None
    assert iter_1.grad is not None


def test_iter_gamma_weights_later_predictions_more():
    b, t, h, w, n = 1, 2, 2, 2, 1
    good = torch.zeros(b, t, h, w, 2)
    bad = torch.full((b, t, h, w, 2), 2.0)
    vis = torch.full((b, t, h, w), 0.9)
    conf = torch.full((b, t, h, w), 0.9)
    gt = torch.zeros(b, t, n, 2)
    visibility = torch.ones(b, t, n)
    valid = torch.ones(b, t, n)

    criterion = CowTrackerDenseLoss(
        coord_weight=1.0,
        visibility_weight=0.0,
        confidence_weight=0.0,
        use_huber=False,
        iter_gamma=0.5,
    )
    early_bad = criterion(
        {
            "track": good,
            "vis": vis,
            "conf": conf,
            "track_iters": [bad, good],
            "vis_iters": [vis, vis],
            "conf_iters": [conf, conf],
        },
        gt,
        visibility,
        valid,
    )
    late_bad = criterion(
        {
            "track": bad,
            "vis": vis,
            "conf": conf,
            "track_iters": [good, bad],
            "vis_iters": [vis, vis],
            "conf_iters": [conf, conf],
        },
        gt,
        visibility,
        valid,
    )
    assert late_bad["coord_loss"] > early_bad["coord_loss"]


def test_select_aggregator_state_dict_accepts_full_or_direct_keys():
    target_keys = {"block.weight", "block.bias"}
    full_state = {
        "aggregator.block.weight": torch.ones(2),
        "aggregator.block.bias": torch.zeros(2),
        "tracking_head.weight": torch.full((2,), 2.0),
    }
    selected = select_aggregator_state_dict(full_state, target_keys)
    assert set(selected) == target_keys
    assert torch.equal(selected["block.weight"], torch.ones(2))

    direct_state = {
        "block.weight": torch.ones(2),
        "block.bias": torch.zeros(2),
        "other.weight": torch.full((2,), 3.0),
    }
    selected = select_aggregator_state_dict(direct_state, target_keys)
    assert set(selected) == target_keys

    prefixed_state = {
        "module.aggregator.block.weight": torch.ones(2),
        "module.aggregator.block.bias": torch.zeros(2),
    }
    selected = select_aggregator_state_dict(prefixed_state, target_keys)
    assert set(selected) == target_keys


def test_freeze_vggt_freezes_only_patch_embedding_layers():
    class TinyVitLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Linear(3, 4)
            self.blocks = torch.nn.Linear(4, 4)

    class TinyAggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = TinyVitLike()
            self.frame_blocks = torch.nn.Linear(4, 4)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = TinyAggregator()

    model = TinyModel()
    frozen = freeze_vggt_patch_embedding(model)

    assert frozen == sum(p.numel() for p in model.aggregator.patch_embed.patch_embed.parameters())
    assert not any(p.requires_grad for p in model.aggregator.patch_embed.patch_embed.parameters())
    assert all(p.requires_grad for p in model.aggregator.patch_embed.blocks.parameters())
    assert all(p.requires_grad for p in model.aggregator.frame_blocks.parameters())


def test_tracking_head_return_all_iters_shapes():
    import pytest

    pytest.importorskip("timm")
    pytest.importorskip("xformers")
    from cowtracker.heads.tracking_head import CowTrackingHead

    head = CowTrackingHead(
        feature_dim=4,
        down_ratio=2,
        warp_iters=2,
        warp_model="vitt",
        warp_vit_num_blocks=1,
    )
    features = torch.randn(1, 2, 4, 8, 8)
    out = head(features, image_size=(16, 16), return_all_iters=True)
    assert out["track"].shape == (1, 2, 16, 16, 2)
    assert out["vis"].shape == (1, 2, 16, 16)
    assert out["conf"].shape == (1, 2, 16, 16)
    assert len(out["track_iters"]) == 2
    assert len(out["vis_iters"]) == 2
    assert len(out["conf_iters"]) == 2
