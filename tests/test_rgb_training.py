import numpy as np
import torch
from PIL import Image

from cowtracker.datasets import RGBTapFormerDataset, cow_rgb_collate
from cowtracker.heads.tracking_head import CowTrackingHead
from cowtracker.layers.cotracker3_encoder import CoTracker3BasicEncoder
from cowtracker.training.losses import CowTrackerDenseLoss, sample_dense_predictions
from cowtracker.utils.visualization import TrackVisualizer
from scripts.train_cowtracker import (
    freeze_vggt_patch_embedding,
    model_uses_vggt,
    select_aggregator_state_dict,
)


def _make_flow_update_head(limit_flow, update_ratio=0.15, magnitude_ratio=1.0):
    head = CowTrackingHead.__new__(CowTrackingHead)
    torch.nn.Module.__init__(head)
    head.limit_flow = bool(limit_flow)
    head.max_flow_update_ratio = float(update_ratio)
    head.max_flow_magnitude_ratio = float(magnitude_ratio)
    return head


def _make_init_head(down_ratio=2):
    head = CowTrackingHead.__new__(CowTrackingHead)
    torch.nn.Module.__init__(head)
    head.down_ratio = int(down_ratio)
    return head


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
    assert out["loss_invisible_coord"] is out["invisible_coord_loss"]
    assert out["loss_vis"] is out["visibility_loss"]
    assert out["loss_conf"] is out["confidence_loss"]


def test_invisible_coord_loss_uses_coord_error_when_huber_enabled():
    b, t, h, w, n = 1, 2, 2, 2, 1
    dense_track = torch.zeros(b, t, h, w, 2)
    dense_track[..., 0] = 10.0
    dense_vis = torch.full((b, t, h, w), 0.5)
    dense_conf = torch.full((b, t, h, w), 0.5)
    gt = torch.zeros(b, t, n, 2)
    visibility = torch.tensor([[[1.0], [0.0]]])
    valid = torch.ones(b, t, n)

    criterion = CowTrackerDenseLoss(use_huber=True, huber_delta=6.0)
    out = criterion(
        {"track": dense_track, "vis": dense_vis, "conf": dense_conf},
        gt,
        visibility,
        valid,
    )

    assert torch.allclose(out["loss_coord"], torch.tensor(21.0))
    assert torch.allclose(out["loss_invisible_coord"], torch.tensor(21.0))


def test_sample_dense_predictions_for_visualization():
    b, t, h, w = 1, 2, 6, 8
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dense = torch.stack([xx, yy], dim=-1).float()[None, None].repeat(b, t, 1, 1, 1)
    query_xy = torch.tensor([[[2.0, 3.0], [5.0, 1.0]]])

    sampled = sample_dense_predictions(dense, query_xy)

    assert sampled.shape == (b, t, 2, 2)
    assert torch.allclose(sampled[:, :, 0], torch.tensor([[[2.0, 3.0], [2.0, 3.0]]]))
    assert torch.allclose(sampled[:, :, 1], torch.tensor([[[5.0, 1.0], [5.0, 1.0]]]))


def test_tracking_head_none_initializers_match_zero_defaults():
    head = _make_init_head(down_ratio=2)

    flow = head._init_flow(
        init_track=None,
        B=1,
        S=3,
        H=4,
        W=5,
        H_img=8,
        W_img=10,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    info = head._init_info(
        init_vis=None,
        init_conf=None,
        B=1,
        S=3,
        H=4,
        W=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(flow, torch.zeros(1, 3, 2, 4, 5))
    assert torch.equal(info, torch.zeros(1, 3, 2, 4, 5))


def test_flow_limit_disabled_preserves_residual_update():
    head = _make_flow_update_head(limit_flow=False)
    flow = torch.ones(1, 2, 2, 4, 5)
    raw_delta = torch.full_like(flow, 100.0)

    updated = head._apply_flow_update(flow, raw_delta)

    assert torch.equal(updated, flow + raw_delta)


def test_flow_limit_bounds_update_and_total_magnitude():
    head = _make_flow_update_head(limit_flow=True, update_ratio=0.1, magnitude_ratio=0.25)
    flow = torch.zeros(1, 1, 2, 10, 20)
    raw_delta = torch.full_like(flow, 1_000_000.0)

    updated = head._apply_flow_update(flow, raw_delta)

    assert updated.abs().max() <= 2.0 + 1e-5

    near_limit_flow = torch.full_like(flow, 4.9)
    updated_near_limit = head._apply_flow_update(near_limit_flow, raw_delta)

    assert updated_near_limit.abs().max() <= 5.0 + 1e-5


def test_track_visualizer_renders_cotracker_style_frames(tmp_path):
    video = torch.zeros(1, 3, 3, 16, 20)
    video[:, :, 0] = 64
    tracks = torch.tensor(
        [
            [
                [[3.0, 4.0], [10.0, 12.0]],
                [[5.0, 4.0], [11.0, 12.0]],
                [[7.0, 4.0], [12.0, 12.0]],
            ]
        ]
    )
    visibility = torch.tensor([[[True, True], [True, False], [True, True]]])
    visualizer = TrackVisualizer(
        save_dir=str(tmp_path),
        show_first_frame=2,
        tracks_leave_trace=-1,
    )

    rendered = visualizer.visualize(video, tracks, visibility, filename=None)

    assert rendered.shape == (5, 16, 20, 3)
    assert rendered.dtype == np.uint8
    assert rendered.max() > 64


def test_cotracker3_basic_encoder_outputs_video_features():
    encoder = CoTracker3BasicEncoder(output_dim=32, stride=4)
    video = torch.rand(1, 3, 3, 32, 40)

    features = encoder(video)
    norms = torch.linalg.vector_norm(features, dim=2)

    assert features.shape == (1, 3, 32, 8, 10)
    assert torch.isfinite(features).all()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_cowtracker_cotracker3_backbone_skips_vggt_components():
    import pytest

    pytest.importorskip("timm")
    from cowtracker.models.cowtracker import CoWTracker

    model = CoWTracker(
        backbone_type="cotracker3",
        features=32,
        down_ratio=4,
        warp_iters=1,
    )

    assert model.backbone_type == "cotracker3"
    assert model.aggregator is None
    assert isinstance(model.feature_extractor, CoTracker3BasicEncoder)
    assert model.tracking_head.down_ratio == 4


def test_model_uses_vggt_reads_backbone_type():
    assert model_uses_vggt({"model": {}})
    assert model_uses_vggt({"model": {"backbone_type": "vggt"}})
    assert not model_uses_vggt({"model": {"backbone_type": "cotracker3"}})


def test_cowtracker_online_sliding_windows_use_memory_and_windowed_merge():
    from cowtracker.inference.windowed import WindowedInference
    from cowtracker.models.cowtracker_online import CoWTrackerOnline

    class TinyTrackingHead(torch.nn.Module):
        def __init__(self, owner):
            super().__init__()
            object.__setattr__(self, "owner", owner)

        def forward(
            self,
            features,
            image_size,
            first_frame_features=None,
            init_track=None,
            init_vis=None,
            init_conf=None,
            return_all_iters=False,
        ):
            del image_size, return_all_iters
            first_value = first_frame_features[:, :, 0, 0, 0].detach().cpu().tolist()[0]
            feature_values = features[:, :, 0, 0, 0].detach().cpu().tolist()[0]
            init_track_values = None if init_track is None else init_track[:, :, 0, 0, 0].detach().cpu().tolist()[0]
            init_vis_values = None if init_vis is None else init_vis[:, :, 0, 0].detach().cpu().tolist()[0]
            init_conf_values = None if init_conf is None else init_conf[:, :, 0, 0].detach().cpu().tolist()[0]
            self.owner.calls.append((first_value + feature_values, init_track_values, init_vis_values, init_conf_values))
            call_offset = 100 * (len(self.owner.calls) - 1)

            b, t, _, h, w = features.shape
            frame_values = features[:, :, 0, 0, 0].view(b, t, 1, 1) + call_offset
            track = features.new_zeros((b, t, h, w, 2))
            track[..., 0] = frame_values
            return {
                "track": track,
                "vis": features.new_ones((b, t, h, w)),
                "conf": features.new_ones((b, t, h, w)),
            }

    class TinyOnline(CoWTrackerOnline):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.window_len = 4
            self.window_stride = 2
            self.num_memory_frames = 10
            self.init_mode = "cotracker"
            self.merge_mode = "overwrite"
            self.windowed = WindowedInference(
                window_len=self.window_len,
                stride=self.window_stride,
                num_memory_frames=self.num_memory_frames,
            )
            self.last_windows = []
            self.calls = []
            self.backbone_type = "cotracker3"
            self.tracking_head = TinyTrackingHead(self)

        def _extract_window_features(self, frames):
            return frames

    model = TinyOnline().eval()
    video = (torch.arange(10) * 255).view(1, 10, 1, 1, 1).repeat(1, 1, 3, 2, 2).float()

    out = model(video)

    assert model.last_windows == [(0, 4), (2, 6), (4, 8), (6, 10)]
    assert [call[0] for call in model.calls] == [
        [0.0, 0.0, 1.0, 2.0, 3.0],
        [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ]
    assert model.calls[0][1] == [0.0, 1.0, 2.0, 3.0]
    assert model.calls[1][1] == [0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    assert model.calls[2][1] == [0.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0]
    assert model.calls[3][1] == [0.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0]
    assert model.calls[1][2] == [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    assert model.calls[1][3] == [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    assert out["track"].shape == (1, 10, 2, 2, 2)
    assert torch.equal(
        out["track"][0, :, 0, 0, 0],
        torch.tensor([0.0, 1.0, 2.0, 3.0, 104.0, 105.0, 206.0, 207.0, 308.0, 309.0]),
    )


def test_cowtracker_online_official_init_mode_skips_prev_window_state():
    from cowtracker.inference.windowed import WindowedInference
    from cowtracker.models.cowtracker_online import CoWTrackerOnline

    class TinyTrackingHead(torch.nn.Module):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def forward(
            self,
            features,
            image_size,
            first_frame_features=None,
            init_track=None,
            init_vis=None,
            init_conf=None,
            return_all_iters=False,
        ):
            del features, image_size, first_frame_features, return_all_iters
            self.owner.init_calls.append((init_track, init_vis, init_conf))
            return {
                "track": torch.zeros((1, 4, 2, 2, 2)),
                "vis": torch.ones((1, 4, 2, 2)),
                "conf": torch.ones((1, 4, 2, 2)),
            }

    class TinyOnline(CoWTrackerOnline):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.window_len = 4
            self.window_stride = 2
            self.num_memory_frames = 10
            self.use_history_frames = True
            self.init_mode = "official"
            self.merge_mode = "overwrite"
            self.windowed = WindowedInference(
                window_len=self.window_len,
                stride=self.window_stride,
                num_memory_frames=self.num_memory_frames,
            )
            self.last_windows = []
            self.backbone_type = "cotracker3"
            self.tracking_head = TinyTrackingHead(self)
            self.init_calls = []

        def _extract_window_features(self, frames):
            return frames

    model = TinyOnline().eval()
    video = (torch.arange(6) * 255).view(1, 6, 1, 1, 1).repeat(1, 1, 3, 2, 2).float()

    out = model(video)

    assert model.last_windows == [(0, 4), (2, 6)]
    assert model.init_calls == [(None, None, None), (None, None, None)]
    assert out["track"].shape == (1, 6, 2, 2, 2)


def test_cowtracker_online_short_video_uses_single_window():
    from cowtracker.models.cowtracker_online import CoWTrackerOnline

    class TinyOnline(CoWTrackerOnline):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.window_len = 4
            self.window_stride = 2
            self.init_mode = "cotracker"
            self.merge_mode = "overwrite"
            self.last_windows = []
            self.calls = 0

        def _forward_window_video(self, video, queries=None, return_all_iters=False):
            del queries, return_all_iters
            self.calls += 1
            b, t, _, h, w = video.shape
            return {
                "track": video.new_zeros((b, t, h, w, 2)),
                "vis": video.new_ones((b, t, h, w)),
                "conf": video.new_ones((b, t, h, w)),
            }

    model = TinyOnline().eval()
    video = torch.zeros(1, 3, 3, 2, 2)

    out = model(video)

    assert model.last_windows == [(0, 3)]
    assert model.calls == 1
    assert out["track"].shape[1] == 3


def test_cowtracker_online_windowed_path_supports_cotracker3_and_iters():
    from cowtracker.inference.windowed import WindowedInference
    from cowtracker.models.cowtracker_online import CoWTrackerOnline

    class FailingAggregator:
        def __call__(self, *args, **kwargs):
            raise AssertionError("cotracker3 online path must not call aggregator")

    class TinyFeatureExtractor(torch.nn.Module):
        def forward(self, frames):
            return frames

    class TinyTrackingHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(
            self,
            features,
            image_size,
            first_frame_features=None,
            init_track=None,
            init_vis=None,
            init_conf=None,
            return_all_iters=False,
        ):
            del image_size
            assert first_frame_features[:, :, 0, 0, 0].eq(0).all()
            assert init_track is not None
            assert init_vis is not None
            assert init_conf is not None
            self.calls += 1
            b, t, _, h, w = features.shape
            values = features[:, :, 0, 0, 0].view(b, t, 1, 1) + 100 * (self.calls - 1)
            track = features.new_zeros((b, t, h, w, 2))
            track[..., 0] = values
            pred = {
                "track": track,
                "vis": features.new_ones((b, t, h, w)),
                "conf": features.new_ones((b, t, h, w)),
            }
            if return_all_iters:
                pred["track_iters"] = [track + 10, track + 20]
                pred["vis_iters"] = [pred["vis"], pred["vis"]]
                pred["conf_iters"] = [pred["conf"], pred["conf"]]
            return pred

    class TinyOnline(CoWTrackerOnline):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.window_len = 4
            self.window_stride = 2
            self.num_memory_frames = 10
            self.init_mode = "cotracker"
            self.merge_mode = "overwrite"
            self.windowed = WindowedInference(
                window_len=self.window_len,
                stride=self.window_stride,
                num_memory_frames=self.num_memory_frames,
            )
            self.last_windows = []
            self.backbone_type = "cotracker3"
            self.aggregator = FailingAggregator()
            self.feature_extractor = TinyFeatureExtractor()
            self.tracking_head = TinyTrackingHead()

    model = TinyOnline().eval()
    video = (torch.arange(6) * 255).view(1, 6, 1, 1, 1).repeat(1, 1, 3, 2, 2).float()

    out = model(video, return_all_iters=True)

    assert model.last_windows == [(0, 4), (2, 6)]
    assert out["track_iters"][0].shape == (1, 6, 2, 2, 2)
    assert out["vis_iters"][0].shape == (1, 6, 2, 2)
    assert out["conf_iters"][0].shape == (1, 6, 2, 2)
    assert torch.equal(
        out["track"][0, :, 0, 0, 0],
        torch.tensor([0.0, 1.0, 2.0, 3.0, 104.0, 105.0]),
    )
    assert torch.equal(
        out["track_iters"][1][0, :, 0, 0, 0],
        torch.tensor([20.0, 21.0, 22.0, 23.0, 124.0, 125.0]),
    )


def test_cowtracker_online_can_disable_history_frames():
    from cowtracker.inference.windowed import WindowedInference
    from cowtracker.models.cowtracker_online import CoWTrackerOnline

    class TinyTrackingHead(torch.nn.Module):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def forward(
            self,
            features,
            image_size,
            first_frame_features=None,
            init_track=None,
            init_vis=None,
            init_conf=None,
            return_all_iters=False,
        ):
            del image_size, init_vis, init_conf, return_all_iters
            self.owner.feature_calls.append(features[:, :, 0, 0, 0].detach().cpu().tolist()[0])
            self.owner.first_frame_features.append(first_frame_features)
            self.owner.init_track_values.append(init_track[:, :, 0, 0, 0].detach().cpu().tolist()[0])
            b, t, _, h, w = features.shape
            track = features.new_zeros((b, t, h, w, 2))
            track[..., 0] = features[:, :, 0, 0, 0].view(b, t, 1, 1)
            return {
                "track": track,
                "vis": features.new_ones((b, t, h, w)),
                "conf": features.new_ones((b, t, h, w)),
            }

    class TinyOnline(CoWTrackerOnline):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.window_len = 4
            self.window_stride = 2
            self.num_memory_frames = 10
            self.use_history_frames = False
            self.init_mode = "cotracker"
            self.merge_mode = "overwrite"
            self.windowed = WindowedInference(
                window_len=self.window_len,
                stride=self.window_stride,
                num_memory_frames=self.num_memory_frames,
            )
            self.last_windows = []
            self.backbone_type = "cotracker3"
            self.tracking_head = TinyTrackingHead(self)
            self.feature_calls = []
            self.first_frame_features = []
            self.init_track_values = []

        def _extract_window_features(self, frames):
            return frames

    model = TinyOnline().eval()
    video = (torch.arange(6) * 255).view(1, 6, 1, 1, 1).repeat(1, 1, 3, 2, 2).float()

    out = model(video)

    assert model.last_windows == [(0, 4), (2, 6)]
    assert model.feature_calls == [[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]]
    assert model.first_frame_features == [None, None]
    assert model.init_track_values == [
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 3.0, 3.0],
    ]
    assert out["track"].shape == (1, 6, 2, 2, 2)


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


def test_iter_gamma_does_not_reweight_visibility_or_confidence_losses():
    b, t, h, w, n = 1, 2, 2, 2, 1
    track = torch.zeros(b, t, h, w, 2)
    vis_good = torch.full((b, t, h, w), 0.9)
    vis_bad = torch.full((b, t, h, w), 0.2)
    conf_good = torch.full((b, t, h, w), 0.9)
    conf_bad = torch.full((b, t, h, w), 0.2)
    gt = torch.zeros(b, t, n, 2)
    visibility = torch.ones(b, t, n)
    valid = torch.ones(b, t, n)

    criterion = CowTrackerDenseLoss(iter_gamma=0.1)
    early_bad = criterion(
        {
            "track": track,
            "vis": vis_good,
            "conf": conf_good,
            "track_iters": [track, track],
            "vis_iters": [vis_bad, vis_good],
            "conf_iters": [conf_bad, conf_good],
        },
        gt,
        visibility,
        valid,
    )
    late_bad = criterion(
        {
            "track": track,
            "vis": vis_bad,
            "conf": conf_bad,
            "track_iters": [track, track],
            "vis_iters": [vis_good, vis_bad],
            "conf_iters": [conf_good, conf_bad],
        },
        gt,
        visibility,
        valid,
    )

    assert torch.allclose(early_bad["visibility_loss"], late_bad["visibility_loss"])
    assert torch.allclose(early_bad["confidence_loss"], late_bad["confidence_loss"])


def test_invisible_points_use_auxiliary_coord_loss_not_visible_coord_loss():
    b, t, h, w, n = 1, 2, 6, 6, 1
    track = torch.zeros(b, t, h, w, 2)
    vis = torch.full((b, t, h, w), 0.8)
    conf = torch.full((b, t, h, w), 0.8)
    gt = torch.tensor([[[[0.0, 0.0]], [[4.0, 4.0]]]])
    visibility = torch.tensor([[[1.0], [0.0]]])
    valid = torch.ones(b, t, n)

    criterion = CowTrackerDenseLoss(
        coord_weight=1.0,
        invisible_coord_weight=1.0,
        visibility_weight=0.0,
        confidence_weight=0.0,
        use_huber=False,
    )
    out = criterion({"track": track, "vis": vis, "conf": conf}, gt, visibility, valid)

    assert torch.allclose(out["coord_loss"], torch.tensor(0.0))
    assert torch.allclose(out["invisible_coord_loss"], torch.tensor(4.0))
    assert torch.allclose(out["loss"], torch.tensor(4.0))


def test_invisible_coord_weight_zero_keeps_total_loss_compatible():
    b, t, h, w, n = 1, 2, 6, 6, 1
    track = torch.zeros(b, t, h, w, 2)
    vis = torch.full((b, t, h, w), 0.8)
    conf = torch.full((b, t, h, w), 0.8)
    gt = torch.tensor([[[[0.0, 0.0]], [[4.0, 4.0]]]])
    visibility = torch.tensor([[[1.0], [0.0]]])
    valid = torch.ones(b, t, n)

    criterion = CowTrackerDenseLoss(invisible_coord_weight=0.0, use_huber=False)
    out = criterion({"track": track, "vis": vis, "conf": conf}, gt, visibility, valid)
    expected = (
        criterion.coord_weight * out["coord_loss"]
        + criterion.visibility_weight * out["visibility_loss"]
        + criterion.confidence_weight * out["confidence_loss"]
    )

    assert out["invisible_coord_loss"] > 0
    assert torch.allclose(out["loss"], expected)


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
