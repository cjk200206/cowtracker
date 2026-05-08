import numpy as np
import pytest
import torch

import scripts.visualize_cowtracker as vis


class _Batch:
    def __init__(self):
        self.video = torch.zeros(1, 3, 3, 8, 10)
        self.trajectory = torch.zeros(1, 3, 4, 2)
        self.visibility = torch.ones(1, 3, 4, dtype=torch.bool)
        self.valid = torch.ones(1, 3, 4, dtype=torch.bool)
        self.seq_name = ["dummy_seq"]


def _args(**kwargs):
    base = {
        "checkpoint": "dummy.ckpt",
        "config": "dummy.yaml",
        "output_dir": "output/vis_test",
        "sample_index": 0,
        "seq_len": None,
        "full_sequence": False,
        "online": False,
        "window_len": None,
        "window_stride": None,
        "num_points": 4,
        "point_source": "gt",
        "precision": None,
        "device": "cpu",
        "num_workers": 0,
        "vis_threshold": 0.5,
        "conf_threshold": None,
        "fps": 10,
        "linewidth": 2,
        "tracks_leave_trace": -1,
        "show_first_frame": 0,
        "draw_gt": False,
        "gt_only": False,
        "save_npz": False,
        "random_temporal_crop": False,
        "seed": 0,
    }
    base.update(kwargs)
    return type("Args", (), base)()


def _setup_common(monkeypatch, args):
    batch = _Batch()
    cfg = {"train": {"precision": "bf16"}}
    gt_tracks = torch.zeros(1, 3, 4, 2)
    gt_visibility = torch.ones(1, 3, 4, dtype=torch.bool)
    query_xy = torch.zeros(1, 4, 2)

    monkeypatch.setattr(vis, "parse_args", lambda: args)
    monkeypatch.setattr(vis, "set_seed", lambda seed: None)
    monkeypatch.setattr(vis, "load_config", lambda path: cfg)
    monkeypatch.setattr(vis, "build_visualization_dataset", lambda *a, **k: object())
    monkeypatch.setattr(vis, "load_one_batch", lambda *a, **k: (batch, 0))
    monkeypatch.setattr(vis, "move_batch_to_device", lambda b, d: b)
    monkeypatch.setattr(vis, "select_gt_points", lambda b, n: (query_xy, gt_tracks, gt_visibility))
    monkeypatch.setattr(vis, "select_grid_points", lambda b, n: (query_xy, None, None))
    monkeypatch.setattr(vis, "output_stem", lambda *a, **k: "stub")
    monkeypatch.setattr(vis.Path, "mkdir", lambda *a, **k: None)

    class DummyVisualizer:
        def __init__(self, *a, **k):
            self.calls = []

        def visualize(self, *vargs, **vkwargs):
            self.calls.append((vargs, vkwargs))
            return np.zeros((3, 8, 10, 3), dtype=np.uint8)

    dummy_visualizer = DummyVisualizer()
    monkeypatch.setattr(vis, "TrackVisualizer", lambda *a, **k: dummy_visualizer)
    return dummy_visualizer


def test_gt_only_skips_model_inference(monkeypatch):
    args = _args(gt_only=True, point_source="gt")
    dummy_visualizer = _setup_common(monkeypatch, args)

    called = {"load_checkpoint": 0, "build_model": 0}
    monkeypatch.setattr(vis, "load_checkpoint", lambda path: called.__setitem__("load_checkpoint", called["load_checkpoint"] + 1))
    monkeypatch.setattr(vis, "build_visualization_model", lambda *a, **k: called.__setitem__("build_model", called["build_model"] + 1))

    vis.main()

    assert called["load_checkpoint"] == 0
    assert called["build_model"] == 0
    assert len(dummy_visualizer.calls) == 1
    _, kwargs = dummy_visualizer.calls[0]
    assert kwargs["gt_tracks"] is None
    assert kwargs["gt_visibility"] is None


def test_gt_only_with_grid_raises(monkeypatch):
    args = _args(gt_only=True, point_source="grid")
    monkeypatch.setattr(vis, "parse_args", lambda: args)
    monkeypatch.setattr(vis, "set_seed", lambda seed: None)
    monkeypatch.setattr(vis, "load_config", lambda path: {"train": {"precision": "bf16"}})

    with pytest.raises(ValueError, match="--gt_only requires --point_source gt"):
        vis.main()


def test_gt_only_save_npz_contains_gt_only(monkeypatch):
    args = _args(gt_only=True, point_source="gt", save_npz=True)
    _setup_common(monkeypatch, args)

    captured = {}

    def fake_save(path, **arrays):
        captured.update(arrays)

    monkeypatch.setattr(vis.np, "savez_compressed", fake_save)
    monkeypatch.setattr(vis, "load_checkpoint", lambda path: (_ for _ in ()).throw(AssertionError("should not load ckpt")))

    vis.main()

    assert "query_xy" in captured
    assert "gt_track" in captured
    assert "gt_visibility" in captured
    assert "pred_track" not in captured
    assert "pred_visibility" not in captured
    assert "pred_confidence" not in captured


def test_non_gt_only_path_unchanged(monkeypatch):
    args = _args(gt_only=False, point_source="gt")
    _setup_common(monkeypatch, args)

    monkeypatch.setattr(vis, "load_checkpoint", lambda path: {"model": {}})

    class DummyModel:
        def __call__(self, video, return_all_iters=False):
            return {"track": torch.zeros(1, 3, 8, 10, 2), "vis": torch.ones(1, 3, 8, 10), "conf": torch.ones(1, 3, 8, 10)}

    monkeypatch.setattr(vis, "build_visualization_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        vis,
        "sample_sparse_predictions",
        lambda pred, q: (
            torch.zeros(1, 3, 4, 2),
            torch.ones(1, 3, 4),
            torch.ones(1, 3, 4),
        ),
    )

    vis.main()
