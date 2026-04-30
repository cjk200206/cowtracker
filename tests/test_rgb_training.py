import numpy as np
import torch
from PIL import Image

from cowtracker.datasets import RGBTapFormerDataset, cow_rgb_collate
from cowtracker.training.losses import CowTrackerDenseLoss


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
