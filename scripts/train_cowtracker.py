#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train CoWTracker on RGB TAPFormer/EventKubric-style data."""

from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Force using only the first GPU for visualization

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cowtracker.datasets import RGBTapFormerDataset, cow_rgb_collate
from cowtracker.training import CowTrackerDenseLoss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config/train_rgb_tapformer_test.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--dry_run_loader", action="store_true")
    parser.add_argument("--dry_run_model", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric_is_better(current: float, best: float, mode: str) -> bool:
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError("best_mode must be one of: min, max")


def checkpoint_state(model, optimizer, cfg, epoch, global_step, best_metric, best_epoch):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
    }


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    raise ValueError("Checkpoint must be a state_dict-like dict.")


def normalize_state_dict_keys(state_dict):
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def select_aggregator_state_dict(state_dict, target_keys):
    """Extract VGGT aggregator weights from full-model or aggregator-only checkpoints."""
    state_dict = normalize_state_dict_keys(state_dict)
    prefixes = ("aggregator.", "model.aggregator.", "module.aggregator.")
    for prefix in prefixes:
        selected = {
            k[len(prefix) :]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if selected:
            return selected

    target_keys = set(target_keys)
    selected = {k: v for k, v in state_dict.items() if k in target_keys}
    if selected:
        return selected

    raise ValueError(
        "Could not find VGGT aggregator weights in checkpoint. Expected keys like "
        "'aggregator.*' from a full VGGT/CoWTracker checkpoint, or direct aggregator "
        "keys matching model.aggregator.state_dict()."
    )


def load_vggt_checkpoint(model, checkpoint_path, strict=False):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt)
    target_keys = model.aggregator.state_dict().keys()
    aggregator_state = select_aggregator_state_dict(state_dict, target_keys)
    incompatible = model.aggregator.load_state_dict(aggregator_state, strict=bool(strict))
    print(
        "Loaded VGGT aggregator weights "
        f"from {checkpoint_path} "
        f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})",
        flush=True,
    )
    return incompatible


def vggt_patch_embedding_modules(aggregator):
    """
    Return the image-to-patch embedding modules in VGGT.

    For the DINOv2 patch embed used by CoWTracker, aggregator.patch_embed is a
    ViT and aggregator.patch_embed.patch_embed is the actual image projection.
    For a plain conv PatchEmbed backbone, aggregator.patch_embed itself is the
    image projection.
    """
    patch_embed = getattr(aggregator, "patch_embed", None)
    if patch_embed is None:
        raise AttributeError("VGGT aggregator does not expose a patch_embed module.")
    inner_patch_embed = getattr(patch_embed, "patch_embed", None)
    return [inner_patch_embed if inner_patch_embed is not None else patch_embed]


def freeze_vggt_patch_embedding(model):
    frozen = 0
    modules = vggt_patch_embedding_modules(model.aggregator)
    for module in modules:
        for p in module.parameters():
            if p.requires_grad:
                frozen += p.numel()
            p.requires_grad = False
    return frozen


def set_vggt_patch_embedding_eval(model):
    for module in vggt_patch_embedding_modules(model.aggregator):
        module.eval()


def build_dataset(cfg):
    data_cfg = cfg["data"]
    return RGBTapFormerDataset(
        data_root=data_cfg["data_root"],
        split=data_cfg.get("split", "test"),
        crop_size=tuple(data_cfg.get("crop_size", [384, 512])),
        seq_len=int(data_cfg.get("seq_len", 24)),
        traj_per_sample=int(data_cfg.get("traj_per_sample", 256)),
        frame_source=data_cfg.get("frame_source", "blur"),
        use_augs=bool(data_cfg.get("use_augs", False)),
        random_temporal_crop=bool(data_cfg.get("random_temporal_crop", True)),
        choose_long_point=bool(data_cfg.get("choose_long_point", False)),
    )


def ensure_vggt_available():
    vggt_root = ROOT / "cowtracker" / "thirdparty" / "vggt"
    required_files = [
        vggt_root / "vggt" / "models" / "aggregator.py",
        vggt_root / "vggt" / "heads" / "dpt_head.py",
    ]
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_rel = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(
            "VGGT submodule is required for CoWTracker model forward, but files are missing:\n"
            f"{missing_rel}\n"
            "Run: git submodule update --init --recursive\n"
            "--dry_run_loader only checks the dataset and does not validate VGGT."
        )


def build_model(cfg, device):
    ensure_vggt_available()
    from cowtracker.models.cowtracker import CoWTracker

    model_cfg = dict(cfg.get("model", {}))
    freeze_aggregator = bool(model_cfg.pop("freeze_aggregator", False))
    freeze_vggt = bool(model_cfg.pop("freeze_vggt", True))
    freeze_feature_extractor = bool(model_cfg.pop("freeze_feature_extractor", False))
    freeze_tracking_head = bool(model_cfg.pop("freeze_tracking_head", False))
    model = CoWTracker(**model_cfg).to(device)

    train_cfg = cfg.get("train", {})
    init_checkpoint = train_cfg.get("init_checkpoint")
    init_from_hf = bool(train_cfg.get("init_from_hf", True))
    if init_checkpoint or init_from_hf:
        ckpt = CoWTracker._load_checkpoint(init_checkpoint)
        state_dict = extract_state_dict(ckpt)
        state_dict = normalize_state_dict_keys(state_dict)
        legacy_prefixes = [
            "tracking_head.feature_extractor.",
            "tracking_head.aggregator.",
            "tracking_head.fnet.",
        ]
        if any(k.startswith(p) for k in state_dict for p in legacy_prefixes):
            state_dict = CoWTracker._remap_legacy_state_dict(state_dict)
        incompatible = model.load_state_dict(state_dict, strict=bool(train_cfg.get("init_strict", False)))
        print(
            "Loaded init weights "
            f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})",
            flush=True,
        )

    vggt_checkpoint = train_cfg.get("vggt_checkpoint")
    if vggt_checkpoint:
        if init_checkpoint or init_from_hf:
            print(
                "vggt_checkpoint is set; its aggregator weights will override aggregator "
                "weights loaded from init_checkpoint/init_from_hf.",
                flush=True,
            )
        load_vggt_checkpoint(
            model,
            vggt_checkpoint,
            strict=bool(train_cfg.get("vggt_strict", False)),
        )
    elif (freeze_aggregator or freeze_vggt) and not (init_checkpoint or init_from_hf):
        print(
            "Warning: VGGT freezing is enabled but no full init weights or vggt_checkpoint "
            "were provided, so randomly initialized VGGT weights will be frozen.",
            flush=True,
        )

    if freeze_aggregator:
        for p in model.aggregator.parameters():
            p.requires_grad = False
        print("VGGT aggregator frozen.", flush=True)
    elif freeze_vggt:
        frozen = freeze_vggt_patch_embedding(model)
        print(
            "VGGT patch-embedding layers frozen; the rest of the VGGT backbone "
            f"remains trainable. Frozen parameters: {frozen:,}",
            flush=True,
        )
    else:
        print("VGGT backbone fully trainable.", flush=True)
    if freeze_feature_extractor:
        for p in model.feature_extractor.parameters():
            p.requires_grad = False
    if freeze_tracking_head:
        for p in model.tracking_head.parameters():
            p.requires_grad = False

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_trainable == 0:
        raise ValueError("No trainable parameters remain after applying freeze settings.")
    print(f"Trainable parameters: {num_trainable:,}", flush=True)
    return model


def set_frozen_modules_eval(model, cfg):
    model_cfg = cfg.get("model", {})
    if bool(model_cfg.get("freeze_aggregator", False)):
        model.aggregator.eval()
    elif bool(model_cfg.get("freeze_vggt", True)):
        set_vggt_patch_embedding_eval(model)
    if bool(model_cfg.get("freeze_feature_extractor", False)):
        model.feature_extractor.eval()
    if bool(model_cfg.get("freeze_tracking_head", False)):
        model.tracking_head.eval()


def move_batch_to_device(batch, device):
    batch.video = batch.video.to(device, non_blocking=True)
    batch.trajectory = batch.trajectory.to(device, non_blocking=True)
    batch.visibility = batch.visibility.to(device, non_blocking=True)
    batch.valid = batch.valid.to(device, non_blocking=True)
    return batch


def autocast_dtype(precision: str):
    precision = str(precision).lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision in {"fp32", "32", "float32", "none"}:
        return None
    raise ValueError("precision must be one of: bf16, fp16, fp32")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    if args.resume is not None:
        train_cfg["resume"] = args.resume
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers

    set_seed(int(train_cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(train_cfg.get("save_dir", "output/train_rgb_tapformer_test"))
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    dataset = build_dataset(cfg)
    num_workers = int(train_cfg.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=bool(train_cfg.get("shuffle", True)),
        num_workers=num_workers,
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
        drop_last=bool(train_cfg.get("drop_last", True)),
        collate_fn=cow_rgb_collate,
    )

    if args.dry_run_loader:
        batch, gotit = next(iter(loader))
        print(f"gotit={gotit.tolist()}")
        print(f"video={tuple(batch.video.shape)}")
        print(f"trajectory={tuple(batch.trajectory.shape)}")
        print(f"visibility={tuple(batch.visibility.shape)}")
        print(f"valid={tuple(batch.valid.shape)}")
        print(f"seq_name={batch.seq_name}")
        return

    model = build_model(cfg, device)
    criterion = CowTrackerDenseLoss(**cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    start_epoch = 0
    global_step = 0
    best_epoch = -1
    best_mode = str(train_cfg.get("best_mode", "min")).lower()
    best_metric_name = str(train_cfg.get("best_metric", "loss"))
    best_metric = math.inf if best_mode == "min" else -math.inf
    resume_path = train_cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(normalize_state_dict_keys(extract_state_dict(ckpt)), strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_metric = float(ckpt.get("best_metric", best_metric))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        print(f"Resumed from {resume_path} at epoch={start_epoch}, step={global_step}", flush=True)

    precision = str(train_cfg.get("precision", "bf16")).lower()
    amp_dtype = autocast_dtype(precision)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.type == "cuda"))
    max_steps = train_cfg.get("max_steps")
    max_steps = None if max_steps is None else int(max_steps)
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 10))
    save_every_steps = int(train_cfg.get("save_every_steps", 1000))
    return_all_iters = bool(train_cfg.get("return_all_iters", False))

    if args.dry_run_model:
        batch, gotit = next(iter(loader))
        if not bool(gotit.all()):
            raise RuntimeError(f"dry_run_model got invalid sample flags: {gotit.tolist()}")
        batch = move_batch_to_device(batch, device)
        model.train()
        set_frozen_modules_eval(model, cfg)
        use_amp = amp_dtype is not None and device.type == "cuda"
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
                predictions = model(batch.video, return_all_iters=return_all_iters)
                loss_dict = criterion(predictions, batch.trajectory, batch.visibility, batch.valid)
        print(f"track={tuple(predictions['track'].shape)}")
        if "track_iters" in predictions:
            print(f"track_iters={len(predictions['track_iters'])}")
        print(f"loss={float(loss_dict['loss']):.4f}")
        return

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, int(train_cfg.get("epochs", 40))):
        last_epoch = epoch
        model.train()
        set_frozen_modules_eval(model, cfg)
        running = {
            "loss": 0.0,
            "coord_loss": 0.0,
            "visibility_loss": 0.0,
            "confidence_loss": 0.0,
            "loss_coord": 0.0,
            "loss_vis": 0.0,
            "loss_conf": 0.0,
            "final_epe": 0.0,
            "vis_acc": 0.0,
        }
        steps_this_epoch = 0

        for batch, gotit in loader:
            if not bool(gotit.all()):
                continue
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            use_amp = amp_dtype is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
                predictions = model(batch.video, return_all_iters=return_all_iters)
                loss_dict = criterion(predictions, batch.trajectory, batch.visibility, batch.valid)
                loss = loss_dict["loss"]

            if precision == "fp16" and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            global_step += 1
            steps_this_epoch += 1
            for key in running:
                running[key] += float(loss_dict[key].detach().item())

            if global_step % log_every == 0:
                denom = max(1, steps_this_epoch)
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={running['loss']/denom:.4f} "
                    f"loss_coord={running['loss_coord']/denom:.4f} "
                    f"loss_vis={running['loss_vis']/denom:.4f} "
                    f"loss_conf={running['loss_conf']/denom:.4f} "
                    f"epe={running['final_epe']/denom:.4f}",
                    flush=True,
                )

            if save_every_steps > 0 and global_step % save_every_steps == 0:
                torch.save(
                    checkpoint_state(model, optimizer, cfg, epoch, global_step, best_metric, best_epoch),
                    save_dir / f"step_{global_step:07d}.pth",
                )

            if max_steps is not None and global_step >= max_steps:
                break

        if steps_this_epoch > 0:
            metrics = {k: v / steps_this_epoch for k, v in running.items()}
            current = float(metrics[best_metric_name])
            if bool(train_cfg.get("save_best", True)) and metric_is_better(current, best_metric, best_mode):
                best_metric = current
                best_epoch = epoch
                torch.save(
                    checkpoint_state(model, optimizer, cfg, epoch, global_step, best_metric, best_epoch),
                    save_dir / "best.pth",
                )
                print(f"[Best] epoch={epoch} {best_metric_name}={best_metric:.4f}", flush=True)
            print(
                f"[Epoch {epoch}] loss={metrics['loss']:.4f}, "
                f"loss_coord={metrics['loss_coord']:.4f}, "
                f"loss_vis={metrics['loss_vis']:.4f}, "
                f"loss_conf={metrics['loss_conf']:.4f}",
                flush=True,
            )

        if max_steps is not None and global_step >= max_steps:
            break

    if bool(train_cfg.get("save_last", True)):
        final_epoch = max(start_epoch, last_epoch)
        torch.save(
            checkpoint_state(model, optimizer, cfg, final_epoch, global_step, best_metric, best_epoch),
            save_dir / "final.pth",
        )
        torch.save(model.state_dict(), save_dir / "final_model_weights.pth")
        print(f"Saved final checkpoint to {save_dir / 'final.pth'}", flush=True)


if __name__ == "__main__":
    main()
