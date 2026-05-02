# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualization utilities for point tracking."""

import colorsys
import os
import random
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


# Bremm 2D colormap for position-based coloring
# This creates a smooth 2D color gradient based on x,y position
BREMM_COLORMAP = None  # Lazy loaded


def _create_bremm_colormap():
    """Create a 2D colormap programmatically (Bremm-style).
    
    This creates a smooth 2D color gradient where:
    - X position maps to hue variation
    - Y position maps to saturation/value variation
    """
    size = 256
    colormap = np.zeros((size, size, 3), dtype=np.uint8)
    
    for y in range(size):
        for x in range(size):
            # Normalize to [0, 1]
            nx = x / (size - 1)
            ny = y / (size - 1)
            
            # Create a 2D color mapping using HSV
            # Hue varies with x, saturation/value with y
            hue = (nx * 0.8 + ny * 0.2) % 1.0  # Mix of x and y for hue
            saturation = 0.6 + 0.4 * (1 - ny)  # Higher saturation at top
            value = 0.7 + 0.3 * nx  # Higher value on right
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colormap[y, x] = [int(c * 255) for c in rgb]
    
    return colormap


def _get_bremm_colormap():
    """Get or create the bremm colormap."""
    global BREMM_COLORMAP
    if BREMM_COLORMAP is None:
        # Try to load from file first
        colormap_file = os.path.join(os.path.dirname(__file__), "bremm.png")
        if os.path.exists(colormap_file):
            BREMM_COLORMAP = (plt.imread(colormap_file) * 255).astype(np.uint8)
            if BREMM_COLORMAP.shape[2] == 4:  # RGBA
                BREMM_COLORMAP = BREMM_COLORMAP[:, :, :3]
        else:
            BREMM_COLORMAP = _create_bremm_colormap()
    return BREMM_COLORMAP


def get_2d_colors(xys: np.ndarray, H: int, W: int) -> np.ndarray:
    """Get colors based on 2D position using Bremm colormap.
    
    This creates position-dependent colors where nearby points have
    similar colors, useful for visualizing spatial coherence.
    
    Args:
        xys: Point coordinates [N, 2] in pixel space (x, y)
        H: Image height
        W: Image width
    
    Returns:
        Array of RGB colors [N, 3] as uint8
    """
    colormap = _get_bremm_colormap()
    height, width = colormap.shape[:2]
    
    N = xys.shape[0]
    output = np.zeros((N, 3), dtype=np.uint8)
    
    # Normalize coordinates to [0, 1]
    xys_norm = xys.copy().astype(np.float32)
    xys_norm[:, 0] = xys_norm[:, 0] / max(W - 1, 1)
    xys_norm[:, 1] = xys_norm[:, 1] / max(H - 1, 1)
    
    # Clip to valid range
    xys_norm = np.clip(xys_norm, 0, 1)
    
    # Map to colormap coordinates
    for i in range(N):
        x, y = xys_norm[i]
        xp = int((width - 1) * x)
        yp = int((height - 1) * y)
        output[i] = colormap[yp, xp]
    
    return output


def get_colors_from_cmap(num_colors: int, cmap: str = "gist_rainbow") -> np.ndarray:
    """Gets colormap for points using matplotlib colormap.
    
    Args:
        num_colors: Number of colors to generate
        cmap: Matplotlib colormap name (e.g., "gist_rainbow", "jet", "turbo")
    
    Returns:
        Array of RGB colors [num_colors, 3] as uint8
    """
    cmap_ = matplotlib.colormaps.get_cmap(cmap)
    colors = []
    for i in range(num_colors):
        c = cmap_(i / float(num_colors))
        colors.append((int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))
    return np.array(colors)


def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap: Optional[Union[List[Tuple[int, int, int]], np.ndarray]] = None,
    rate: int = 1,
    show_bkg: bool = True,
) -> np.ndarray:
    """Paint point tracks on video frames using GPU-accelerated scatter.

    Args:
        frames: Video frames [T, H, W, C] in uint8
        point_tracks: Track coordinates [P, T, 2] (x, y)
        visibles: Visibility mask [P, T]
        colormap: Optional list/array of RGB colors for each point
        rate: Subsampling rate for visualization (affects point size)
        show_bkg: Whether to show background (True) or black out (False)

    Returns:
        Painted frames [T, H, W, C] in uint8
    """
    print("Starting visualization...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames_t = (
        torch.from_numpy(frames).float().permute(0, 3, 1, 2).to(device)
    )  # [T,C,H,W]

    if show_bkg:
        frames_t = frames_t * 0.5  # darken to see tracks better
    else:
        frames_t = frames_t * 0.0  # black out background

    point_tracks_t = torch.from_numpy(point_tracks).to(device)  # [P,T,2]
    visibles_t = torch.from_numpy(visibles).to(device)  # [P,T]
    T, C, H, W = frames_t.shape
    P = point_tracks.shape[0]

    # Use gist_rainbow colormap (matching app3.py behavior)
    if colormap is None:
        colormap = get_colors_from_cmap(P, "gist_rainbow")
    colors = torch.tensor(colormap, dtype=torch.float32, device=device)  # [P,3]

    # Adjust radius based on rate
    if rate == 1:
        radius = 1
    elif rate == 2:
        radius = 1
    elif rate == 4:
        radius = 2
    elif rate == 8:
        radius = 4
    else:
        radius = 6

    sharpness = 0.15 + 0.05 * np.log2(rate)

    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1)
    icon = icon.view(1, D, D)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")

    for t in range(T):
        mask = visibles_t[:, t]
        if mask.sum() == 0:
            continue
        xy = point_tracks_t[mask, t] + 0.5
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]
        N = xy.shape[0]
        cx = xy[:, 0].long()
        cy = xy[:, 1].long()
        x_grid = cx[:, None, None] + disp_x
        y_grid = cy[:, None, None] + disp_y
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(N, D, D)[valid]
        colors_valid = (
            colors_now[:, :, None, None]
            .expand(N, 3, D, D)
            .permute(1, 0, 2, 3)[:, valid]
        )
        idx_flat = (y_valid * W + x_valid).long()

        accum = torch.zeros_like(frames_t[t])
        weight = torch.zeros(1, H * W, device=device)
        img_flat = accum.view(C, -1)
        weighted_colors = colors_valid * icon_weights
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        alpha = weight.clamp(0, 1)
        accum = accum / (weight + 1e-6)
        frames_t[t] = frames_t[t] * (1 - alpha) + accum * alpha

    print("Visualization done.")
    return frames_t.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


class TrackVisualizer:
    """CoTracker/ETAP-style point-track visualizer.

    The public track format is [B, T, N, 2], matching CoTracker and ETAP.
    """

    def __init__(
        self,
        save_dir: str = "output/vis_cowtracker",
        fps: int = 10,
        mode: str = "rainbow",
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = -1,
        pad_value: int = 0,
        grayscale: bool = False,
    ) -> None:
        self.save_dir = save_dir
        self.fps = int(fps)
        self.mode = mode
        self.linewidth = int(linewidth)
        self.show_first_frame = int(show_first_frame)
        self.tracks_leave_trace = int(tracks_leave_trace)
        self.pad_value = int(pad_value)
        self.grayscale = bool(grayscale)
        os.makedirs(self.save_dir, exist_ok=True)

    def visualize(
        self,
        video: Union[np.ndarray, torch.Tensor],
        tracks: Union[np.ndarray, torch.Tensor],
        visibility: Union[np.ndarray, torch.Tensor],
        filename: Optional[str] = "video",
        query_frame: int = 0,
        gt_tracks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gt_visibility: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """Render and optionally save a point-track video.

        Args:
            video: [B,T,3,H,W], [T,3,H,W], or [T,H,W,3] RGB frames.
            tracks: [B,T,N,2] or [T,N,2] xy tracks.
            visibility: [B,T,N] or [T,N] visibility mask.
            filename: Output filename stem/path. If None, only returns frames.
            query_frame: Frame used to assign stable rainbow colors.
            gt_tracks: Optional GT tracks in the same format, drawn as red crosses.
            gt_visibility: Optional GT visibility mask.

        Returns:
            Rendered frames [T,H,W,3] as uint8, including the repeated first frame.
        """
        frames = self._as_video_numpy(video)
        pred_tracks = self._as_tracks_numpy(tracks)
        pred_visibility = self._as_visibility_numpy(visibility)
        gt_tracks_np = None if gt_tracks is None else self._as_tracks_numpy(gt_tracks)
        gt_visibility_np = None if gt_visibility is None else self._as_visibility_numpy(gt_visibility)

        if pred_tracks.shape[:2] != pred_visibility.shape[:2]:
            raise ValueError(
                f"tracks shape {pred_tracks.shape} and visibility shape "
                f"{pred_visibility.shape} are incompatible"
            )
        if frames.shape[0] != pred_tracks.shape[0]:
            raise ValueError(
                f"video has {frames.shape[0]} frames but tracks have {pred_tracks.shape[0]}"
            )

        rendered = self._draw_tracks(
            frames,
            pred_tracks,
            pred_visibility,
            int(query_frame),
            gt_tracks_np,
            gt_visibility_np,
        )
        if self.show_first_frame > 0 and rendered.shape[0] > 0:
            rendered = np.concatenate(
                [np.repeat(rendered[:1], self.show_first_frame, axis=0), rendered],
                axis=0,
            )

        if filename is not None:
            self.save_video(rendered, filename)
        return rendered

    @staticmethod
    def _as_numpy(value: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _as_video_numpy(self, video: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        frames = self._as_numpy(video)
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim != 4:
            raise ValueError(f"video must have 4 or 5 dimensions, got {frames.shape}")
        if frames.shape[1] == 3:
            frames = np.transpose(frames, (0, 2, 3, 1))
        if frames.shape[-1] != 3:
            raise ValueError(f"video must contain RGB frames, got {frames.shape}")
        frames = frames.astype(np.float32)
        if frames.size > 0 and frames.max() <= 1.0:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)
        if self.grayscale:
            gray = np.mean(frames, axis=-1, keepdims=True).astype(np.uint8)
            frames = np.repeat(gray, 3, axis=-1)
        if self.pad_value > 0:
            frames = np.pad(
                frames,
                ((0, 0), (self.pad_value, self.pad_value), (self.pad_value, self.pad_value), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        return frames

    def _as_tracks_numpy(self, tracks: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        tracks = self._as_numpy(tracks).astype(np.float32)
        if tracks.ndim == 4:
            tracks = tracks[0]
        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise ValueError(f"tracks must have shape [B,T,N,2] or [T,N,2], got {tracks.shape}")
        if self.pad_value > 0:
            tracks = tracks.copy()
            tracks[..., 0] += self.pad_value
            tracks[..., 1] += self.pad_value
        return tracks

    def _as_visibility_numpy(self, visibility: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        visibility = self._as_numpy(visibility)
        if visibility.ndim == 4 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        if visibility.ndim == 3:
            visibility = visibility[0]
        if visibility.ndim != 2:
            raise ValueError(
                f"visibility must have shape [B,T,N], [B,T,N,1], or [T,N], got {visibility.shape}"
            )
        return visibility.astype(bool)

    def _colors_for_tracks(self, tracks: np.ndarray, query_frame: int, height: int) -> np.ndarray:
        num_points = tracks.shape[1]
        if num_points == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        query_frame = int(np.clip(query_frame, 0, max(tracks.shape[0] - 1, 0)))
        if self.mode == "rainbow":
            cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
            y = tracks[query_frame, :, 1]
            y = np.nan_to_num(y, nan=0.0, posinf=float(height - 1), neginf=0.0)
            y = np.clip(y / max(height - 1, 1), 0.0, 1.0)
            colors = [cmap(float(v))[:3] for v in y]
            return (np.asarray(colors) * 255).astype(np.uint8)
        if self.mode == "bremm":
            return get_2d_colors(tracks[query_frame], height, max(1, int(np.nanmax(tracks[..., 0]) + 1)))
        return get_colors_from_cmap(num_points, "gist_rainbow")

    def _draw_tracks(
        self,
        frames: np.ndarray,
        tracks: np.ndarray,
        visibility: np.ndarray,
        query_frame: int,
        gt_tracks: Optional[np.ndarray],
        gt_visibility: Optional[np.ndarray],
    ) -> np.ndarray:
        num_frames, height, width = frames.shape[:3]
        colors = self._colors_for_tracks(tracks, query_frame, height)
        rendered = []
        radius = max(2, self.linewidth + 1)

        for t in range(num_frames):
            image = Image.fromarray(frames[t].copy())
            draw = ImageDraw.Draw(image)
            if self.tracks_leave_trace < 0:
                start = 0
            elif self.tracks_leave_trace == 0:
                start = t
            else:
                start = max(0, t - self.tracks_leave_trace)

            for n, color_arr in enumerate(colors):
                color = tuple(int(c) for c in color_arr)
                for k in range(start, t):
                    if not (visibility[k, n] and visibility[k + 1, n]):
                        continue
                    p0 = tracks[k, n]
                    p1 = tracks[k + 1, n]
                    if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                        continue
                    draw.line(
                        [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))],
                        fill=color,
                        width=self.linewidth,
                    )

                p = tracks[t, n]
                if not np.isfinite(p).all():
                    continue
                xy = (
                    float(p[0] - radius),
                    float(p[1] - radius),
                    float(p[0] + radius),
                    float(p[1] + radius),
                )
                if visibility[t, n]:
                    draw.ellipse(xy, fill=color, outline=color)
                else:
                    draw.ellipse(xy, outline=color, width=max(1, self.linewidth))

            if gt_tracks is not None:
                gt_vis_t = None if gt_visibility is None else gt_visibility[t]
                self._draw_gt_crosses(draw, gt_tracks[t], gt_vis_t)

            rendered.append(np.asarray(image))
        return np.stack(rendered, axis=0)

    @staticmethod
    def _draw_gt_crosses(
        draw: ImageDraw.ImageDraw,
        tracks_t: np.ndarray,
        visibility_t: Optional[np.ndarray],
    ) -> None:
        color = (255, 32, 32)
        radius = 4
        for idx, point in enumerate(tracks_t):
            if visibility_t is not None and not bool(visibility_t[idx]):
                continue
            if not np.isfinite(point).all():
                continue
            x, y = float(point[0]), float(point[1])
            draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=color, width=2)
            draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=color, width=2)

    def save_video(self, frames: np.ndarray, filename: str) -> str:
        import imageio.v2 as imageio

        output_path = filename
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.save_dir, output_path)
        root, ext = os.path.splitext(output_path)
        if not ext:
            output_path = root + ".mp4"
            ext = ".mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with imageio.get_writer(output_path, fps=self.fps, macro_block_size=1) as writer:
                for frame in frames:
                    writer.append_data(frame)
            return output_path
        except Exception:
            fallback = root + ".gif"
            imageio.mimsave(fallback, list(frames), fps=self.fps)
            return fallback
