"""Animated replay of the showcase scenario: docs/img/replay.gif and out/replay.mp4.

Phases: scanning (7 poses), mapping (perceived traversability), planning (TerraNav and
Baseline-Occ), executing (two 0.3 m robots at the same speed, rough-exposure counters), final
hold. A replay of the scan-once, plan-once pipeline's products, not a live simulation.
MP4 via imageio-ffmpeg (bundled ffmpeg), 1280x720; GIF via Pillow, 800 px wide, one palette.
Run: uv run python scripts/make_replay.py [--showcase docs/img/showcase.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from terranav.config import SimConfig
from terranav.sim.showcase import load_checked_run
from terranav.viz.figures import PLANNER_COLOURS
from terranav.viz.showcase3d import (
    GOAL_COLOUR,
    LETHAL_COLOUR,
    PLANNER_BRIGHT,
    POINT_COLOUR,
    START_COLOUR,
    ReplayTiming,
    replay_frames,
)

GIF_WIDTH = 800  # GIF frames are downscaled to this width (MP4 keeps the render size)
GIF_COLOURS = 160
PALETTE_SAMPLES = 12
KEY_COLOURS = (
    *PLANNER_COLOURS.values(),
    *PLANNER_BRIGHT.values(),
    START_COLOUR,
    GOAL_COLOUR,
    LETHAL_COLOUR,
    POINT_COLOUR,
)


def write_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """One adaptive palette for all frames, built from PALETTE_SAMPLES frames spread over the
    whole replay (every phase's colours); no dithering, so unchanged regions stay identical
    between frames and compress well."""
    h, w = frames[0].shape[:2]
    size = (GIF_WIDTH, round(h * GIF_WIDTH / w))
    small = [Image.fromarray(f).resize(size, Image.Resampling.LANCZOS) for f in frames]
    picks = np.linspace(0, len(small) - 1, PALETTE_SAMPLES).round().astype(int)
    # swatches of the scene's fixed colours, so small objects (goal sphere) keep their colour
    rgb = [tuple(int(c[k : k + 2], 16) for k in (1, 3, 5)) for c in KEY_COLOURS]
    swatch = np.repeat(np.array(rgb, dtype=np.uint8)[:, None, :], size[0], axis=1)
    swatches = np.repeat(swatch, 20, axis=0)
    montage = Image.fromarray(np.vstack([np.asarray(small[i]) for i in picks] + [swatches]))
    palette = montage.quantize(GIF_COLOURS, method=Image.Quantize.MEDIANCUT)
    quant = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in small]
    quant[0].save(
        path,
        save_all=True,
        append_images=quant[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
    )


def write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    h, w = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (w, h),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=8,
        ffmpeg_log_level="error",
    )
    writer.send(None)  # start the generator
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showcase", type=Path, default=Path("docs/img/showcase.json"))
    parser.add_argument("--gif", type=Path, default=Path("docs/img/replay.gif"))
    parser.add_argument("--mp4", type=Path, default=Path("out/replay.mp4"))
    args = parser.parse_args()
    show, run = load_checked_run(args.showcase)
    timing = ReplayTiming()
    title = f"{show['family']} dev seed {show['seed']}, alpha = {show['alpha']:g}"
    frames = list(replay_frames(run, SimConfig().map_grid, title, timing))
    print(f"{len(frames)} frames = {len(frames) / timing.fps:.1f} s at {timing.fps} fps")
    for path in (args.gif, args.mp4):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(frames, args.mp4, timing.fps)
    write_gif(frames, args.gif, timing.fps)
    for path in (args.gif, args.mp4):
        print(f"Saved {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
