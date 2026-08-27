"""Convert a grayscale multi-frame TIFF stack to an animated GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def tiff_to_gif(
    input_path: Path,
    output_path: Path,
    frame_duration_ms: int = 200,
) -> None:
    """Convert a uint8 TIFF stack with shape (frames, height, width) to GIF."""

    movie = tifffile.imread(input_path)

    if movie.ndim != 3:
        raise ValueError(
            f"Expected TIFF shape (frames, height, width), got {movie.shape}"
        )

    if movie.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 TIFF, got {movie.dtype}"
        )

    frames = [
        Image.fromarray(frame, mode="L")
        for frame in movie
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )

    print(
        f"Wrote {output_path} "
        f"({len(frames)} frames, {frame_duration_ms} ms/frame)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a multi-frame TIFF stack to an animated GIF."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input TIFF file",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output GIF file",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=200,
        help="Duration of each GIF frame in milliseconds (default: 200)",
    )

    args = parser.parse_args()

    tiff_to_gif(
        input_path=args.input,
        output_path=args.output,
        frame_duration_ms=args.duration_ms,
    )


if __name__ == "__main__":
    main()
