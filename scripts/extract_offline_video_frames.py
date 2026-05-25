#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image

import sys
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from alpamayo1_5 import helper


def extract_video_to_jpgs(
    video_path: Path,
    output_dir: Path,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = sorted(output_dir.glob("*.jpg"))
        if existing:
            print(f"[skip] {video_path.name} -> {output_dir} already has {len(existing)} jpg files")
            return len(existing)

    for old_file in output_dir.glob("*.jpg"):
        old_file.unlink()

    frame_count = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            rgb = frame.to_ndarray(format="rgb24")
            image = Image.fromarray(rgb)
            out_path = output_dir / f"{frame_idx:06d}.jpg"
            image.save(out_path, quality=jpeg_quality)
            frame_count += 1

    print(f"[done] {video_path.name} -> {output_dir} | frames={frame_count}")
    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Extract offline camera mp4 files into jpg frame sequences.")
    parser.add_argument("--clip-dir", type=Path, required=True, help="Offline clip directory")
    parser.add_argument(
        "--output-root-name",
        type=str,
        default="extracted_frames",
        help="Subdirectory name under clip-dir for extracted frames",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames")
    args = parser.parse_args()

    clip_dir = args.clip_dir.expanduser().resolve()
    if not clip_dir.exists():
        raise FileNotFoundError(f"Clip dir does not exist: {clip_dir}")

    camera_paths = helper.discover_offline_camera_files(clip_dir)
    if not camera_paths:
        raise ValueError(f"No camera mp4 files found in {clip_dir}")

    output_root = clip_dir / args.output_root_name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"clip_dir={clip_dir}")
    print(f"output_root={output_root}")
    print(f"num_cameras={len(camera_paths)}")

    total_frames = 0
    for camera_path in camera_paths:
        # Use the mp4 stem as subdirectory name
        camera_output_dir = output_root / camera_path.stem
        total_frames += extract_video_to_jpgs(
            video_path=camera_path,
            output_dir=camera_output_dir,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )

    print(f"[summary] total_frames_written={total_frames}")


if __name__ == "__main__":
    main()