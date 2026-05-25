# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load an offline local clip into the Alpamayo inference format.

This module keeps the public ``load_offline_dataset(...)`` signature compatible
with the existing notebooks, while improving:

1. Time-alignment diagnostics
2. Coordinate-frame naming / structure clarity
3. Clip-level caching for pose parsing and video decoding

Coordinate-frame summary used in this file
------------------------------------------
1. Global ENU frame
   - Derived from lat/lon/alt
   - x=east, y=north, z=up

2. t0-local frame
   - Origin at ego pose of t0
   - Axes aligned with ego orientation at t0
   - Used as the raw local trajectory frame for xyz

3. Model/action rotation frame
   - A fixed planar transform is applied to rotations only
   - This preserves the empirically validated behavior of the original code
   - xyz remains in raw t0-local frame for compatibility

Notebook-side plotting / evaluation may still apply an additional eval-side
xy rotation to compare raw offline GT/history against model outputs in a
common +x-forward plotting convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
import torch.nn.functional as F
from einops import rearrange

from alpamayo1_5 import helper


DEFAULT_FPS = 30.0
DEFAULT_VIDEO_FRAME0_GPS_TIME_SOD = 175484.98


def _debug_print(debug: bool, *args: Any, **kwargs: Any) -> None:
    if debug:
        print(*args, **kwargs)


def _geodetic_to_local_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
) -> np.ndarray:
    """Convert geodetic coordinates to a local ENU metric frame.

    Uses a small-area tangent-plane approximation around the first valid sample:
      east  = dlon * R * cos(lat0)
      north = dlat * R
      up    = alt - alt0

    This is appropriate for local driving clips and preserves meter-scale motion.
    """
    earth_radius_m = 6378137.0

    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    alt_m = np.asarray(alt_m, dtype=np.float64)

    lat0_rad = np.deg2rad(lat_deg[0])
    lon0_rad = np.deg2rad(lon_deg[0])

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    east_m = (lon_rad - lon0_rad) * earth_radius_m * np.cos(lat0_rad)
    north_m = (lat_rad - lat0_rad) * earth_radius_m
    up_m = alt_m - alt_m[0]

    # ENU ordering:
    #   x = east
    #   y = north
    #   z = up
    return np.stack([east_m, north_m, up_m], axis=-1)


def _load_ego_pose_log(
    ego_log_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, spt.Rotation]:
    """Load ego poses from a mixed NMEA-like ego_pos.log.

    This parser uses only ``$GPFPD`` rows and ignores all other rows such as
    ``$GTIMU``.
    """
    ego_log_path = Path(ego_log_path).expanduser()
    if not ego_log_path.exists():
        raise FileNotFoundError(f"Missing ego pose log: {ego_log_path}")

    rows = []

    with ego_log_path.open("r") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if "*" in line:
                line = line.split("*", 1)[0]

            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue

            if parts[0] != "$GPFPD":
                continue

            if len(parts) < 12:
                raise ValueError(
                    f"Expected at least 12 comma-separated fields in $GPFPD line "
                    f"{line_idx} of {ego_log_path}, got {len(parts)}: {parts}"
                )

            try:
                gps_week = float(parts[1])
                sod = float(parts[2])
                heading_deg = float(parts[3])
                pitch_deg = float(parts[4])
                roll_deg = float(parts[5])
                lat_deg = float(parts[6])
                lon_deg = float(parts[7])
                alt_m = float(parts[8])
                ve_mps = float(parts[9])
                vn_mps = float(parts[10])
                vu_mps = float(parts[11])
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse numeric fields from $GPFPD line {line_idx}: {raw_line}"
                ) from e

            rows.append(
                {
                    "gps_week": gps_week,
                    "sod": sod,
                    "heading_deg": heading_deg,
                    "pitch_deg": pitch_deg,
                    "roll_deg": roll_deg,
                    "lat_deg": lat_deg,
                    "lon_deg": lon_deg,
                    "alt_m": alt_m,
                    "ve_mps": ve_mps,
                    "vn_mps": vn_mps,
                    "vu_mps": vu_mps,
                }
            )

    if not rows:
        raise ValueError(f"No valid $GPFPD rows found in ego pose log: {ego_log_path}")

    df = pd.DataFrame(rows).sort_values("sod").reset_index(drop=True)

    timestamps_sod = df["sod"].to_numpy(dtype=np.float64)
    heading_deg = df["heading_deg"].to_numpy(dtype=np.float64)
    pitch_deg = df["pitch_deg"].to_numpy(dtype=np.float64)
    roll_deg = df["roll_deg"].to_numpy(dtype=np.float64)
    lat_deg = df["lat_deg"].to_numpy(dtype=np.float64)
    lon_deg = df["lon_deg"].to_numpy(dtype=np.float64)
    alt_m = df["alt_m"].to_numpy(dtype=np.float64)

    if not np.all(np.diff(timestamps_sod) >= 0):
        raise ValueError("Ego pose timestamps are not sorted in non-decreasing order.")

    xyz_global_enu = _geodetic_to_local_enu(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
    )

    # Heading in ego_pos.log is already ENU yaw:
    # east = 0 deg, counterclockwise positive
    yaw_deg = heading_deg

    rotations_global_enu = spt.Rotation.from_euler(
        "xyz",
        np.stack([roll_deg, pitch_deg, yaw_deg], axis=-1),
        degrees=True,
    )

    return timestamps_sod, xyz_global_enu, rotations_global_enu


def _interpolate_positions(
    timestamps_sod: np.ndarray,
    positions: np.ndarray,
    query_times_sod: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate positions at query times."""
    t = np.asarray(timestamps_sod, dtype=np.float64)
    x = np.asarray(query_times_sod, dtype=np.float64)

    if np.any(x < t[0]) or np.any(x > t[-1]):
        raise ValueError(
            f"Query time out of pose range: pose range [{t[0]:.3f}, {t[-1]:.3f}], "
            f"query range [{x.min():.3f}, {x.max():.3f}]"
        )

    return np.stack(
        [np.interp(x, t, positions[:, dim]) for dim in range(positions.shape[1])],
        axis=-1,
    )


def _interpolate_rotations(
    timestamps_sod: np.ndarray,
    rotations: spt.Rotation,
    query_times_sod: np.ndarray,
) -> np.ndarray:
    """Slerp rotations at query times."""
    t = np.asarray(timestamps_sod, dtype=np.float64)
    x = np.asarray(query_times_sod, dtype=np.float64)

    if np.any(x < t[0]) or np.any(x > t[-1]):
        raise ValueError(
            f"Query time out of pose range: pose range [{t[0]:.3f}, {t[-1]:.3f}], "
            f"query range [{x.min():.3f}, {x.max():.3f}]"
        )

    slerp = spt.Slerp(t, rotations)
    return slerp(x).as_matrix()


def _lookup_pose_samples(
    timestamps_sod: np.ndarray,
    positions: np.ndarray,
    rotations: spt.Rotation,
    query_times_sod: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample interpolated ego poses for query times."""
    interp_positions = _interpolate_positions(timestamps_sod, positions, query_times_sod)
    interp_rotations = _interpolate_rotations(timestamps_sod, rotations, query_times_sod)
    return interp_positions, interp_rotations


def _decode_video_frames(video_path: str | Path) -> list[np.ndarray]:
    """Decode all frames from an offline mp4 into a list of uint8 RGB images."""
    video_path = Path(video_path).expanduser()
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"No video frames decoded from {video_path}")
    return frames


def _sample_decoded_video_frames(
    decoded_frames: list[np.ndarray],
    frame_indices: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, int]:
    """Sample selected frames from an already-decoded frame list."""
    max_index = len(decoded_frames) - 1
    clipped_indices = np.clip(frame_indices.astype(np.int64), 0, max_index)
    sampled = np.stack([decoded_frames[index] for index in clipped_indices], axis=0)
    return (
        rearrange(torch.from_numpy(sampled), "t h w c -> t c h w"),
        clipped_indices,
        len(decoded_frames),
    )


def _resize_frames(
    frames: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Resize frames of shape [T, C, H, W] to a common spatial size."""
    if frames.ndim != 4:
        raise ValueError(f"Expected frames to have shape [T, C, H, W], got {frames.shape}")
    frames = frames.float()
    frames = F.interpolate(
        frames,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    return frames.round().clamp(0, 255).to(torch.uint8)


def _convert_global_enu_pose_to_t0_local_frame(
    xyz_global_enu: np.ndarray,
    rot_global_enu_mats: np.ndarray,
    t0_xyz_global_enu: np.ndarray,
    t0_rot_global_enu: spt.Rotation,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert global ENU poses into a t0-centered local ego frame."""
    t0_rot_inv = t0_rot_global_enu.inv()
    xyz_t0_local = t0_rot_inv.apply(xyz_global_enu - t0_xyz_global_enu)
    rot_t0_local = (t0_rot_inv * spt.Rotation.from_matrix(rot_global_enu_mats)).as_matrix()
    return xyz_t0_local, rot_t0_local


def _convert_t0_local_rot_to_model_action_frame(
    rot_t0_local: np.ndarray,
) -> np.ndarray:
    """Convert raw t0-local rotations into the model/action rotation convention."""
    R_T0_LOCAL_TO_MODEL_ACTION = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return np.einsum("ij,tjk->tik", R_T0_LOCAL_TO_MODEL_ACTION, rot_t0_local)


def _build_time_alignment_summary(
    *,
    t0_sod: float,
    pose_times_sod: np.ndarray,
    history_times_sod: np.ndarray,
    future_times_sod: np.ndarray,
    image_timestamps_sod: np.ndarray,
    requested_video_frame_indices: np.ndarray,
    actual_video_frame_indices: np.ndarray,
    num_clipped_frames_total: int,
    num_cameras: int,
    num_frames_per_camera: int,
) -> dict[str, Any]:
    """Build a structured time-alignment diagnostic summary."""
    summary = {
        "t0_sod": float(t0_sod),
        "num_cameras": int(num_cameras),
        "num_frames_per_camera": int(num_frames_per_camera),
        "pose_time_range_sod": (
            float(pose_times_sod[0]),
            float(pose_times_sod[-1]),
        ),
        "history_time_range_sod": (
            float(history_times_sod[0]),
            float(history_times_sod[-1]),
        ),
        "future_time_range_sod": (
            float(future_times_sod[0]),
            float(future_times_sod[-1]),
        ),
        "image_time_range_sod": (
            float(image_timestamps_sod[0]),
            float(image_timestamps_sod[-1]),
        ),
        "requested_video_frame_index_range": (
            int(requested_video_frame_indices.min()),
            int(requested_video_frame_indices.max()),
        ),
        "actual_video_frame_index_range": (
            int(actual_video_frame_indices.min()),
            int(actual_video_frame_indices.max()),
        ),
        "num_clipped_frames_total": int(num_clipped_frames_total),
        "pose_margin_left_sec": float(history_times_sod[0] - pose_times_sod[0]),
        "pose_margin_right_sec": float(pose_times_sod[-1] - future_times_sod[-1]),
    }
    return summary


def _debug_print_time_alignment_summary(debug: bool, summary: dict[str, Any]) -> None:
    """Pretty-print a compact time-alignment diagnostic summary."""
    if not debug:
        return

    print("[Time alignment summary]")
    print(f"  t0_sod: {summary['t0_sod']:.6f}")
    print(
        "  pose_time_range_sod: "
        f"[{summary['pose_time_range_sod'][0]:.6f}, {summary['pose_time_range_sod'][1]:.6f}]"
    )
    print(
        "  history_time_range_sod: "
        f"[{summary['history_time_range_sod'][0]:.6f}, {summary['history_time_range_sod'][1]:.6f}]"
    )
    print(
        "  future_time_range_sod: "
        f"[{summary['future_time_range_sod'][0]:.6f}, {summary['future_time_range_sod'][1]:.6f}]"
    )
    print(
        "  image_time_range_sod: "
        f"[{summary['image_time_range_sod'][0]:.6f}, {summary['image_time_range_sod'][1]:.6f}]"
    )
    print(
        "  requested_video_frame_index_range: "
        f"[{summary['requested_video_frame_index_range'][0]}, "
        f"{summary['requested_video_frame_index_range'][1]}]"
    )
    print(
        "  actual_video_frame_index_range: "
        f"[{summary['actual_video_frame_index_range'][0]}, "
        f"{summary['actual_video_frame_index_range'][1]}]"
    )
    print(f"  num_clipped_frames_total: {summary['num_clipped_frames_total']}")
    print(f"  pose_margin_left_sec: {summary['pose_margin_left_sec']:.6f}")
    print(f"  pose_margin_right_sec: {summary['pose_margin_right_sec']:.6f}")


@dataclass
class OfflineClipCache:
    clip_dir: Path
    camera_paths: list[Path]
    pose_times_sod: np.ndarray
    pose_xyz_global_enu: np.ndarray
    pose_rot_global_enu: spt.Rotation
    decoded_frames_by_camera: dict[str, list[np.ndarray]]

    @classmethod
    def build(cls, clip_dir: str | Path, debug: bool = False) -> "OfflineClipCache":
        clip_dir = Path(clip_dir).expanduser()
        ego_log_path = clip_dir / "ego_pos.log"

        if not clip_dir.exists():
            raise FileNotFoundError(f"Offline clip directory does not exist: {clip_dir}")
        if not ego_log_path.exists():
            raise FileNotFoundError(f"Missing ego pose log: {ego_log_path}")

        camera_paths = helper.discover_offline_camera_files(clip_dir)
        pose_times_sod, pose_xyz_global_enu, pose_rot_global_enu = _load_ego_pose_log(ego_log_path)

        decoded_frames_by_camera = {}
        for camera_path in camera_paths:
            _debug_print(debug, f"Decoding video once for cache: {camera_path}")
            decoded_frames_by_camera[str(camera_path)] = _decode_video_frames(camera_path)

        return cls(
            clip_dir=clip_dir,
            camera_paths=camera_paths,
            pose_times_sod=pose_times_sod,
            pose_xyz_global_enu=pose_xyz_global_enu,
            pose_rot_global_enu=pose_rot_global_enu,
            decoded_frames_by_camera=decoded_frames_by_camera,
        )


_CLIP_CACHE_REGISTRY: dict[str, OfflineClipCache] = {}


def get_or_build_offline_clip_cache(
    clip_dir: str | Path,
    *,
    debug: bool = False,
    force_rebuild: bool = False,
) -> OfflineClipCache:
    """Get or build a process-local cache for an offline clip directory."""
    clip_dir = str(Path(clip_dir).expanduser().resolve())
    if force_rebuild or clip_dir not in _CLIP_CACHE_REGISTRY:
        _CLIP_CACHE_REGISTRY[clip_dir] = OfflineClipCache.build(clip_dir, debug=debug)
    return _CLIP_CACHE_REGISTRY[clip_dir]


def clear_offline_clip_cache(clip_dir: str | Path | None = None) -> None:
    """Clear one cached clip, or all cached clips if clip_dir is None."""
    global _CLIP_CACHE_REGISTRY
    if clip_dir is None:
        _CLIP_CACHE_REGISTRY = {}
        return

    clip_dir = str(Path(clip_dir).expanduser().resolve())
    _CLIP_CACHE_REGISTRY.pop(clip_dir, None)


def load_offline_dataset(
    clip_dir: str | Path,
    t0_sod: float,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    num_frames: int = 4,
    fps: float = DEFAULT_FPS,
    frame0_gps_time_sod: float = DEFAULT_VIDEO_FRAME0_GPS_TIME_SOD,
    debug: bool = False,
    image_size: tuple[int, int] = (448, 800),
) -> dict[str, Any]:
    """Load an offline clip folder into the Alpamayo inference format.

    Public return fields used by the existing notebooks remain compatible.

    This version also reuses clip-level caches so sliding-window notebook runs
    avoid repeated pose parsing and repeated full-video decoding.
    """
    cache = get_or_build_offline_clip_cache(clip_dir, debug=debug)
    clip_dir = cache.clip_dir

    _debug_print(debug, f"clip_dir={clip_dir}")
    _debug_print(
        debug,
        f"t0_sod={t0_sod:.6f}, fps={fps:.3f}, frame0_gps_time_sod={frame0_gps_time_sod:.6f}",
    )
    _debug_print(
        debug,
        "config="
        f"num_history_steps={num_history_steps}, num_future_steps={num_future_steps}, "
        f"time_step={time_step:.3f}, num_frames={num_frames}, image_size={image_size}",
    )
    _debug_print(debug, f"camera_files={[path.name for path in cache.camera_paths]}")

    pose_times_sod = cache.pose_times_sod
    pose_xyz_global_enu = cache.pose_xyz_global_enu
    pose_rot_global_enu = cache.pose_rot_global_enu

    _debug_print(
        debug,
        f"num_pose_samples={len(pose_times_sod)}, "
        f"pose_first_sod={pose_times_sod[0]:.6f}, pose_last_sod={pose_times_sod[-1]:.6f}",
    )
    _debug_print(
        debug,
        "pose_global_enu_examples="
        f"first={pose_xyz_global_enu[0].tolist()}, "
        f"last={pose_xyz_global_enu[-1].tolist()}",
    )

    history_offsets = np.arange(
        -(num_history_steps - 1) * time_step,
        time_step / 2,
        time_step,
        dtype=np.float64,
    )
    future_offsets = np.arange(
        time_step,
        (num_future_steps + 0.5) * time_step,
        time_step,
        dtype=np.float64,
    )
    history_times_sod = t0_sod + history_offsets
    future_times_sod = t0_sod + future_offsets
    all_query_times = np.concatenate([history_times_sod, future_times_sod], axis=0)

    if all_query_times.min() < pose_times_sod[0] or all_query_times.max() > pose_times_sod[-1]:
        raise ValueError(
            "Requested history/future window exceeds ego pose log range. "
            f"Pose range: [{pose_times_sod[0]:.3f}, {pose_times_sod[-1]:.3f}], "
            f"requested: [{all_query_times.min():.3f}, {all_query_times.max():.3f}]"
        )

    ego_history_xyz_global_enu, ego_history_rot_global_enu = _lookup_pose_samples(
        pose_times_sod,
        pose_xyz_global_enu,
        pose_rot_global_enu,
        history_times_sod,
    )
    ego_future_xyz_global_enu, ego_future_rot_global_enu = _lookup_pose_samples(
        pose_times_sod,
        pose_xyz_global_enu,
        pose_rot_global_enu,
        future_times_sod,
    )

    t0_xyz_global_enu = ego_history_xyz_global_enu[-1].copy()
    t0_rot_global_enu = spt.Rotation.from_matrix(ego_history_rot_global_enu[-1])

    ego_history_xyz_t0_local, ego_history_rot_t0_local = _convert_global_enu_pose_to_t0_local_frame(
        xyz_global_enu=ego_history_xyz_global_enu,
        rot_global_enu_mats=ego_history_rot_global_enu,
        t0_xyz_global_enu=t0_xyz_global_enu,
        t0_rot_global_enu=t0_rot_global_enu,
    )
    ego_future_xyz_t0_local, ego_future_rot_t0_local = _convert_global_enu_pose_to_t0_local_frame(
        xyz_global_enu=ego_future_xyz_global_enu,
        rot_global_enu_mats=ego_future_rot_global_enu,
        t0_xyz_global_enu=t0_xyz_global_enu,
        t0_rot_global_enu=t0_rot_global_enu,
    )

    ego_history_rot_model_frame = _convert_t0_local_rot_to_model_action_frame(
        ego_history_rot_t0_local
    )
    ego_future_rot_model_frame = _convert_t0_local_rot_to_model_action_frame(
        ego_future_rot_t0_local
    )

    _debug_print(debug, f"t0_xyz_global_enu={t0_xyz_global_enu.tolist()}")
    _debug_print(debug, f"history_last_t0_local_xyz={ego_history_xyz_t0_local[-1].tolist()}")
    _debug_print(debug, f"future_first_t0_local_xyz={ego_future_xyz_t0_local[0].tolist()}")
    _debug_print(debug, f"future_last_t0_local_xyz={ego_future_xyz_t0_local[-1].tolist()}")

    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_t0_local.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_model_frame.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )
    ego_future_xyz_tensor = (
        torch.from_numpy(ego_future_xyz_t0_local.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )
    ego_future_rot_tensor = (
        torch.from_numpy(ego_future_rot_model_frame.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )

    image_offsets = np.arange(
        -(num_frames - 1) * time_step,
        time_step / 2,
        time_step,
        dtype=np.float64,
    )
    image_timestamps_sod = t0_sod + image_offsets
    requested_video_frame_indices_1d = np.rint(
        (image_timestamps_sod - frame0_gps_time_sod) * fps
    ).astype(np.int64)
    relative_timestamps_1d = image_timestamps_sod - t0_sod

    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []
    actual_video_frame_indices_list = []
    clipped_frame_mask_list = []
    video_num_frames_list = []

    for camera_path in cache.camera_paths:
        decoded_frames = cache.decoded_frames_by_camera[str(camera_path)]
        frames_tensor, clipped_indices, num_decoded_frames = _sample_decoded_video_frames(
            decoded_frames,
            requested_video_frame_indices_1d,
        )
        cam_idx = helper.infer_camera_index(camera_path.name)

        frames_tensor = _resize_frames(frames_tensor, image_size)
        clipped_mask = clipped_indices != requested_video_frame_indices_1d

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(image_timestamps_sod.astype(np.float64)))
        actual_video_frame_indices_list.append(torch.from_numpy(clipped_indices.astype(np.int64)))
        clipped_frame_mask_list.append(torch.from_numpy(clipped_mask.astype(np.bool_)))
        video_num_frames_list.append(num_decoded_frames)

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    absolute_timestamps_sod = torch.stack(timestamps_list, dim=0)
    actual_video_frame_indices = torch.stack(actual_video_frame_indices_list, dim=0)
    clipped_frame_mask = torch.stack(clipped_frame_mask_list, dim=0)
    video_num_frames = torch.tensor(video_num_frames_list, dtype=torch.int64)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    absolute_timestamps_sod = absolute_timestamps_sod[sort_order]
    actual_video_frame_indices = actual_video_frame_indices[sort_order]
    clipped_frame_mask = clipped_frame_mask[sort_order]
    video_num_frames = video_num_frames[sort_order]

    relative_timestamps = torch.from_numpy(
        np.broadcast_to(
            relative_timestamps_1d[None, :],
            (len(camera_indices), num_frames),
        ).astype(np.float32)
    )
    requested_video_frame_indices = torch.from_numpy(
        np.broadcast_to(
            requested_video_frame_indices_1d[None, :],
            (len(camera_indices), num_frames),
        ).astype(np.int64)
    )

    num_clipped_frames_per_camera = clipped_frame_mask.sum(dim=1).to(torch.int64)
    num_clipped_frames_total = int(clipped_frame_mask.sum().item())

    time_alignment_summary = _build_time_alignment_summary(
        t0_sod=float(t0_sod),
        pose_times_sod=pose_times_sod,
        history_times_sod=history_times_sod,
        future_times_sod=future_times_sod,
        image_timestamps_sod=image_timestamps_sod,
        requested_video_frame_indices=requested_video_frame_indices.cpu().numpy(),
        actual_video_frame_indices=actual_video_frame_indices.cpu().numpy(),
        num_clipped_frames_total=num_clipped_frames_total,
        num_cameras=len(camera_indices),
        num_frames_per_camera=num_frames,
    )
    _debug_print_time_alignment_summary(debug, time_alignment_summary)

    _debug_print(
        debug,
        f"requested_image_timestamps_sod={image_timestamps_sod.tolist()}",
    )
    _debug_print(
        debug,
        f"requested_video_frame_indices_first_camera="
        f"{requested_video_frame_indices[0].tolist() if len(requested_video_frame_indices) else []}",
    )
    _debug_print(
        debug,
        f"actual_video_frame_indices_first_camera="
        f"{actual_video_frame_indices[0].tolist() if len(actual_video_frame_indices) else []}",
    )
    _debug_print(
        debug,
        f"clipped_frame_mask_first_camera="
        f"{clipped_frame_mask[0].tolist() if len(clipped_frame_mask) else []}",
    )
    _debug_print(
        debug,
        f"num_clipped_frames_per_camera={num_clipped_frames_per_camera.tolist()}",
    )
    _debug_print(
        debug,
        "tensor_shapes="
        f"image_frames={tuple(image_frames.shape)}, "
        f"ego_history_xyz={tuple(ego_history_xyz_tensor.shape)}, "
        f"ego_history_rot={tuple(ego_history_rot_tensor.shape)}, "
        f"ego_future_xyz={tuple(ego_future_xyz_tensor.shape)}, "
        f"ego_future_rot={tuple(ego_future_rot_tensor.shape)}",
    )
    _debug_print(
        debug,
        "note=ego_history_xyz / ego_future_xyz are in raw t0-local xyz frame; "
        "ego_history_rot / ego_future_rot are in model/action rotation frame; "
        "notebook eval plotting may still apply an eval-side xy rotation for comparison",
    )

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps_sod": absolute_timestamps_sod,
        "video_frame_indices": requested_video_frame_indices,
        "actual_video_frame_indices": actual_video_frame_indices,
        "video_num_frames": video_num_frames,
        "frame0_gps_time_sod": float(frame0_gps_time_sod),
        "fps": float(fps),
        "clip_dir": str(clip_dir),

        "time_alignment_summary": time_alignment_summary,
        "requested_image_timestamps_sod": torch.from_numpy(
            np.broadcast_to(
                image_timestamps_sod[None, :],
                (len(camera_indices), num_frames),
            ).astype(np.float64)
        ),
        "requested_video_frame_indices": requested_video_frame_indices,
        "clipped_frame_mask": clipped_frame_mask,
        "num_clipped_frames_per_camera": num_clipped_frames_per_camera,
        "num_clipped_frames_total": num_clipped_frames_total,
        "pose_time_range_sod": (
            float(pose_times_sod[0]),
            float(pose_times_sod[-1]),
        ),
        "history_time_range_sod": (
            float(history_times_sod[0]),
            float(history_times_sod[-1]),
        ),
        "future_time_range_sod": (
            float(future_times_sod[0]),
            float(future_times_sod[-1]),
        ),
        "image_time_range_sod": (
            float(image_timestamps_sod[0]),
            float(image_timestamps_sod[-1]),
        ),

        "coordinate_frame_summary": {
            "position_frame": "t0_local_raw_xyz",
            "rotation_frame": "model_action_rotation_frame",
            "global_frame": "global_enu",
            "note": (
                "xyz tensors are in raw t0-local frame; rotation tensors are in "
                "model/action frame after a fixed planar transform; eval plotting "
                "may apply an additional xy-plane rotation notebook-side"
            ),
        },
    }