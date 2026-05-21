# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load an offline local clip into the Alpamayo inference format."""

from __future__ import annotations

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
    """Convert geodetic coordinates to a local ENU metric frame."""
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

    return np.stack([east_m, north_m, up_m], axis=-1)


def _load_ego_pose_log(
    ego_log_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, spt.Rotation]:
    """Load ego poses from a mixed NMEA-like ego_pos.log."""
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

    xyz_enu = _geodetic_to_local_enu(lat_deg=lat_deg, lon_deg=lon_deg, alt_m=alt_m)

    # IMPORTANT:
    # heading is already ENU yaw:
    #   east = 0 deg, counterclockwise positive
    yaw_deg = heading_deg

    rotations = spt.Rotation.from_euler(
        "xyz",
        np.stack([roll_deg, pitch_deg, yaw_deg], axis=-1),
        degrees=True,
    )

    return timestamps_sod, xyz_enu, rotations


def _interpolate_positions(
    timestamps_sod: np.ndarray,
    positions: np.ndarray,
    query_times_sod: np.ndarray,
) -> np.ndarray:
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
    interp_positions = _interpolate_positions(timestamps_sod, positions, query_times_sod)
    interp_rotations = _interpolate_rotations(timestamps_sod, rotations, query_times_sod)
    return interp_positions, interp_rotations


def _load_video_frames(video_path: str | Path) -> list[np.ndarray]:
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


def _sample_video_frames_from_decoded(
    decoded_frames: list[np.ndarray] | np.ndarray,
    frame_indices: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, int]:
    """Sample video frames from pre-decoded frames."""
    num_decoded_frames = len(decoded_frames)
    max_index = num_decoded_frames - 1
    clipped_indices = np.clip(frame_indices.astype(np.int64), 0, max_index)
    sampled = np.stack([decoded_frames[index] for index in clipped_indices], axis=0)
    return rearrange(torch.from_numpy(sampled), "t h w c -> t c h w"), clipped_indices, num_decoded_frames


def _resize_frames(
    frames: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
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


def preload_offline_clip_to_cache(
    clip_dir: str | Path,
    fps: float = DEFAULT_FPS,
    frame0_gps_time_sod: float = DEFAULT_VIDEO_FRAME0_GPS_TIME_SOD,
    debug: bool = False,
) -> dict[str, Any]:
    """Preload an offline clip into memory for repeated sliding-window access.

    This function decodes all camera videos once and parses ego poses once.
    It is intended for repeated calls to `load_offline_dataset_from_cache(...)`.
    """
    clip_dir = Path(clip_dir).expanduser()
    ego_log_path = clip_dir / "ego_pos.log"
    if not clip_dir.exists():
        raise FileNotFoundError(f"Offline clip directory does not exist: {clip_dir}")
    if not ego_log_path.exists():
        raise FileNotFoundError(f"Missing ego pose log: {ego_log_path}")

    camera_paths = helper.discover_offline_camera_files(clip_dir)
    _debug_print(debug, f"[preload] clip_dir={clip_dir}")
    _debug_print(debug, f"[preload] camera_files={[path.name for path in camera_paths]}")

    pose_times_sod, pose_xyz_enu, pose_rot_global = _load_ego_pose_log(ego_log_path)
    _debug_print(
        debug,
        f"[preload] pose_range_sod=[{pose_times_sod[0]:.6f}, {pose_times_sod[-1]:.6f}], "
        f"num_pose_samples={len(pose_times_sod)}",
    )

    decoded_frames_by_camera_index: dict[int, list[np.ndarray]] = {}
    camera_paths_by_index: dict[int, str] = {}
    video_num_frames_by_index: dict[int, int] = {}

    for camera_path in camera_paths:
        cam_idx = helper.infer_camera_index(camera_path.name)
        decoded_frames = _load_video_frames(camera_path)
        decoded_frames_by_camera_index[cam_idx] = decoded_frames
        camera_paths_by_index[cam_idx] = str(camera_path)
        video_num_frames_by_index[cam_idx] = len(decoded_frames)
        _debug_print(
            debug,
            f"[preload] decoded camera={camera_path.name}, cam_idx={cam_idx}, num_frames={len(decoded_frames)}",
        )

    sorted_camera_indices = sorted(decoded_frames_by_camera_index.keys())

    cache = {
        "clip_dir": str(clip_dir),
        "fps": float(fps),
        "frame0_gps_time_sod": float(frame0_gps_time_sod),
        "camera_indices_sorted": sorted_camera_indices,
        "camera_paths_by_index": camera_paths_by_index,
        "decoded_frames_by_camera_index": decoded_frames_by_camera_index,
        "video_num_frames_by_index": video_num_frames_by_index,
        "pose_times_sod": pose_times_sod,
        "pose_xyz_enu": pose_xyz_enu,
        "pose_rot_global": pose_rot_global,
    }

    _debug_print(debug, f"[preload] cache ready with cameras={sorted_camera_indices}")
    return cache


def load_offline_dataset_from_cache(
    clip_cache: dict[str, Any],
    t0_sod: float,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    num_frames: int = 4,
    image_size: tuple[int, int] = (448, 800),
    debug: bool = False,
) -> dict[str, Any]:
    """Load one sliding window from a preloaded clip cache."""
    fps = float(clip_cache["fps"])
    frame0_gps_time_sod = float(clip_cache["frame0_gps_time_sod"])
    pose_times_sod = clip_cache["pose_times_sod"]
    pose_xyz_enu = clip_cache["pose_xyz_enu"]
    pose_rot_global = clip_cache["pose_rot_global"]

    _debug_print(debug, f"[cache-load] t0_sod={t0_sod:.6f}")

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

    ego_history_xyz, ego_history_rot = _lookup_pose_samples(
        pose_times_sod,
        pose_xyz_enu,
        pose_rot_global,
        history_times_sod,
    )
    ego_future_xyz, ego_future_rot = _lookup_pose_samples(
        pose_times_sod,
        pose_xyz_enu,
        pose_rot_global,
        future_times_sod,
    )

    t0_xyz = ego_history_xyz[-1].copy()
    t0_rot = spt.Rotation.from_matrix(ego_history_rot[-1])
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_matrix(ego_history_rot)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_matrix(ego_future_rot)).as_matrix()

    # IMPORTANT:
    # Keep xyz untouched; apply fixed transform only to rotations.
    R_fix = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ego_history_rot_local = np.einsum("ij,tjk->tik", R_fix, ego_history_rot_local)
    ego_future_rot_local = np.einsum("ij,tjk->tik", R_fix, ego_future_rot_local)

    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_local.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_local.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )
    ego_future_xyz_tensor = (
        torch.from_numpy(ego_future_xyz_local.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )
    ego_future_rot_tensor = (
        torch.from_numpy(ego_future_rot_local.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )

    image_offsets = np.arange(
        -(num_frames - 1) * time_step,
        time_step / 2,
        time_step,
        dtype=np.float64,
    )
    image_timestamps_sod = t0_sod + image_offsets
    image_frame_indices = np.rint((image_timestamps_sod - frame0_gps_time_sod) * fps).astype(np.int64)
    relative_timestamps = image_timestamps_sod - t0_sod

    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []
    actual_video_frame_indices_list = []
    video_num_frames_list = []

    for cam_idx in clip_cache["camera_indices_sorted"]:
        decoded_frames = clip_cache["decoded_frames_by_camera_index"][cam_idx]
        frames_tensor, clipped_indices, num_decoded_frames = _sample_video_frames_from_decoded(
            decoded_frames, image_frame_indices
        )
        frames_tensor = _resize_frames(frames_tensor, image_size)

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(image_timestamps_sod.astype(np.float64)))
        actual_video_frame_indices_list.append(torch.from_numpy(clipped_indices.astype(np.int64)))
        video_num_frames_list.append(num_decoded_frames)

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps_sod = torch.stack(timestamps_list, dim=0)
    actual_video_frame_indices = torch.stack(actual_video_frame_indices_list, dim=0)
    video_num_frames = torch.tensor(video_num_frames_list, dtype=torch.int64)

    relative_timestamps = torch.from_numpy(
        np.broadcast_to(relative_timestamps[None, :], (len(camera_indices), num_frames)).astype(np.float32)
    )
    video_frame_indices = torch.from_numpy(
        np.broadcast_to(image_frame_indices[None, :], (len(camera_indices), num_frames)).astype(np.int64)
    )

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps_sod": all_timestamps_sod,
        "video_frame_indices": video_frame_indices,
        "actual_video_frame_indices": actual_video_frame_indices,
        "video_num_frames": video_num_frames,
        "frame0_gps_time_sod": float(frame0_gps_time_sod),
        "fps": float(fps),
        "clip_dir": clip_cache["clip_dir"],
    }


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
    """Backward-compatible non-cached API.

    This method preloads the clip on each call, so for repeated sliding-window
    use you should prefer:
        preload_offline_clip_to_cache(...)
        load_offline_dataset_from_cache(...)
    """
    clip_cache = preload_offline_clip_to_cache(
        clip_dir=clip_dir,
        fps=fps,
        frame0_gps_time_sod=frame0_gps_time_sod,
        debug=debug,
    )
    return load_offline_dataset_from_cache(
        clip_cache=clip_cache,
        t0_sod=t0_sod,
        num_history_steps=num_history_steps,
        num_future_steps=num_future_steps,
        time_step=time_step,
        num_frames=num_frames,
        image_size=image_size,
        debug=debug,
    )