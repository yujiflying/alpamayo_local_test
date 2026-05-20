# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load local offline multi-camera clips for model inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch.nn.functional as F
import av
import numpy as np
import scipy.spatial.transform as spt
import torch
from einops import rearrange

from alpamayo1_5 import helper

DEFAULT_VIDEO_FRAME0_GPS_TIME_SOD = 175484.98
DEFAULT_FPS = 30.0

# WGS84
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _debug_print(enabled: bool, message: str) -> None:
    """Print a debug message when debug mode is enabled."""
    if enabled:
        print(f"[load_offline_dataset] {message}")


def _geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    """Convert WGS84 geodetic coordinates to ECEF."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)

    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + alt_m) * sin_lat

    return np.stack([x, y, z], axis=-1)


def _ecef_to_enu(
    ecef_xyz: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> np.ndarray:
    """Convert ECEF coordinates to local ENU coordinates around a reference geodetic point."""
    ref_ecef = _geodetic_to_ecef(
        np.asarray([ref_lat_deg], dtype=np.float64),
        np.asarray([ref_lon_deg], dtype=np.float64),
        np.asarray([ref_alt_m], dtype=np.float64),
    )[0]

    dx = ecef_xyz - ref_ecef

    lat0 = np.deg2rad(ref_lat_deg)
    lon0 = np.deg2rad(ref_lon_deg)

    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    sin_lon0 = np.sin(lon0)
    cos_lon0 = np.cos(lon0)

    rot = np.array(
        [
            [-sin_lon0, cos_lon0, 0.0],
            [-sin_lat0 * cos_lon0, -sin_lat0 * sin_lon0, cos_lat0],
            [cos_lat0 * cos_lon0, cos_lat0 * sin_lon0, sin_lat0],
        ],
        dtype=np.float64,
    )

    return dx @ rot.T


def _parse_gpfpd_line(line: str) -> tuple[float, float, float, float, np.ndarray] | None:
    """Parse one $GPFPD record from ego_pos.log.

    Returns:
        (gps_time_sod, latitude_deg, longitude_deg, altitude_m, rotation_matrix)
    """
    line = line.strip()
    if not line or not line.startswith("$GPFPD"):
        return None

    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 10:
        return None

    gps_time_sod = float(parts[2])
    heading_deg = float(parts[3])
    pitch_deg = float(parts[4])
    roll_deg = float(parts[5])
    latitude_deg = float(parts[6])
    longitude_deg = float(parts[7])
    altitude_m = float(parts[8])

    rotation = spt.Rotation.from_euler("zyx", [heading_deg, pitch_deg, roll_deg], degrees=True)
    return gps_time_sod, latitude_deg, longitude_deg, altitude_m, rotation.as_matrix()


def _load_ego_pose_log(ego_log_path: str | Path) -> tuple[np.ndarray, np.ndarray, spt.Rotation]:
    """Load ego_pos.log in GPFPD format.

    Returns:
        timestamps_sod: [N]
        positions_enu_m: [N, 3]
        rotations: scipy Rotation object of length N
    """
    ego_log_path = Path(ego_log_path).expanduser()
    timestamps = []
    latitudes = []
    longitudes = []
    altitudes = []
    rotations = []

    with ego_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_gpfpd_line(line)
            if parsed is None:
                continue
            gps_time_sod, lat_deg, lon_deg, alt_m, rotation = parsed
            timestamps.append(gps_time_sod)
            latitudes.append(lat_deg)
            longitudes.append(lon_deg)
            altitudes.append(alt_m)
            rotations.append(rotation)

    if not timestamps:
        raise ValueError(f"No valid $GPFPD records found in {ego_log_path}")

    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    latitudes_np = np.asarray(latitudes, dtype=np.float64)
    longitudes_np = np.asarray(longitudes, dtype=np.float64)
    altitudes_np = np.asarray(altitudes, dtype=np.float64)
    rotations_np = np.asarray(rotations, dtype=np.float64)

    sort_idx = np.argsort(timestamps_np)
    timestamps_np = timestamps_np[sort_idx]
    latitudes_np = latitudes_np[sort_idx]
    longitudes_np = longitudes_np[sort_idx]
    altitudes_np = altitudes_np[sort_idx]
    rotations_np = rotations_np[sort_idx]

    ecef_xyz = _geodetic_to_ecef(latitudes_np, longitudes_np, altitudes_np)
    ref_lat_deg = float(latitudes_np[0])
    ref_lon_deg = float(longitudes_np[0])
    ref_alt_m = float(altitudes_np[0])
    positions_enu_m = _ecef_to_enu(ecef_xyz, ref_lat_deg, ref_lon_deg, ref_alt_m)

    return timestamps_np, positions_enu_m, spt.Rotation.from_matrix(rotations_np)


def _interpolate_positions(
    timestamps_sod: np.ndarray,
    positions: np.ndarray,
    query_times_sod: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate positions at query times."""
    x = np.asarray(query_times_sod, dtype=np.float64)
    t = np.asarray(timestamps_sod, dtype=np.float64)

    if np.any(x < t[0]) or np.any(x > t[-1]):
        raise ValueError(
            f"Query time out of pose range: pose range [{t[0]:.3f}, {t[-1]:.3f}], "
            f"query range [{x.min():.3f}, {x.max():.3f}]"
        )

    interpolated = np.empty((len(x), 3), dtype=np.float64)
    for dim in range(3):
        interpolated[:, dim] = np.interp(x, t, positions[:, dim])
    return interpolated


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


def _sample_video_frames(
    video_path: str | Path,
    frame_indices: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, int]:
    """Load selected video frames and convert them to model input layout."""
    decoded_frames = _load_video_frames(video_path)
    max_index = len(decoded_frames) - 1
    clipped_indices = np.clip(frame_indices.astype(np.int64), 0, max_index)
    sampled = np.stack([decoded_frames[index] for index in clipped_indices], axis=0)
    return rearrange(torch.from_numpy(sampled), "t h w c -> t c h w"), clipped_indices, len(decoded_frames)


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

    The offline clip directory should contain camera mp4 files and ``ego_pos.log``.
    Ego pose timestamps are in GPS seconds-of-day, while video frame timestamps are
    derived from ``frame0_gps_time_sod`` and ``fps``.
    """
    clip_dir = Path(clip_dir).expanduser()
    ego_log_path = clip_dir / "ego_pos.log"
    if not clip_dir.exists():
        raise FileNotFoundError(f"Offline clip directory does not exist: {clip_dir}")
    if not ego_log_path.exists():
        raise FileNotFoundError(f"Missing ego pose log: {ego_log_path}")

    _debug_print(debug, f"clip_dir={clip_dir}")
    _debug_print(debug, f"ego_log_path={ego_log_path}")
    _debug_print(debug, f"t0_sod={t0_sod:.6f}, fps={fps:.3f}, frame0_gps_time_sod={frame0_gps_time_sod:.6f}")
    _debug_print(
        debug,
        "config="
        f"num_history_steps={num_history_steps}, num_future_steps={num_future_steps}, "
        f"time_step={time_step:.3f}, num_frames={num_frames}, image_size={image_size}",
    )

    camera_paths = helper.discover_offline_camera_files(clip_dir)
    _debug_print(debug, f"camera_files={[path.name for path in camera_paths]}")

    pose_times_sod, pose_xyz_enu, pose_rot_global = _load_ego_pose_log(ego_log_path)
    _debug_print(
        debug,
        f"pose_range_sod=[{pose_times_sod[0]:.6f}, {pose_times_sod[-1]:.6f}], "
        f"num_pose_samples={len(pose_times_sod)}",
    )
    _debug_print(
        debug,
        "pose_enu_examples="
        f"first={pose_xyz_enu[0].tolist()}, "
        f"mid={pose_xyz_enu[len(pose_xyz_enu) // 2].tolist()}, "
        f"last={pose_xyz_enu[-1].tolist()}",
    )

    history_offsets = np.arange(-(num_history_steps - 1), 1, dtype=np.float64) * time_step
    future_offsets = np.arange(1, num_future_steps + 1, dtype=np.float64) * time_step
    history_times_sod = t0_sod + history_offsets
    future_times_sod = t0_sod + future_offsets

    all_query_times = np.concatenate([history_times_sod, future_times_sod])
    if all_query_times.min() < pose_times_sod[0] or all_query_times.max() > pose_times_sod[-1]:
        raise ValueError(
            "Requested history/future window exceeds ego pose log range. "
            f"Pose range: [{pose_times_sod[0]:.3f}, {pose_times_sod[-1]:.3f}], "
            f"requested: [{all_query_times.min():.3f}, {all_query_times.max():.3f}]"
        )

    _debug_print(
        debug,
        f"history_range_sod=[{history_times_sod[0]:.6f}, {history_times_sod[-1]:.6f}]",
    )
    _debug_print(
        debug,
        f"future_range_sod=[{future_times_sod[0]:.6f}, {future_times_sod[-1]:.6f}]",
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

    _debug_print(debug, f"t0_xyz_enu={t0_xyz.tolist()}")
    _debug_print(debug, f"history_last_local_xyz={ego_history_xyz_local[-1].tolist()}")
    _debug_print(debug, f"future_first_local_xyz={ego_future_xyz_local[0].tolist()}")
    _debug_print(debug, f"future_last_local_xyz={ego_future_xyz_local[-1].tolist()}")

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

    image_times_sod = t0_sod + np.arange(-(num_frames - 1), 1, dtype=np.float64) * time_step
    image_frame_indices = np.rint((image_times_sod - frame0_gps_time_sod) * fps).astype(np.int64)

    _debug_print(debug, f"image_times_sod={image_times_sod.tolist()}")
    _debug_print(debug, f"requested_frame_indices={image_frame_indices.tolist()}")

    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []
    actual_video_frame_indices_list = []
    video_num_frames_list = []
    for camera_path in camera_paths:
        frames_tensor, clipped_indices, num_decoded_frames = _sample_video_frames(
            camera_path, image_frame_indices
        )
        cam_idx = helper.infer_camera_index(camera_path.name)

        frames_tensor = _resize_frames(frames_tensor, image_size)
        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(image_times_sod.astype(np.float64)))
        actual_video_frame_indices_list.append(torch.from_numpy(clipped_indices.astype(np.int64)))
        video_num_frames_list.append(num_decoded_frames)

        _debug_print(
            debug,
            f"camera={camera_path.name}, cam_idx={cam_idx}, "
            f"num_video_frames={num_decoded_frames}, actual_frame_indices={clipped_indices.tolist()}",
        )

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps_sod = torch.stack(timestamps_list, dim=0)
    actual_video_frame_indices = torch.stack(actual_video_frame_indices_list, dim=0)
    video_num_frames = torch.tensor(video_num_frames_list, dtype=torch.int64)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps_sod = all_timestamps_sod[sort_order]
    actual_video_frame_indices = actual_video_frame_indices[sort_order]
    video_num_frames = video_num_frames[sort_order]

    relative_timestamps = (all_timestamps_sod - t0_sod).float()

    _debug_print(debug, f"sorted_camera_indices={camera_indices.tolist()}")
    _debug_print(debug, f"relative_timestamps={relative_timestamps[0].tolist() if len(relative_timestamps) else []}")
    _debug_print(
        debug,
        f"actual_video_frame_indices_first_camera={actual_video_frame_indices[0].tolist() if len(actual_video_frame_indices) else []}",
    )
    _debug_print(
        debug,
        "tensor_shapes="
        f"image_frames={tuple(image_frames.shape)}, "
        f"ego_history_xyz={tuple(ego_history_xyz_tensor.shape)}, "
        f"ego_future_xyz={tuple(ego_future_xyz_tensor.shape)}",
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
        "video_frame_indices": torch.from_numpy(image_frame_indices.astype(np.int64)).unsqueeze(0).repeat(
            len(camera_indices), 1
        ),
        "actual_video_frame_indices": actual_video_frame_indices,
        "video_num_frames": video_num_frames,
        "t0_sod": float(t0_sod),
        "frame0_gps_time_sod": float(frame0_gps_time_sod),
        "fps": float(fps),
        "clip_dir": str(clip_dir),
    }
