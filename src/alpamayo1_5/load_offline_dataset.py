# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load an offline local clip into the Alpamayo inference format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import numpy as np
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


def _load_ego_pose_log(
    ego_log_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, spt.Rotation]:
    """Load ego poses from ego_pos.log.

    Expected format per line:
        sod x y z r00 r01 r02 r10 r11 r12 r20 r21 r22

    where:
        - sod is GPS seconds-of-day
        - x,y,z are ENU positions in meters
        - r** entries form a 3x3 rotation matrix
    """
    ego_log_path = Path(ego_log_path).expanduser()
    if not ego_log_path.exists():
        raise FileNotFoundError(f"Missing ego pose log: {ego_log_path}")

    rows = []
    with ego_log_path.open("r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 13:
                raise ValueError(
                    f"Expected 13 columns in {ego_log_path} at line {line_idx}, got {len(parts)}"
                )
            rows.append([float(x) for x in parts])

    if not rows:
        raise ValueError(f"No valid rows found in ego pose log: {ego_log_path}")

    arr = np.asarray(rows, dtype=np.float64)
    timestamps_sod = arr[:, 0]
    xyz_enu = arr[:, 1:4]
    rot_mats = arr[:, 4:].reshape(-1, 3, 3)

    if not np.all(np.diff(timestamps_sod) >= 0):
        raise ValueError("Ego pose timestamps are not sorted in non-decreasing order.")

    rotations = spt.Rotation.from_matrix(rot_mats)
    return timestamps_sod, xyz_enu, rotations


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

    interp = np.stack([np.interp(x, t, positions[:, dim]) for dim in range(positions.shape[1])], axis=-1)
    return interp


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
        f"last={pose_xyz_enu[-1].tolist()}",
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

    # Step 1: construct local frame from ego pose at t0
    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_matrix(ego_history_rot)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_matrix(ego_future_rot)).as_matrix()

    # Step 2: align rotation convention with the model/action-space convention.
    #
    # IMPORTANT:
    # - We do NOT apply the previous fixed planar rotation to xyz.
    # - Instead, we apply the fixed transform to rotations so that the resulting
    #   local trajectory representation is consistent with the model's +x-forward
    #   yaw convention used downstream by the action-space code.
    #
    # The fixed planar transform is:
    #   x' = -y
    #   y' =  x
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

    for camera_path in camera_paths:
        frames_tensor, clipped_indices, num_decoded_frames = _sample_video_frames(
            camera_path, image_frame_indices
        )
        cam_idx = helper.infer_camera_index(camera_path.name)

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

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps_sod = all_timestamps_sod[sort_order]
    actual_video_frame_indices = actual_video_frame_indices[sort_order]
    video_num_frames = video_num_frames[sort_order]

    relative_timestamps = torch.from_numpy(
        np.broadcast_to(relative_timestamps[None, :], (len(camera_indices), num_frames)).astype(np.float32)
    )
    video_frame_indices = torch.from_numpy(
        np.broadcast_to(image_frame_indices[None, :], (len(camera_indices), num_frames)).astype(np.int64)
    )

    _debug_print(
        debug,
        f"relative_timestamps={relative_timestamps[0].tolist() if len(relative_timestamps) else []}",
    )
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
        "video_frame_indices": video_frame_indices,
        "actual_video_frame_indices": actual_video_frame_indices,
        "video_num_frames": video_num_frames,
        "frame0_gps_time_sod": float(frame0_gps_time_sod),
        "fps": float(fps),
        "clip_dir": str(clip_dir),
    }