from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import torch


def set_reproducible_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def wrap_to_180_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def rotate_xy(xy, angle_deg):
    rad = np.deg2rad(angle_deg)
    c = np.cos(rad)
    s = np.sin(rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return xy @ R.T


def rotate_xyz_xy_plane(xyz, angle_deg):
    xyz = xyz.copy()
    xyz[:, :2] = rotate_xy(xyz[:, :2], angle_deg)
    return xyz


def global_motion_yaw_deg(xyz):
    xy = xyz[:, :2]
    disp = xy[-1] - xy[0]
    if np.linalg.norm(disp) < 1e-6:
        return np.nan
    return float(np.rad2deg(np.arctan2(disp[1], disp[0])))


def mean_speed_mps(xyz, dt):
    xy = xyz[:, :2]
    dxy = xy[1:] - xy[:-1]
    step_dist = np.linalg.norm(dxy, axis=1)
    if len(step_dist) == 0:
        return 0.0
    return float(step_dist.mean() / dt)


def yaw_from_rot_plus_x_deg(rot_mats):
    return np.rad2deg(np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0]))


def history_consistency_table(hist_xyz, hist_rot, dt):
    dxy = hist_xyz[1:, :2] - hist_xyz[:-1, :2]
    step_speed = np.linalg.norm(dxy, axis=1) / dt
    dxy_yaw_deg = np.rad2deg(np.arctan2(dxy[:, 1], dxy[:, 0]))
    rot_yaw_deg = yaw_from_rot_plus_x_deg(hist_rot)[1:]
    yaw_minus_dxy_deg = wrap_to_180_deg(rot_yaw_deg - dxy_yaw_deg)

    return pd.DataFrame({
        "step_idx": np.arange(len(dxy)),
        "dx": dxy[:, 0],
        "dy": dxy[:, 1],
        "step_speed_mps": step_speed,
        "dxy_yaw_deg": dxy_yaw_deg,
        "rot_yaw_deg": rot_yaw_deg,
        "yaw_minus_dxy_deg": yaw_minus_dxy_deg,
        "abs_yaw_minus_dxy_deg": np.abs(yaw_minus_dxy_deg),
    })


def history_consistency_summary(hist_xyz, hist_rot, dt):
    dxy = hist_xyz[1:, :2] - hist_xyz[:-1, :2]
    if len(dxy) == 0:
        return {
            "mean_abs_yaw_minus_dxy_deg": np.nan,
            "last5_mean_abs_yaw_minus_dxy_deg": np.nan,
        }

    dxy_yaw_deg = np.rad2deg(np.arctan2(dxy[:, 1], dxy[:, 0]))
    rot_yaw_deg = yaw_from_rot_plus_x_deg(hist_rot)[1:]
    yaw_minus_dxy_deg = wrap_to_180_deg(rot_yaw_deg - dxy_yaw_deg)
    abs_err = np.abs(yaw_minus_dxy_deg)

    return {
        "mean_abs_yaw_minus_dxy_deg": float(abs_err.mean()),
        "last5_mean_abs_yaw_minus_dxy_deg": (
            float(abs_err[-5:].mean()) if len(abs_err) >= 5 else float(abs_err.mean())
        ),
    }


def segment_mean_speed_table(gt_xyz, model_xyz, dt, segment_sec=1.0):
    seg_len = int(round(segment_sec / dt))
    rows = []
    start = 0
    seg_id = 0

    while start < len(gt_xyz) - 1:
        end = min(start + seg_len + 1, len(gt_xyz))

        def _mean_speed(xyz):
            seg = xyz[start:end]
            if len(seg) < 2:
                return np.nan
            dxy = seg[1:, :2] - seg[:-1, :2]
            return float(np.linalg.norm(dxy, axis=1).mean() / dt)

        rows.append({
            "segment_id": seg_id,
            "t_start_sec": round(start * dt, 2),
            "t_end_sec": round((end - 1) * dt, 2),
            "gt_mean_speed_mps": _mean_speed(gt_xyz),
            "model_mean_speed_mps": _mean_speed(model_xyz),
        })

        start += seg_len
        seg_id += 1

    return pd.DataFrame(rows)


def summarize_main_metrics(gt_xyz, pred_xyz_np, dt, cot_value=None):
    gt_xy = gt_xyz[:, :2].T
    pred_xy = pred_xyz_np[:, :, :2].transpose(0, 2, 1)

    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    best_idx = int(diff.argmin())
    min_ade = float(diff.min())
    pred_best_xyz = pred_xyz_np[best_idx]

    gt_final_xy = gt_xyz[-1, :2]
    pred_final_xy = pred_best_xyz[-1, :2]
    final_point_error = float(np.linalg.norm(pred_final_xy - gt_final_xy))

    gt_mean_speed = mean_speed_mps(gt_xyz, dt)
    pred_mean_speed = mean_speed_mps(pred_best_xyz, dt)
    speed_error = float(pred_mean_speed - gt_mean_speed)

    gt_yaw = global_motion_yaw_deg(gt_xyz)
    pred_yaw = global_motion_yaw_deg(pred_best_xyz)
    if np.isfinite(gt_yaw) and np.isfinite(pred_yaw):
        yaw_error = float(np.rad2deg(wrap_to_pi(np.deg2rad(pred_yaw - gt_yaw))))
    else:
        yaw_error = np.nan

    df_metrics = pd.DataFrame([{
        "best_sample_idx": best_idx,
        "minADE_m": min_ade,
        "final_point_error_m": final_point_error,
        "gt_final_x": float(gt_final_xy[0]),
        "gt_final_y": float(gt_final_xy[1]),
        "pred_final_x": float(pred_final_xy[0]),
        "pred_final_y": float(pred_final_xy[1]),
        "gt_mean_speed_mps": gt_mean_speed,
        "pred_mean_speed_mps": pred_mean_speed,
        "speed_error_mps": speed_error,
        "gt_yaw_deg": gt_yaw,
        "pred_yaw_deg": pred_yaw,
        "yaw_error_deg": yaw_error,
        "cot": "" if cot_value is None else str(cot_value),
    }])

    return df_metrics, best_idx, pred_best_xyz


def _compute_adaptive_xy_limits(*arrays, min_span=20.0, pad_ratio=0.12, pad_min=2.0):
    pts = [np.array([[0.0, 0.0]], dtype=np.float32)]
    for arr in arrays:
        if arr is None:
            continue
        pts.append(arr[:, :2])

    pts = np.concatenate(pts, axis=0)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    xspan = xmax - xmin
    yspan = ymax - ymin
    span = max(float(xspan), float(yspan), float(min_span))
    xcenter = 0.5 * (xmin + xmax)
    ycenter = 0.5 * (ymin + ymax)
    pad = max(span * pad_ratio, pad_min)
    half = 0.5 * span + pad
    return (xcenter - half, xcenter + half), (ycenter - half, ycenter + half)


def build_model_inputs(data: dict[str, Any], tokenized_inputs, device: str):
    from alpamayo1_5 import helper

    model_inputs = {
        "tokenized_data": tokenized_inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, device)


def run_offline_inference_window(
    *,
    data: dict[str, Any],
    model,
    processor,
    device: str,
    top_p: float,
    temperature: float,
    num_traj_samples: int,
    max_generation_length: int,
    time_step: float,
    eval_xy_rotation_deg: float,
    nav_text: str | None = None,
):
    from alpamayo1_5 import helper

    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
        nav_text=nav_text,
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = build_model_inputs(data, inputs, device)

    if device.startswith("cuda"):
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_context = torch.autocast(device_type=device, enabled=False)

    with autocast_context:
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=True,
        )

    hist_xyz_raw = data["ego_history_xyz"].cpu().numpy()[0, 0, :, :]
    hist_rot = data["ego_history_rot"].cpu().numpy()[0, 0, :, :, :]
    gt_xyz_raw = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :]
    pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, :, :, :]

    hist_xyz_plot = rotate_xyz_xy_plane(hist_xyz_raw, eval_xy_rotation_deg)
    gt_xyz_plot = rotate_xyz_xy_plane(gt_xyz_raw, eval_xy_rotation_deg)

    cot_value = extra["cot"][0]
    if hasattr(cot_value, "tolist"):
        cot_value = cot_value.tolist()

    history_diag = history_consistency_summary(hist_xyz_raw, hist_rot, time_step)
    df_metrics, best_idx, pred_best_xyz = summarize_main_metrics(
        gt_xyz=gt_xyz_plot,
        pred_xyz_np=pred_xyz_np,
        dt=time_step,
        cot_value=cot_value,
    )

    return {
        "messages": messages,
        "tokenized_inputs": inputs,
        "pred_xyz": pred_xyz,
        "pred_rot": pred_rot,
        "extra": extra,
        "pred_xyz_np": pred_xyz_np,
        "hist_xyz_raw": hist_xyz_raw,
        "hist_rot": hist_rot,
        "gt_xyz_raw": gt_xyz_raw,
        "hist_xyz_plot": hist_xyz_plot,
        "gt_xyz_plot": gt_xyz_plot,
        "cot_value": cot_value,
        "history_diag": history_diag,
        "df_metrics": df_metrics,
        "best_idx": best_idx,
        "pred_best_xyz": pred_best_xyz,
        "nav_text": nav_text,
    }