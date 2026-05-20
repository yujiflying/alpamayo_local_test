# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sliding offline inference over a whole local clip with a live-updating matplotlib window."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from alpamayo1_5 import helper
from alpamayo1_5.load_offline_dataset import _load_ego_pose_log, load_offline_dataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


# ============================================================
# Config
# ============================================================
CLIP_DIR = Path("/workspace/dataset/")
MODEL_PATH = Path(
    "/root/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/"
    "snapshots/f11cd25b758ab560114019b555dde2a8b92d88b4"
)
PROCESSOR_PATH = Path(
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
COSMOS_PROCESSOR_PATH = Path(
    "/root/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-8B/"
    "snapshots/f715d875a8a99a0a2b65aa28633afd9417e46bd9"
)

DEVICE = "cuda"

NUM_HISTORY_STEPS = 16
NUM_FUTURE_STEPS = 64
TIME_STEP = 0.1
NUM_FRAMES = 4
FPS = 30.0
FRAME0_GPS_TIME_SOD = 175484.98

NUM_TRAJ_SAMPLES = 1
TEMPERATURE = 0.6
TOP_P = 0.98
MAX_GENERATION_LENGTH = 256

START_T0_SOD = 175490.23
END_T0_SOD = 175500.23
SLIDE_STEP_SOD = 1.0

PAUSE_SEC = 0.3
SHOW_HISTORY = True
SAVE_SUMMARY_CSV = True
OUTPUT_DIR = REPO_ROOT / "outputs" / "offline_sliding_live"
SUMMARY_CSV = OUTPUT_DIR / "sliding_live_summary.csv"


# ============================================================
# Helpers
# ============================================================
def wrap_to_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2 * np.pi) - np.pi


def global_motion_yaw_deg(xyz: np.ndarray) -> float:
    xy = xyz[:, :2]
    disp = xy[-1] - xy[0]
    if np.linalg.norm(disp) < 1e-6:
        return float("nan")
    return float(np.rad2deg(np.arctan2(disp[1], disp[0])))


def mean_speed_mps(xyz: np.ndarray, dt: float) -> float:
    xy = xyz[:, :2]
    dxy = xy[1:] - xy[:-1]
    step_dist = np.linalg.norm(dxy, axis=1)
    if len(step_dist) == 0:
        return 0.0
    return float(step_dist.mean() / dt)


def compute_adaptive_xy_limits(
    hist_xyz: np.ndarray | None,
    gt_xyz: np.ndarray,
    pred_xyz_np: np.ndarray,
    min_span: float = 20.0,
    pad_ratio: float = 0.12,
    pad_min: float = 2.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = [gt_xyz[:, :2], np.array([[0.0, 0.0]], dtype=np.float32)]

    if hist_xyz is not None:
        pts.append(hist_xyz[:, :2])

    for sample_idx in range(pred_xyz_np.shape[0]):
        pts.append(pred_xyz_np[sample_idx, :, :2])

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

    xlim = (xcenter - half, xcenter + half)
    ylim = (ycenter - half, ycenter + half)
    return xlim, ylim


def update_live_plot(
    fig,
    ax,
    t0_sod: float,
    hist_xyz: np.ndarray | None,
    gt_xyz: np.ndarray,
    pred_xyz_np: np.ndarray,
    min_ade: float,
    final_point_error: float,
    speed_error: float,
    yaw_error: float,
    cot_text: str,
) -> None:
    ax.clear()

    xlim, ylim = compute_adaptive_xy_limits(
        hist_xyz=hist_xyz,
        gt_xyz=gt_xyz,
        pred_xyz_np=pred_xyz_np,
    )

    if hist_xyz is not None:
        ax.plot(
            hist_xyz[:, 0],
            hist_xyz[:, 1],
            color="tab:blue",
            marker="o",
            linewidth=2,
            markersize=3,
            alpha=0.9,
            label="History",
        )

    ax.plot(
        gt_xyz[:, 0],
        gt_xyz[:, 1],
        "k-o",
        linewidth=2,
        markersize=3,
        label="GT",
    )

    for sample_idx in range(pred_xyz_np.shape[0]):
        ax.plot(
            pred_xyz_np[sample_idx, :, 0],
            pred_xyz_np[sample_idx, :, 1],
            "-o",
            linewidth=2,
            markersize=3,
            alpha=0.8,
            label=f"Pred {sample_idx}",
        )

    ax.scatter([0.0], [0.0], c="red", marker="x", s=100, label="t0 ego")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    title = (
        f"Sliding offline inference @ t0_sod={t0_sod:.2f}\n"
        f"minADE={min_ade:.3f}m | final_err={final_point_error:.3f}m | "
        f"speed_err={speed_error:.3f}m/s | yaw_err={yaw_error:.3f}deg"
    )
    ax.set_title(title)

    cot_text = cot_text if len(cot_text) < 220 else cot_text[:217] + "..."
    fig.suptitle(f"CoT: {cot_text}", fontsize=10, y=0.98)

    ax.legend(loc="best")
    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.pause(PAUSE_SEC)


# ============================================================
# Main
# ============================================================
def main() -> None:
    os.environ["ALPAMAYO_VLM_PROCESSOR_PATH"] = str(COSMOS_PROCESSOR_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("CLIP_DIR =", CLIP_DIR)
    print("MODEL_PATH =", MODEL_PATH)
    print("PROCESSOR_PATH =", PROCESSOR_PATH)
    print("ALPAMAYO_VLM_PROCESSOR_PATH =", os.environ["ALPAMAYO_VLM_PROCESSOR_PATH"])
    print("START_T0_SOD =", START_T0_SOD)
    print("END_T0_SOD =", END_T0_SOD)
    print("SLIDE_STEP_SOD =", SLIDE_STEP_SOD)
    print("PAUSE_SEC =", PAUSE_SEC)

    print("\nLoading model and processor ...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=False,
    )

    model = Alpamayo1_5.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        quantization_config=quantization_config,
    )

    processor = helper.get_processor(
        model.tokenizer,
        processor_name_or_path=PROCESSOR_PATH,
    )
    print("Model and processor loaded.")

    pose_times_sod, _, _ = _load_ego_pose_log(CLIP_DIR / "ego_pos.log")

    history_margin = (NUM_HISTORY_STEPS - 1) * TIME_STEP
    future_margin = NUM_FUTURE_STEPS * TIME_STEP
    valid_t0_min = float(pose_times_sod[0] + history_margin)
    valid_t0_max = float(pose_times_sod[-1] - future_margin)

    requested_t0_values = np.arange(START_T0_SOD, END_T0_SOD + 1e-9, SLIDE_STEP_SOD)
    t0_values = requested_t0_values[
        (requested_t0_values >= valid_t0_min) & (requested_t0_values <= valid_t0_max)
    ]

    print("\nPose time range:")
    print("pose_start =", float(pose_times_sod[0]))
    print("pose_end   =", float(pose_times_sod[-1]))
    print("valid_t0_min =", valid_t0_min)
    print("valid_t0_max =", valid_t0_max)
    print("num_requested_t0 =", len(requested_t0_values))
    print("num_valid_t0 =", len(t0_values))

    if len(t0_values) == 0:
        raise ValueError("No valid t0 values remain after filtering.")

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    rows = []

    try:
        for idx, t0_sod in enumerate(t0_values):
            print("\n" + "=" * 80)
            print(f"[{idx + 1}/{len(t0_values)}] Running t0_sod={t0_sod:.3f}")
            print("=" * 80)

            t_start = time.time()

            try:
                print("[A] Loading offline dataset ...")
                data = load_offline_dataset(
                    clip_dir=CLIP_DIR,
                    t0_sod=float(t0_sod),
                    num_history_steps=NUM_HISTORY_STEPS,
                    num_future_steps=NUM_FUTURE_STEPS,
                    time_step=TIME_STEP,
                    num_frames=NUM_FRAMES,
                    fps=FPS,
                    frame0_gps_time_sod=FRAME0_GPS_TIME_SOD,
                    debug=False,
                )

                print("[B] Building chat template ...")
                messages = helper.create_message(
                    frames=data["image_frames"].flatten(0, 1),
                    camera_indices=data["camera_indices"],
                )

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                model_inputs = {
                    "tokenized_data": inputs,
                    "ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"],
                }
                model_inputs = helper.to_device(model_inputs, DEVICE)

                print("[C] Running inference ...")
                if DEVICE.startswith("cuda"):
                    torch.cuda.manual_seed_all(42)
                    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
                else:
                    autocast_context = torch.autocast(device_type=DEVICE, enabled=False)

                with autocast_context:
                    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=TOP_P,
                        temperature=TEMPERATURE,
                        num_traj_samples=NUM_TRAJ_SAMPLES,
                        max_generation_length=MAX_GENERATION_LENGTH,
                        return_extra=True,
                    )

                if DEVICE.startswith("cuda"):
                    torch.cuda.synchronize()

                print("[D] Computing metrics ...")
                cot_value = extra["cot"][0]
                if hasattr(cot_value, "tolist"):
                    cot_value = cot_value.tolist()
                cot_text = str(cot_value)

                hist_xyz = data["ego_history_xyz"].cpu().numpy()[0, 0, :, :]
                gt_xyz = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :]
                gt_xy = gt_xyz[:, :2].T

                pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, :, :, :]
                pred_xy = pred_xyz_np[:, :, :2].transpose(0, 2, 1)

                diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
                min_ade = float(diff.min())
                best_idx = int(diff.argmin())

                pred_best_xyz = pred_xyz_np[best_idx]
                gt_final_xy = gt_xyz[-1, :2]
                pred_final_xy = pred_best_xyz[-1, :2]
                final_point_error = float(np.linalg.norm(pred_final_xy - gt_final_xy))

                gt_mean_speed = mean_speed_mps(gt_xyz, TIME_STEP)
                pred_mean_speed = mean_speed_mps(pred_best_xyz, TIME_STEP)
                speed_error = float(pred_mean_speed - gt_mean_speed)

                gt_yaw = global_motion_yaw_deg(gt_xyz)
                pred_yaw = global_motion_yaw_deg(pred_best_xyz)
                if np.isfinite(gt_yaw) and np.isfinite(pred_yaw):
                    yaw_error = float(np.rad2deg(wrap_to_pi(np.deg2rad(pred_yaw - gt_yaw))))
                else:
                    yaw_error = float("nan")

                elapsed = time.time() - t_start

                print("Chain-of-Causation:")
                print(cot_value)
                print(
                    f"minADE={min_ade:.3f} m | "
                    f"final_err={final_point_error:.3f} m | "
                    f"speed_err={speed_error:.3f} m/s | "
                    f"yaw_err={yaw_error:.3f} deg | "
                    f"time={elapsed:.2f}s"
                )

                update_live_plot(
                    fig=fig,
                    ax=ax,
                    t0_sod=float(t0_sod),
                    hist_xyz=hist_xyz if SHOW_HISTORY else None,
                    gt_xyz=gt_xyz,
                    pred_xyz_np=pred_xyz_np,
                    min_ade=min_ade,
                    final_point_error=final_point_error,
                    speed_error=speed_error,
                    yaw_error=yaw_error,
                    cot_text=cot_text,
                )

                rows.append(
                    {
                        "t0_sod": float(t0_sod),
                        "minADE_m": min_ade,
                        "best_sample_idx": best_idx,
                        "gt_final_x": float(gt_final_xy[0]),
                        "gt_final_y": float(gt_final_xy[1]),
                        "pred_final_x": float(pred_final_xy[0]),
                        "pred_final_y": float(pred_final_xy[1]),
                        "final_point_error_m": final_point_error,
                        "gt_mean_speed_mps": gt_mean_speed,
                        "pred_mean_speed_mps": pred_mean_speed,
                        "speed_error_mps": speed_error,
                        "gt_global_motion_yaw_deg": gt_yaw,
                        "pred_global_motion_yaw_deg": pred_yaw,
                        "yaw_error_deg": yaw_error,
                        "elapsed_sec": elapsed,
                        "cot": cot_text,
                    }
                )

            except Exception as e:
                elapsed = time.time() - t_start
                print(f"ERROR at t0_sod={t0_sod:.3f}: {repr(e)}")

                ax.clear()
                ax.text(
                    0.5,
                    0.5,
                    f"ERROR at t0_sod={t0_sod:.3f}\n{repr(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_title("Sliding offline inference error")
                fig.tight_layout()
                fig.canvas.draw_idle()
                plt.pause(PAUSE_SEC)

                rows.append(
                    {
                        "t0_sod": float(t0_sod),
                        "minADE_m": np.nan,
                        "best_sample_idx": np.nan,
                        "gt_final_x": np.nan,
                        "gt_final_y": np.nan,
                        "pred_final_x": np.nan,
                        "pred_final_y": np.nan,
                        "final_point_error_m": np.nan,
                        "gt_mean_speed_mps": np.nan,
                        "pred_mean_speed_mps": np.nan,
                        "speed_error_mps": np.nan,
                        "gt_global_motion_yaw_deg": np.nan,
                        "pred_global_motion_yaw_deg": np.nan,
                        "yaw_error_deg": np.nan,
                        "elapsed_sec": elapsed,
                        "cot": f"ERROR: {repr(e)}",
                    }
                )

    finally:
        plt.ioff()

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("Sliding live summary table")
    print("=" * 80)
    print(df)

    print("\nSummary stats:")
    summary_cols = [
        "minADE_m",
        "final_point_error_m",
        "speed_error_mps",
        "yaw_error_deg",
        "elapsed_sec",
    ]
    print(df[summary_cols].describe())

    if SAVE_SUMMARY_CSV:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(SUMMARY_CSV, index=False)
        print(f"\nSaved summary CSV to: {SUMMARY_CSV}")

    plt.show()


if __name__ == "__main__":
    main()