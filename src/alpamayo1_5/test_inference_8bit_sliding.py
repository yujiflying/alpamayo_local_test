# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run sliding-window offline inference for a full local clip."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from transformers import BitsAndBytesConfig

from alpamayo1_5 import helper
from alpamayo1_5.load_offline_dataset import (
    DEFAULT_FPS,
    DEFAULT_VIDEO_FRAME0_GPS_TIME_SOD,
    _load_ego_pose_log,
    load_offline_dataset,
)
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

DEFAULT_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/"
    "snapshots/f11cd25b758ab560114019b555dde2a8b92d88b4"
)
DEFAULT_PROCESSOR_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliding-window 8bit Alpamayo inference")
    parser.add_argument("clip_dir", help="Path to offline clip directory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Alpamayo model path")
    parser.add_argument("--processor-path", default=DEFAULT_PROCESSOR_PATH, help="Local processor path")
    parser.add_argument("--device", default="cuda", help="Torch device for inputs")
    parser.add_argument("--stride-s", type=float, default=0.5, help="Sliding window stride in seconds")
    parser.add_argument("--start-sod", type=float, default=None, help="Optional manual start t0_sod")
    parser.add_argument("--end-sod", type=float, default=None, help="Optional manual end t0_sod")
    parser.add_argument("--num-history-steps", type=int, default=16)
    parser.add_argument("--num-future-steps", type=int, default=64)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--frame0-gps-time-sod", type=float, default=DEFAULT_VIDEO_FRAME0_GPS_TIME_SOD)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument(
        "--summary-csv",
        default="sliding_summary.csv",
        help="CSV file for per-window summary results",
    )
    parser.add_argument(
        "--trajectory-csv",
        default="sliding_trajectories.csv",
        help="CSV file for predicted and ground-truth trajectories",
    )
    return parser.parse_args()


def compute_valid_t0_range(
    clip_dir: Path,
    num_history_steps: int,
    num_future_steps: int,
    time_step: float,
) -> tuple[float, float]:
    pose_times_sod, _, _ = _load_ego_pose_log(clip_dir / "ego_pos.log")
    min_t0 = float(pose_times_sod[0] + (num_history_steps - 1) * time_step)
    max_t0 = float(pose_times_sod[-1] - num_future_steps * time_step)
    return min_t0, max_t0


def main() -> None:
    args = parse_args()
    clip_dir = Path(args.clip_dir).expanduser()

    min_t0, max_t0 = compute_valid_t0_range(
        clip_dir=clip_dir,
        num_history_steps=args.num_history_steps,
        num_future_steps=args.num_future_steps,
        time_step=args.time_step,
    )

    start_t0 = args.start_sod if args.start_sod is not None else min_t0
    end_t0 = args.end_sod if args.end_sod is not None else max_t0

    if start_t0 < min_t0:
        raise ValueError(f"start_sod {start_t0} is earlier than valid min_t0 {min_t0}")
    if end_t0 > max_t0:
        raise ValueError(f"end_sod {end_t0} is later than valid max_t0 {max_t0}")
    if start_t0 > end_t0:
        raise ValueError(f"start_sod {start_t0} must be <= end_sod {end_t0}")

    print(f"Sliding inference clip_dir={clip_dir}")
    print(f"Valid t0 range: [{min_t0:.3f}, {max_t0:.3f}]")
    print(f"Using t0 range: [{start_t0:.3f}, {end_t0:.3f}] stride={args.stride_s:.3f}s")

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=False,
    )

    model = Alpamayo1_5.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        quantization_config=quantization_config,
    )
    processor = helper.get_processor(model.tokenizer, processor_name_or_path=args.processor_path)

    summary_csv_path = Path(args.summary_csv).expanduser()
    trajectory_csv_path = Path(args.trajectory_csv).expanduser()

    t0_values = np.arange(start_t0, end_t0 + 1e-9, args.stride_s, dtype=np.float64)

    with (
        summary_csv_path.open("w", encoding="utf-8", newline="") as summary_f,
        trajectory_csv_path.open("w", encoding="utf-8", newline="") as traj_f,
    ):
        summary_writer = csv.writer(summary_f)
        traj_writer = csv.writer(traj_f)

        summary_writer.writerow(
            [
                "t0_sod",
                "frame_index_t0",
                "image_frame_indices",
                "camera_indices",
                "min_ade",
                "load_latency_sec",
                "inference_latency_sec",
                "total_latency_sec",
                "cot",
            ]
        )

        traj_writer.writerow(
            [
                "t0_sod",
                "frame_index_t0",
                "traj_sample_idx",
                "future_step_idx",
                "future_time_sod",
                "pred_x",
                "pred_y",
                "pred_z",
                "gt_x",
                "gt_y",
                "gt_z",
            ]
        )

        for idx, t0_sod in enumerate(t0_values, start=1):
            window_start_time = time.perf_counter()
            print(f"[{idx}/{len(t0_values)}] t0_sod={t0_sod:.3f}")

            load_start = time.perf_counter()
            data = load_offline_dataset(
                clip_dir=clip_dir,
                t0_sod=float(t0_sod),
                num_history_steps=args.num_history_steps,
                num_future_steps=args.num_future_steps,
                time_step=args.time_step,
                num_frames=args.num_frames,
                fps=args.fps,
                frame0_gps_time_sod=args.frame0_gps_time_sod,
                debug=False,
            )
            load_latency_sec = time.perf_counter() - load_start

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
            model_inputs = helper.to_device(model_inputs, args.device)

            if args.device.startswith("cuda"):
                torch.cuda.manual_seed_all(42)
                autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
            else:
                autocast_context = torch.autocast(device_type=args.device, enabled=False)

            inference_start = time.perf_counter()
            with autocast_context:
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    num_traj_samples=args.num_traj_samples,
                    max_generation_length=args.max_generation_length,
                    return_extra=True,
                )
            inference_latency_sec = time.perf_counter() - inference_start

            gt_xyz = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :]  # [T, 3]
            gt_xy = gt_xyz[:, :2].T  # [2, T]

            pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, :, :, :]  # [K, T, 3]
            pred_xy = pred_xyz_np[:, :, :2].transpose(0, 2, 1)  # [K, 2, T]

            diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
            min_ade = float(diff.min())

            total_latency_sec = time.perf_counter() - window_start_time
            frame_index_t0 = int(round((float(t0_sod) - args.frame0_gps_time_sod) * args.fps))

            cot_value = extra["cot"][0]
            if hasattr(cot_value, "tolist"):
                cot_value = cot_value.tolist()

            summary_writer.writerow(
                [
                    float(t0_sod),
                    frame_index_t0,
                    ";".join(map(str, data["video_frame_indices"][0].cpu().tolist())),
                    ";".join(map(str, data["camera_indices"].cpu().tolist())),
                    min_ade,
                    load_latency_sec,
                    inference_latency_sec,
                    total_latency_sec,
                    cot_value,
                ]
            )
            summary_f.flush()

            future_times_sod = (float(t0_sod) + np.arange(1, args.num_future_steps + 1) * args.time_step).tolist()

            num_samples = pred_xyz_np.shape[0]
            num_future_steps = pred_xyz_np.shape[1]
            for sample_idx in range(num_samples):
                for step_idx in range(num_future_steps):
                    pred_point = pred_xyz_np[sample_idx, step_idx]
                    gt_point = gt_xyz[step_idx]
                    traj_writer.writerow(
                        [
                            float(t0_sod),
                            frame_index_t0,
                            sample_idx,
                            step_idx,
                            future_times_sod[step_idx],
                            float(pred_point[0]),
                            float(pred_point[1]),
                            float(pred_point[2]),
                            float(gt_point[0]),
                            float(gt_point[1]),
                            float(gt_point[2]),
                        ]
                    )
            traj_f.flush()

            print(
                f"  frame_index_t0={frame_index_t0}, "
                f"minADE={min_ade:.3f}, "
                f"load={load_latency_sec:.3f}s, "
                f"infer={inference_latency_sec:.3f}s, "
                f"total={total_latency_sec:.3f}s"
            )

    print(f"Saved summary CSV to {summary_csv_path}")
    print(f"Saved trajectory CSV to {trajectory_csv_path}")


if __name__ == "__main__":
    main()