# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline end-to-end inference example for local Alpamayo samples."""

from __future__ import annotations

import argparse
import numpy as np
import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_offline_dataset import load_offline_dataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

DEFAULT_MODEL_PATH = "/home/yuji/Alpamayo/model_weight"
DEFAULT_PROCESSOR_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alpamayo offline inference on a local clip")
    parser.add_argument("clip_dir", help="Path to offline clip directory, e.g. clip_xxx")
    parser.add_argument("--t0-sod", type=float, required=True, help="Manual t0 in GPS seconds-of-day")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Alpamayo model path")
    parser.add_argument(
        "--processor-path",
        default=DEFAULT_PROCESSOR_PATH,
        help="Local Qwen processor path or HF identifier",
    )
    parser.add_argument("--device", default="cuda", help="Torch device to use")
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-generation-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    """Run inference on an offline clip and report minADE."""
    args = parse_args()

    print(f"Loading offline clip from: {args.clip_dir}")
    data = load_offline_dataset(
        clip_dir=args.clip_dir,
        t0_sod=args.t0_sod,
        debug=True,
    )
    print("Offline dataset loaded.")

    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )

    model = Alpamayo1_5.from_pretrained(args.model_path, dtype=torch.bfloat16).to(args.device)
    processor = helper.get_processor(model.tokenizer, processor_name_or_path=args.processor_path)

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

    with autocast_context:
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )

    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print("minADE:", min_ade, "meters")
    if min_ade >= 1.0:
        print(f"WARNING: minADE ({min_ade:.2f}m) is above 1.0m. Model sampling can be stochastic.")


if __name__ == "__main__":
    main()
