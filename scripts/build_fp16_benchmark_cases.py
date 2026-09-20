#!/usr/bin/env python3
"""Maintainer utility: build the committed ten-task FP16 benchmark artifact."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from test import assert_fp16_only, configure_determinism, cpu_copy, make_processors, move_batch


DATASET_INDICES = [0, 112, 417, 812, 924, 1014, 1407, 1827, 2451, 2551]
MODEL_INPUT_KEYS = {
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.state",
    "observation.language.tokens",
    "observation.language.attention_mask",
}


def assert_all_floats_fp16(value, path="artifact") -> None:
    if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float16:
        raise RuntimeError(f"{path} has forbidden dtype {value.dtype}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_all_floats_fp16(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_all_floats_fp16(item, f"{path}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smolvla_base")
    parser.add_argument("--dataset-root", default="libero_goal_no_noops_1.0.0_lerobot")
    parser.add_argument("--repo-id", default="libero_goal_no_noops_1.0.0_lerobot")
    parser.add_argument("--output", type=Path, default=Path("benchmark/smolvla_fp16_10_cases.pt"))
    args = parser.parse_args()

    configure_determinism(0)
    device = torch.device("cuda")
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device=device, dtype=torch.float16).eval()
    preprocessor, postprocessor = make_processors(policy, args.model, device)
    dataset = LeRobotDataset(root=args.dataset_root, repo_id=args.repo_id)
    cases = []

    for dataset_index in DATASET_INDICES:
        frame = dict(dataset[dataset_index])
        batch = move_batch(preprocessor(frame), device)
        batch = {key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS}
        noise = torch.zeros(
            (1, policy.config.chunk_size, policy.config.max_action_dim),
            dtype=torch.float16,
            device=device,
        )
        assert_fp16_only(policy, batch, noise)
        policy.reset()
        with torch.inference_mode():
            normalized_action = policy.predict_action_chunk(batch, noise=noise)
            action = postprocessor(normalized_action)
        case = {
            "dataset_index": dataset_index,
            "task_index": int(frame["task_index"]),
            "episode_index": int(frame["episode_index"]),
            "frame_index": int(frame["frame_index"]),
            "batch": cpu_copy(batch),
            "noise": noise.cpu(),
            "expected_normalized_action": normalized_action.cpu(),
            "expected_action": action.cpu(),
        }
        assert_all_floats_fp16(case)
        cases.append(case)
        print(f"built case {len(cases)}/10 from dataset index {dataset_index}")

    artifact = {
        "format_version": 1,
        "dtype": "float16",
        "noise": "zeros",
        "dataset_repo_id": "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
        "dataset_codebase_version": "v3.0",
        "model_repo_id": "lerobot/smolvla_base",
        "cases": cases,
    }
    assert_all_floats_fp16(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
