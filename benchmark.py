#!/usr/bin/env python3
"""Replay the committed ten-case FP16 SmolVLA benchmark."""

import argparse
import statistics
import time
from pathlib import Path

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from test import assert_fp16_only, configure_determinism, make_processors, move_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smolvla_base")
    parser.add_argument("--cases", type=Path, default=Path("benchmark/smolvla_fp16_10_cases.pt"))
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This FP16 benchmark requires CUDA.")
    configure_determinism(0)
    device = torch.device("cuda")
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device=device, dtype=torch.float16).eval()
    _, postprocessor = make_processors(policy, args.model, device)
    artifact = torch.load(args.cases, map_location="cpu", weights_only=True)

    durations_ms = []
    for case_index, case in enumerate(artifact["cases"]):
        batch = move_batch(case["batch"], device)
        noise = move_batch(case["noise"], device)
        assert_fp16_only(policy, batch, noise)
        policy.reset()

        if case_index == 0:
            for _ in range(args.warmup):
                with torch.inference_mode():
                    policy.predict_action_chunk(batch, noise=noise)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            normalized_action = policy.predict_action_chunk(batch, noise=noise)
        torch.cuda.synchronize()
        durations_ms.append((time.perf_counter() - start) * 1000)

        expected = case["expected_normalized_action"]
        actual = normalized_action.cpu()
        if not torch.equal(actual, expected):
            max_diff = (actual - expected).abs().max().item()
            raise RuntimeError(f"case {case_index} is not bit-exact; max abs diff={max_diff}")
        action = postprocessor(normalized_action)
        if action.dtype != torch.float16:
            raise RuntimeError(f"case {case_index} postprocessed output is {action.dtype}, expected FP16")
        print(f"case={case_index} task={case['task_index']} frame={case['dataset_index']} ms={durations_ms[-1]:.3f}")

    print(
        f"mean_ms={statistics.fmean(durations_ms):.3f} "
        f"min_ms={min(durations_ms):.3f} max_ms={max(durations_ms):.3f}"
    )
    print("All 10 FP16 outputs are bit-exact.")


if __name__ == "__main__":
    main()
