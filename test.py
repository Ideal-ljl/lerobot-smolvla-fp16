#!/usr/bin/env python3
"""Deterministic, FP16-only SmolVLA inference and input replay."""

import os

# Required by deterministic CUDA matrix multiplication.  This must be set before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


DEFAULT_MODEL = "/mnt/aigo-inspect/lerobot/smolvla_base"
DEFAULT_DATASET = "/mnt/aigo-inspect/lerobot/libero_goal_no_noops_1.0.0_lerobot"
DEFAULT_REPO_ID = "libero_goal_no_noops_1.0.0_lerobot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--input-file", type=Path, default=Path("smolvla_replay_input.pt"))
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Load the already-preprocessed batch and noise from --input-file.",
    )
    parser.add_argument(
        "--seeded-noise",
        action="store_true",
        help="Use seeded Gaussian noise instead of the default all-zero deterministic latent.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def move_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        dtype = torch.float16 if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=dtype)
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch(item, device) for item in value)
    return value


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def assert_fp16_only(policy: torch.nn.Module, batch: dict[str, Any], noise: torch.Tensor) -> None:
    bad_model = [
        (name, tensor.dtype)
        for name, tensor in list(policy.named_parameters()) + list(policy.named_buffers())
        if tensor.is_floating_point() and tensor.dtype != torch.float16
    ]
    bad_inputs = [
        (name, tensor.dtype)
        for name, tensor in batch.items()
        if isinstance(tensor, torch.Tensor)
        and tensor.is_floating_point()
        and tensor.dtype != torch.float16
    ]
    if bad_model or bad_inputs or noise.dtype != torch.float16:
        raise RuntimeError(
            f"FP16-only check failed: model={bad_model[:10]}, inputs={bad_inputs}, noise={noise.dtype}"
        )


def make_processors(policy: SmolVLAPolicy, model_path: str, device: torch.device):
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        preprocessor_overrides={
            "device_processor": {"device": str(device), "float_dtype": "float16"},
            "normalizer_processor": {"device": str(device), "dtype": torch.float16},
            "rename_observations_processor": {
                "rename_map": {
                    "observation.images.image": "observation.images.camera1",
                    "observation.images.wrist_image": "observation.images.camera2",
                }
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {"device": "cpu", "dtype": torch.float16},
            "device_processor": {"device": "cpu", "float_dtype": "float16"},
        },
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FP16-only SmolVLA inference requires CUDA; CPU fallback is intentionally disabled.")

    configure_determinism(args.seed)
    device = torch.device("cuda")
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device=device, dtype=torch.float16).eval()
    preprocessor, postprocessor = make_processors(policy, args.model, device)

    if args.replay:
        saved = torch.load(args.input_file, map_location="cpu", weights_only=True)
        batch = move_batch(saved["batch"], device)
        noise = move_batch(saved["noise"], device)
    else:
        dataset = LeRobotDataset(root=args.dataset_root, repo_id=args.repo_id)
        batch = move_batch(preprocessor(dict(dataset[args.frame_index])), device)
        shape = (batch["observation.state"].shape[0], policy.config.chunk_size, policy.config.max_action_dim)
        if args.seeded_noise:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            noise = torch.randn(shape, generator=generator, dtype=torch.float16).to(device)
        else:
            # No sampling at all: identical latent input on every run and every machine.
            noise = torch.zeros(shape, dtype=torch.float16, device=device)
        torch.save({"batch": cpu_copy(batch), "noise": cpu_copy(noise)}, args.input_file)
        print(f"Saved replay input to {args.input_file}")

    assert_fp16_only(policy, batch, noise)
    policy.reset()
    with torch.inference_mode():
        normalized_action = policy.predict_action_chunk(batch, noise=noise)
        pred_action = postprocessor(normalized_action)

    if args.replay and "expected_normalized_action" in saved:
        expected = saved["expected_normalized_action"]
        actual = normalized_action.cpu()
        if not torch.equal(actual, expected):
            max_diff = (actual - expected).abs().max().item()
            raise RuntimeError(f"Replay mismatch: output is not bit-exact (max abs diff={max_diff}).")
        print("Replay verification: bit-exact output match")
    elif not args.replay:
        torch.save(
            {
                "batch": cpu_copy(batch),
                "noise": cpu_copy(noise),
                "expected_normalized_action": normalized_action.cpu(),
            },
            args.input_file,
        )

    print("normalized action:", normalized_action.cpu())
    print("pred action:", pred_action)


if __name__ == "__main__":
    main()
