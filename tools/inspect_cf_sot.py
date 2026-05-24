#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inspect CF-SOT parameters after training.

Checkpoint-only mode can inspect Fourier coefficients and context alpha.
Actual context_raw and full CF-SOT scale depend on batch gate trajectories, so
those fields are reported as NaN unless --context_scalar_abs_mean is supplied.

Usage:
python tools/inspect_cf_sot.py \
  --checkpoint mrpc=outputs/cf_sot/mamba-130m/glue_mrpc/checkpoint-1834 \
  --checkpoint rte=outputs/cf_sot/mamba-130m/glue_rte/checkpoint-620 \
  --checkpoint cola=outputs/cf_sot/mamba-130m/glue_cola/checkpoint-8552
"""

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def resolve_checkpoint_path(path: str) -> str:
    if os.path.isfile(path):
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    preferred_names = [
        "peft.pt",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.bin",
        "checkpoint.bin",
        "adapter_model.safetensors",
        "model.safetensors",
        "pytorch_model.safetensors",
        "checkpoint.safetensors",
        "checkpoint.pt",
        "model.pt",
        "state_dict.pt",
        "checkpoint.pth",
        "model.pth",
    ]
    for name in preferred_names:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate

    candidates = []
    allowed_exts = (".bin", ".pt", ".pth", ".ckpt", ".safetensors")
    excluded_names = {
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "scaler.pt",
        "training_args.bin",
    }
    for root, _, files in os.walk(path):
        for filename in files:
            if filename in excluded_names:
                continue
            if filename.endswith(allowed_exts):
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under directory: {path}")

    candidates.sort()
    return candidates[0]


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    checkpoint_file = resolve_checkpoint_path(path)

    if checkpoint_file.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "This checkpoint is a .safetensors file. Install safetensors or "
                "point --checkpoint to a .bin/.pt file instead."
            ) from exc
        obj = load_file(checkpoint_file, device="cpu")
    else:
        obj = torch.load(checkpoint_file, map_location="cpu")

    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "module"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

    state = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        name = key
        for prefix in ["module.", "model."]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        state[name] = value.detach().cpu()

    return state


def infer_task_from_path(path: str) -> str:
    normalized = path.replace("\\", "/").lower()
    for task in ["mrpc", "rte", "cola", "sst2", "qnli", "qqp", "mnli"]:
        if f"glue_{task}" in normalized or f"/{task}/" in normalized:
            return task.upper()
    return "unknown"


def parse_checkpoint_arg(value: str) -> Tuple[str, str]:
    if "=" in value:
        task, path = value.split("=", 1)
        return task.upper(), path
    return infer_task_from_path(value), value


def is_cf_sot_coeff(name: str) -> bool:
    lname = name.lower()
    return "cf_sot_tf_coeff" in lname or "suffixtuning_cf_sot_tf_coeff" in lname


def is_cf_sot_context_alpha(name: str) -> bool:
    lname = name.lower()
    return "cf_sot_context_alpha" in lname or "suffixtuning_cf_sot_context_alpha" in lname


def build_fourier_basis(
    max_seq_len: int,
    num_freqs: int,
    normalize: bool = True,
    dtype=torch.float32,
) -> torch.Tensor:
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be >= 1.")
    if num_freqs < 1:
        raise ValueError("num_freqs must be >= 1.")

    num_basis = 2 * num_freqs
    omega_min = math.pi / float(max_seq_len)
    omega_max = math.pi

    if num_freqs == 1:
        omegas = torch.tensor([omega_min], dtype=dtype)
    else:
        omegas = torch.tensor(
            [
                omega_min * (omega_max / omega_min) ** (k / (num_freqs - 1))
                for k in range(num_freqs)
            ],
            dtype=dtype,
        )

    t = torch.arange(max_seq_len, dtype=dtype)
    phases = t[:, None] * omegas[None, :]
    phi = torch.stack((torch.sin(phases), torch.cos(phases)), dim=-1).reshape(max_seq_len, num_basis)

    if normalize:
        phi = phi / math.sqrt(float(num_basis))

    return phi


def flatten_coeff_tensor(tensor: torch.Tensor, expected_m: int) -> torch.Tensor:
    x = tensor.float()
    if x.numel() == expected_m:
        return x.reshape(1, expected_m)
    if x.shape[-1] == expected_m:
        return x.reshape(-1, expected_m)
    if x.numel() % expected_m == 0:
        return x.reshape(-1, expected_m)
    raise ValueError(f"Cannot reshape coeff tensor with shape {tuple(x.shape)} to [N, {expected_m}]")


def flatten_alpha_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().reshape(-1)


def inspect_coeffs(
    state: Dict[str, torch.Tensor],
    phi: torch.Tensor,
    eps: float,
    expected_m: int,
) -> List[Dict[str, Any]]:
    rows = []
    for name, tensor in state.items():
        if not is_cf_sot_coeff(name):
            continue
        coeffs = flatten_coeff_tensor(tensor, expected_m)
        for row_idx, coeff in enumerate(coeffs):
            freq_raw = phi @ coeff
            scale = 1.0 + eps * torch.tanh(freq_raw)
            dev = scale - 1.0
            rows.append(
                {
                    "param_name": name,
                    "row": row_idx,
                    "coeff_shape": tuple(tensor.shape),
                    "tf_coeff_l2": coeff.norm(p=2).item(),
                    "tf_coeff_abs_mean": coeff.abs().mean().item(),
                    "freq_raw_abs_mean": freq_raw.abs().mean().item(),
                    "freq_raw_abs_max": freq_raw.abs().max().item(),
                    "scale_abs_dev_mean": dev.abs().mean().item(),
                    "scale_abs_dev_max": dev.abs().max().item(),
                    "scale_std": scale.std(unbiased=False).item(),
                    "scale_min": scale.min().item(),
                    "scale_max": scale.max().item(),
                }
            )
    return rows


def inspect_alphas(state: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    rows = []
    for name, tensor in state.items():
        if not is_cf_sot_context_alpha(name):
            continue
        values = flatten_alpha_tensor(tensor)
        for row_idx, value in enumerate(values):
            rows.append(
                {
                    "param_name": name,
                    "row": row_idx,
                    "context_alpha": value.item(),
                    "context_alpha_abs": value.abs().item(),
                }
            )
    return rows


def summarize_task(
    task: str,
    checkpoint: str,
    max_seq_len: int,
    num_freqs: int,
    eps: float,
    normalize_basis: bool,
    context_scalar_abs_mean: float | None,
) -> Dict[str, Any]:
    state = load_state_dict(checkpoint)
    expected_m = 2 * num_freqs
    phi = build_fourier_basis(max_seq_len, num_freqs, normalize=normalize_basis)
    coeff_rows = inspect_coeffs(state, phi=phi, eps=eps, expected_m=expected_m)
    alpha_rows = inspect_alphas(state)

    def avg(rows, key):
        if not rows:
            return float("nan")
        return sum(row[key] for row in rows) / len(rows)

    def mx(rows, key):
        if not rows:
            return float("nan")
        return max(row[key] for row in rows)

    def mn(rows, key):
        if not rows:
            return float("nan")
        return min(row[key] for row in rows)

    context_alpha = avg(alpha_rows, "context_alpha")
    if context_scalar_abs_mean is None or not alpha_rows:
        context_raw_abs_mean = float("nan")
    else:
        context_raw_abs_mean = avg(alpha_rows, "context_alpha_abs") * float(context_scalar_abs_mean)

    return {
        "task": task,
        "checkpoint": checkpoint,
        "num_tensors": len(state),
        "num_cf_sot_coeff_rows": len(coeff_rows),
        "num_cf_sot_context_alpha_rows": len(alpha_rows),
        "avg_tf_coeff_l2": avg(coeff_rows, "tf_coeff_l2"),
        "avg_mean_abs_a": avg(coeff_rows, "tf_coeff_abs_mean"),
        "context_alpha": context_alpha,
        "avg_freq_raw_abs_mean": avg(coeff_rows, "freq_raw_abs_mean"),
        "avg_context_raw_abs_mean": context_raw_abs_mean,
        "avg_mean_abs_scale_minus_1": avg(coeff_rows, "scale_abs_dev_mean"),
        "max_max_abs_scale_minus_1": mx(coeff_rows, "scale_abs_dev_max"),
        "avg_std_scale": avg(coeff_rows, "scale_std"),
        "global_min_scale": mn(coeff_rows, "scale_min"),
        "global_max_scale": mx(coeff_rows, "scale_max"),
        "coeff_rows": coeff_rows,
        "alpha_rows": alpha_rows,
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def print_summary(summary: Dict[str, Any], max_rows: int):
    print("\n========== CF-SOT Usage Summary ==========")
    print(f"Task: {summary['task']}")
    print(f"Checkpoint: {summary['checkpoint']}")
    print(f"num coefficient rows/layers: {summary['num_cf_sot_coeff_rows']}")
    print(f"num context alpha rows/layers: {summary['num_cf_sot_context_alpha_rows']}")
    print(f"avg tf_coeff_l2:        {fmt(summary['avg_tf_coeff_l2'])}")
    print(f"avg mean|a|:            {fmt(summary['avg_mean_abs_a'])}")
    print(f"context_alpha value:    {fmt(summary['context_alpha'])}")
    print(f"avg freq_raw_abs_mean:  {fmt(summary['avg_freq_raw_abs_mean'])}")
    print(f"avg context_raw_abs_mean: {fmt(summary['avg_context_raw_abs_mean'])}")
    print(f"avg mean|scale-1|:      {fmt(summary['avg_mean_abs_scale_minus_1'])}")
    print(f"max max|scale-1|:       {fmt(summary['max_max_abs_scale_minus_1'])}")
    print(f"avg std(scale):         {fmt(summary['avg_std_scale'])}")
    print(f"global min(scale):      {fmt(summary['global_min_scale'])}")
    print(f"global max(scale):      {fmt(summary['global_max_scale'])}")

    if math.isnan(summary["avg_context_raw_abs_mean"]):
        print(
            "NOTE: context_raw_abs_mean and true CF-SOT scale require gate trajectories. "
            "Checkpoint-only mode reports frequency-only scale statistics."
        )

    rows = summary["coeff_rows"][:max_rows]
    if rows:
        print("\nidx | param | row | coeff_l2 | mean|a| | freq_raw_abs_mean | mean|scale-1|")
        print("-" * 88)
        for idx, row in enumerate(rows):
            print(
                f"{idx:03d} | {row['param_name']} | {row['row']} | "
                f"{row['tf_coeff_l2']:.6g} | {row['tf_coeff_abs_mean']:.6g} | "
                f"{row['freq_raw_abs_mean']:.6g} | {row['scale_abs_dev_mean']:.6g}"
            )
        if len(summary["coeff_rows"]) > max_rows:
            print(f"... truncated: showing {max_rows}/{len(summary['coeff_rows'])} coefficient rows")


def print_task_table(summaries: List[Dict[str, Any]]):
    if len(summaries) <= 1:
        return
    print("\n| Task | tf_coeff norm | context_alpha | freq_raw abs mean | context_raw abs mean | mean(|scale-1|) |")
    print("|---|---:|---:|---:|---:|---:|")
    for summary in summaries:
        print(
            f"| {summary['task']} | "
            f"{fmt(summary['avg_tf_coeff_l2'])} | "
            f"{fmt(summary['context_alpha'])} | "
            f"{fmt(summary['avg_freq_raw_abs_mean'])} | "
            f"{fmt(summary['avg_context_raw_abs_mean'])} | "
            f"{fmt(summary['avg_mean_abs_scale_minus_1'])} |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path, optionally prefixed as task=path. Can be repeated.",
    )
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--num_freqs", type=int, default=4)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--no_normalize_basis", action="store_true")
    parser.add_argument("--context_scalar_abs_mean", type=float, default=None)
    parser.add_argument("--max_rows", type=int, default=200)
    args = parser.parse_args()

    summaries = []
    for item in args.checkpoint:
        task, path = parse_checkpoint_arg(item)
        summary = summarize_task(
            task=task,
            checkpoint=path,
            max_seq_len=args.max_seq_len,
            num_freqs=args.num_freqs,
            eps=args.eps,
            normalize_basis=not args.no_normalize_basis,
            context_scalar_abs_mean=args.context_scalar_abs_mean,
        )
        summaries.append(summary)
        print_summary(summary, max_rows=args.max_rows)

    print_task_table(summaries)


if __name__ == "__main__":
    main()
