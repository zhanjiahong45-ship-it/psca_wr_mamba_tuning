#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inspect TF-SOT usage after training.

This script checks:
1. Whether tf_sot coefficients are non-zero.
2. Whether scale s_{l,t} deviates from 1.
3. Per-layer coeff norm and scale statistics.

Usage:
python tools/inspect_tf_sot.py \
  --checkpoint /path/to/checkpoint.pt \
  --max_seq_len 256 \
  --num_freqs 4 \
  --eps 0.1
"""

import argparse
import math
import os
import sys
from typing import Dict, Any, List

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
        raise FileNotFoundError(
            f"No checkpoint file found under directory: {path}. "
            f"Expected one of: {', '.join(preferred_names)}"
        )

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

    # common checkpoint formats
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "module"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

    # remove common prefixes
    new_obj = {}
    for k, v in obj.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        for prefix in ["module.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_obj[nk] = v.detach().cpu()

    return new_obj


def is_tf_sot_coeff(name: str) -> bool:
    """
    Adjust this if your actual parameter name differs.
    This tries to catch common names.
    """
    patterns = [
        "tf_sot",
        "tfsot",
        "tf_coeff",
        "tf_sot_coeff",
        "suffixtuning_tf",
        "temporal_fourier",
        "fourier_coeff",
    ]
    lname = name.lower()
    return any(p in lname for p in patterns)


def build_fourier_basis(
    max_seq_len: int,
    num_freqs: int,
    normalize: bool = True,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Return phi: [L, M], where M = 2 * num_freqs.
    phi_{2k}(t)   = sin(omega_k * t)
    phi_{2k+1}(t) = cos(omega_k * t)
    """
    L = max_seq_len
    K = num_freqs
    M = 2 * K

    omega_min = math.pi / L
    omega_max = math.pi

    if K == 1:
        omegas = torch.tensor([omega_min], dtype=dtype)
    else:
        omegas = torch.tensor(
            [
                omega_min * (omega_max / omega_min) ** (k / (K - 1))
                for k in range(K)
            ],
            dtype=dtype,
        )

    t = torch.arange(L, dtype=dtype)

    basis = []
    for w in omegas:
        basis.append(torch.sin(w * t))
        basis.append(torch.cos(w * t))

    phi = torch.stack(basis, dim=-1)  # [L, M]

    if normalize:
        phi = phi / math.sqrt(M)

    return phi


def flatten_coeff_tensor(tensor: torch.Tensor, expected_m: int) -> torch.Tensor:
    """
    Convert coefficient tensor into [N, M].
    Handles:
    - [M]
    - [num_layers, M]
    - [num_adapters?, num_layers, M]
    - any tensor whose last dim is M
    """
    x = tensor.float()

    if x.numel() == expected_m:
        return x.reshape(1, expected_m)

    if x.shape[-1] == expected_m:
        return x.reshape(-1, expected_m)

    # If shape is odd but total divisible by M, still try.
    if x.numel() % expected_m == 0:
        return x.reshape(-1, expected_m)

    raise ValueError(
        f"Cannot reshape coeff tensor with shape {tuple(x.shape)} to [N, {expected_m}]"
    )


def inspect_coeff(
    name: str,
    coeff_tensor: torch.Tensor,
    phi: torch.Tensor,
    eps: float,
    expected_m: int,
) -> List[Dict[str, Any]]:
    coeffs = flatten_coeff_tensor(coeff_tensor, expected_m=expected_m)
    rows = []

    for idx, coeff in enumerate(coeffs):
        raw = phi @ coeff  # [L]
        scale = 1.0 + eps * torch.tanh(raw)
        dev = scale - 1.0

        row = {
            "param_name": name,
            "layer_or_row": idx,
            "coeff_shape": tuple(coeff_tensor.shape),
            "coeff_l2": coeff.norm(p=2).item(),
            "coeff_abs_mean": coeff.abs().mean().item(),
            "coeff_abs_max": coeff.abs().max().item(),
            "raw_abs_mean": raw.abs().mean().item(),
            "raw_abs_max": raw.abs().max().item(),
            "scale_abs_dev_mean": dev.abs().mean().item(),
            "scale_abs_dev_max": dev.abs().max().item(),
            "scale_std": scale.std().item(),
            "scale_min": scale.min().item(),
            "scale_max": scale.max().item(),
        }
        rows.append(row)

    return rows


def print_table(rows: List[Dict[str, Any]], max_rows: int = 200):
    if not rows:
        print("No TF-SOT coefficient parameters found.")
        return

    header = (
        "idx | param | row | coeff_l2 | mean|a| | max|a| | "
        "mean|s-1| | max|s-1| | std(s) | min(s) | max(s)"
    )
    print("\n" + header)
    print("-" * len(header))

    for i, r in enumerate(rows[:max_rows]):
        print(
            f"{i:03d} | "
            f"{r['param_name']} | "
            f"{r['layer_or_row']} | "
            f"{r['coeff_l2']:.6g} | "
            f"{r['coeff_abs_mean']:.6g} | "
            f"{r['coeff_abs_max']:.6g} | "
            f"{r['scale_abs_dev_mean']:.6g} | "
            f"{r['scale_abs_dev_max']:.6g} | "
            f"{r['scale_std']:.6g} | "
            f"{r['scale_min']:.6g} | "
            f"{r['scale_max']:.6g}"
        )

    if len(rows) > max_rows:
        print(f"... truncated: showing {max_rows}/{len(rows)} rows")


def print_summary(rows: List[Dict[str, Any]]):
    if not rows:
        return

    def avg(key):
        return sum(r[key] for r in rows) / len(rows)

    def mx(key):
        return max(r[key] for r in rows)

    print("\n========== TF-SOT Usage Summary ==========")
    print(f"num coefficient rows/layers: {len(rows)}")
    print(f"avg coeff_l2:              {avg('coeff_l2'):.6g}")
    print(f"avg mean|a|:               {avg('coeff_abs_mean'):.6g}")
    print(f"max max|a|:                {mx('coeff_abs_max'):.6g}")
    print(f"avg mean|s-1|:             {avg('scale_abs_dev_mean'):.6g}")
    print(f"max max|s-1|:              {mx('scale_abs_dev_max'):.6g}")
    print(f"avg std(s):                {avg('scale_std'):.6g}")
    print(f"global min(s):             {min(r['scale_min'] for r in rows):.6g}")
    print(f"global max(s):             {max(r['scale_max'] for r in rows):.6g}")

    print("\n========== Rough Interpretation ==========")
    mean_dev = avg("scale_abs_dev_mean")
    avg_l2 = avg("coeff_l2")

    if mean_dev < 1e-4 or avg_l2 < 1e-4:
        print("TF-SOT likely did NOT learn: scale is almost 1 or coeff is almost 0.")
    elif mean_dev < 2e-3:
        print("TF-SOT learned weakly: modulation exists but is small.")
    else:
        print("TF-SOT is being used: temporal scale clearly deviates from 1.")

    print("\nSuggested thresholds:")
    print("- mean|s-1| < 1e-4: almost unused")
    print("- mean|s-1| around 1e-4 ~ 1e-3: weak")
    print("- mean|s-1| > 0.002: meaningful modulation")
    print("- max|s-1| > 0.03: some positions have strong modulation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--num_freqs", type=int, default=4)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--no_normalize_basis", action="store_true")
    parser.add_argument("--max_rows", type=int, default=200)
    args = parser.parse_args()

    state = load_state_dict(args.checkpoint)
    expected_m = 2 * args.num_freqs

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Total tensor params in state dict: {len(state)}")
    print(
        f"max_seq_len={args.max_seq_len}, num_freqs={args.num_freqs}, "
        f"M={expected_m}, eps={args.eps}"
    )

    phi = build_fourier_basis(
        max_seq_len=args.max_seq_len,
        num_freqs=args.num_freqs,
        normalize=not args.no_normalize_basis,
    )

    rows = []
    print("\nSearching TF-SOT coefficient parameters...")
    for name, tensor in state.items():
        if is_tf_sot_coeff(name):
            print(f"FOUND: {name}, shape={tuple(tensor.shape)}")
            try:
                rows.extend(
                    inspect_coeff(
                        name=name,
                        coeff_tensor=tensor,
                        phi=phi,
                        eps=args.eps,
                        expected_m=expected_m,
                    )
                )
            except Exception as e:
                print(f"WARNING: failed to inspect {name}: {e}")

    print_table(rows, max_rows=args.max_rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
