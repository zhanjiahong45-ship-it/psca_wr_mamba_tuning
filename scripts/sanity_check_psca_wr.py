import argparse
import copy
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MAMBA_SRC = REPO_ROOT / "src" / "mamba"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LOCAL_MAMBA_SRC))
os.environ.setdefault("S6_BRIDGE_USE_LOCAL_MAMBA", "1")

from mamba_ssm.models.config_mamba import MambaConfig
from modules.mamba_peft import MambaPeft
from modules.mixer_seq_simple import MambaLMHeadModelPeft
from modules.psca_wr import PSCAWRConfig, inject_psca_wr_adapters


def run_zero_init_check(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32

    config = MambaConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        vocab_size=args.vocab_size,
        ssm_cfg={
            "d_state": args.d_state,
            "d_conv": args.d_conv,
            "expand": args.expand,
            "backend": args.backend,
        },
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        pad_vocab_size_multiple=8,
    )
    base = MambaLMHeadModelPeft(
        config,
        device=device,
        dtype=dtype,
        mamba_cls=MambaPeft,
        use_fast_path=False,
    )
    psca = copy.deepcopy(base)
    psca_config = PSCAWRConfig(
        use_psca_wr=True,
        psca_rank=args.psca_rank,
        psca_alpha=args.psca_alpha,
        psca_dropout=0.0,
        psca_init_zero=True,
        psca_adapt_b=args.psca_adapt_b,
        psca_adapt_c=args.psca_adapt_c,
        psca_use_projector_shift=True,
        psca_projector_residual=args.psca_projector_residual,
        psca_projector_scale=args.psca_projector_scale,
        psca_fallback_lite=args.psca_fallback_lite,
        psca_random_gate=False,
        psca_independent_gate=False,
        psca_debug=args.debug_shapes,
    )
    inject_psca_wr_adapters(psca, psca_config)
    base.eval()
    psca.eval()

    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    with torch.no_grad():
        logits_base = base(input_ids=input_ids).logits
        logits_psca = psca(input_ids=input_ids).logits

    max_abs_diff = (logits_psca - logits_base).detach().float().abs().max().item()
    print(f"max_abs_diff: {max_abs_diff:.10g}")
    if max_abs_diff >= args.tolerance:
        raise SystemExit(f"PSCA-WR zero-init equivalence failed: {max_abs_diff} >= {args.tolerance}")
    return max_abs_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="torch_logcumsumexp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--d_state", type=int, default=4)
    parser.add_argument("--d_conv", type=int, default=3)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--psca_rank", type=int, default=8)
    parser.add_argument("--psca_alpha", type=float, default=1.0)
    parser.add_argument("--psca_adapt_b", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--psca_adapt_c", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--psca_projector_residual", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--psca_projector_scale", type=float, default=1e-3)
    parser.add_argument("--psca_fallback_lite", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--debug_shapes", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    run_zero_init_check(args)


if __name__ == "__main__":
    main()
