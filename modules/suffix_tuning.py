from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum
import math
from typing import Dict, List, Optional, Union
from peft.config import PeftConfig
import torch
import torch.nn.functional as F
from torch import nn
from einops import einsum, repeat, rearrange

from peft.tuners.tuners_utils import BaseTunerLayer

from modules.mamba_peft_utils import MambaPeftType, register_peft_config, register_peft_tuner
from modules.mamba_tuner_utils import MambaBaseTuner


class SuffixTuningBiasType(str, enum.Enum):
    SILU_Z = "SILU_Z"
    SILU_Z_C = "SILU_Z_C"
    SILU_Z_C_IA3 = "SILU_Z_C_IA3"
    SILU_Z_C_SCALE = "SILU_Z_C_SCALE"
    # DSOT: dynamic state offset modulated by causal context of silu(z).
    DYNAMIC_SILU_Z_C = "DYNAMIC_SILU_Z_C"
    # K-SOT: kernelized feature-space readout of the original C_t.
    KERNEL_SILU_Z_C = "KERNEL_SILU_Z_C"
    # Key-aware DSOT: blend vanilla causal context with key-token weighted context.
    KEY_AWARE_DYNAMIC_SILU_Z_C = "KEY_AWARE_DYNAMIC_SILU_Z_C"


class SuffixTuningBiasInit(str, enum.Enum):
    ZERO = "ZERO"
    RANDOM = "RANDOM"


class LoraParam(nn.Module):
    def __init__(self, d1, d2, r, dropout, ratio, device, dtype):
        super().__init__()

        if dropout is None:
            dropout = 0

        if ratio is not None:
            assert d1 % ratio == 0
            d1 = d1 // ratio
            d2 = d2 * ratio

        self.ratio = ratio
        self.lora_A = nn.Parameter(torch.zeros(r, d2, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d1, r, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self):
        out = self.lora_B @ self.lora_A

        if hasattr(self, "ratio") and self.ratio is not None:
            d1, d2 = out.shape
            out = out.reshape(d1 * self.ratio, d2 // self.ratio)

        return out


@register_peft_config(MambaPeftType.SUFFIX_TUNING)
@dataclass
class SuffixTuningConfig(PeftConfig):
    method: Optional[str] = field(default=None)
    bias_init: str = field(default=None)
    bias_type: str = field(default=None)
    finetune_parameters: List[str] = field(default=None)
    C_scale_shape: str = field(default=None)
    C_bias_shape: str = field(default=None)
    r: int = field(default=None, metadata={"help": "Lora attention dimension"})
    r_ratio: int = field(default=None)
    dropout: float = field(default=None)
    ksot_phi_dim: Optional[int] = field(default=None)
    ksot_lambda_init: float = field(default=1e-3)
    ksot_activation: str = field(default="silu")
    num_experts: int = field(default=1)
    consistency_lambda: float = field(default=0.0)
    use_consistency: bool = field(default=False)
    inference_merge: str = field(default="mean")
    use_sft: bool = field(default=False)
    sft_type: str = field(default="channel")
    sft_delta_scale: float = field(default=0.01)
    sft_clamp: Optional[float] = field(default=0.2)
    sft_init: str = field(default="zero")
    sft_apply_position: str = field(default="after_softplus")
    use_tf_sot: bool = field(default=False)
    tf_sot_num_freqs: int = field(default=4)
    tf_sot_num_basis: int = field(default=8)
    tf_sot_eps: float = field(default=0.1)
    tf_sot_max_seq_len: int = field(default=256)
    tf_sot_freq_grid: str = field(default="geometric")
    tf_sot_normalize_basis: bool = field(default=True)
    tf_sot_init_coeff: str = field(default="zero")
    use_cf_sot: bool = field(default=False)
    cf_sot_num_freqs: int = field(default=4)
    cf_sot_num_basis: int = field(default=8)
    cf_sot_eps: float = field(default=0.1)
    cf_sot_max_seq_len: int = field(default=256)
    cf_sot_freq_grid: str = field(default="geometric")
    cf_sot_normalize_basis: bool = field(default=True)
    cf_sot_context_center: bool = field(default=True)
    cf_sot_context_reduce: str = field(default="mean")
    cf_sot_init_tf_coeff: str = field(default="zero")
    cf_sot_init_context_alpha: str = field(default="zero")

    def __post_init__(self):
        self.peft_type = MambaPeftType.SUFFIX_TUNING
        self.method = None if self.method is None else str(self.method).lower()
        self.bias_init = SuffixTuningBiasInit(self.bias_init)
        self.bias_type = SuffixTuningBiasType(self.bias_type)
        self.num_experts = int(self.num_experts)
        self.consistency_lambda = float(self.consistency_lambda)
        self.use_consistency = bool(self.use_consistency)
        self.inference_merge = str(self.inference_merge).lower()
        self.use_sft = bool(self.use_sft)
        self.sft_type = str(self.sft_type).lower()
        self.sft_delta_scale = float(self.sft_delta_scale)
        self.sft_clamp = None if self.sft_clamp is None else float(self.sft_clamp)
        self.sft_init = str(self.sft_init).lower()
        self.sft_apply_position = str(self.sft_apply_position).lower()
        self.use_tf_sot = bool(self.use_tf_sot)
        self.tf_sot_num_freqs = int(self.tf_sot_num_freqs)
        self.tf_sot_num_basis = int(self.tf_sot_num_basis)
        self.tf_sot_eps = float(self.tf_sot_eps)
        self.tf_sot_max_seq_len = int(self.tf_sot_max_seq_len)
        self.tf_sot_freq_grid = str(self.tf_sot_freq_grid).lower()
        self.tf_sot_normalize_basis = bool(self.tf_sot_normalize_basis)
        self.tf_sot_init_coeff = str(self.tf_sot_init_coeff).lower()
        self.use_cf_sot = bool(self.use_cf_sot)
        self.cf_sot_num_freqs = int(self.cf_sot_num_freqs)
        self.cf_sot_num_basis = int(self.cf_sot_num_basis)
        self.cf_sot_eps = float(self.cf_sot_eps)
        self.cf_sot_max_seq_len = int(self.cf_sot_max_seq_len)
        self.cf_sot_freq_grid = str(self.cf_sot_freq_grid).lower()
        self.cf_sot_normalize_basis = bool(self.cf_sot_normalize_basis)
        self.cf_sot_context_center = bool(self.cf_sot_context_center)
        self.cf_sot_context_reduce = str(self.cf_sot_context_reduce).lower()
        self.cf_sot_init_tf_coeff = str(self.cf_sot_init_tf_coeff).lower()
        self.cf_sot_init_context_alpha = str(self.cf_sot_init_context_alpha).lower()

        if self.use_tf_sot and self.use_cf_sot:
            raise ValueError("use_tf_sot and use_cf_sot cannot both be enabled.")

        if self.use_tf_sot:
            if self.bias_type != SuffixTuningBiasType.SILU_Z_C:
                raise ValueError("TF-SOT currently supports the original SOT bias_type SILU_Z_C only.")
            if self.method is not None:
                raise ValueError("TF-SOT should not be mixed with other SOT methods.")
            if self.use_sft:
                raise ValueError("TF-SOT should not be mixed with SOT+SFT in the same adapter.")
            if self.tf_sot_num_freqs < 1:
                raise ValueError("TF-SOT requires tf_sot_num_freqs >= 1.")
            if self.tf_sot_num_basis != 2 * self.tf_sot_num_freqs:
                raise ValueError("TF-SOT requires tf_sot_num_basis == 2 * tf_sot_num_freqs.")
            if self.tf_sot_eps < 0:
                raise ValueError("TF-SOT requires tf_sot_eps >= 0.")
            if self.tf_sot_max_seq_len < 1:
                raise ValueError("TF-SOT requires tf_sot_max_seq_len >= 1.")
            if self.tf_sot_freq_grid != "geometric":
                raise ValueError("TF-SOT currently supports tf_sot_freq_grid='geometric' only.")
            if self.tf_sot_init_coeff != "zero":
                raise ValueError("TF-SOT currently supports tf_sot_init_coeff='zero' only.")

        if self.use_cf_sot:
            if self.bias_type != SuffixTuningBiasType.SILU_Z_C:
                raise ValueError("CF-SOT currently supports the original SOT bias_type SILU_Z_C only.")
            if self.method is not None:
                raise ValueError("CF-SOT should not be mixed with other SOT methods.")
            if self.use_sft:
                raise ValueError("CF-SOT should not be mixed with SOT+SFT in the same adapter.")
            if self.cf_sot_num_freqs < 1:
                raise ValueError("CF-SOT requires cf_sot_num_freqs >= 1.")
            if self.cf_sot_num_basis != 2 * self.cf_sot_num_freqs:
                raise ValueError("CF-SOT requires cf_sot_num_basis == 2 * cf_sot_num_freqs.")
            if self.cf_sot_eps < 0:
                raise ValueError("CF-SOT requires cf_sot_eps >= 0.")
            if self.cf_sot_max_seq_len < 1:
                raise ValueError("CF-SOT requires cf_sot_max_seq_len >= 1.")
            if self.cf_sot_freq_grid != "geometric":
                raise ValueError("CF-SOT currently supports cf_sot_freq_grid='geometric' only.")
            if self.cf_sot_context_reduce != "mean":
                raise ValueError("CF-SOT currently supports cf_sot_context_reduce='mean' only.")
            if self.cf_sot_init_tf_coeff != "zero":
                raise ValueError("CF-SOT currently supports cf_sot_init_tf_coeff='zero' only.")
            if self.cf_sot_init_context_alpha != "zero":
                raise ValueError("CF-SOT currently supports cf_sot_init_context_alpha='zero' only.")

        if self.method == "adamix_sot":
            if self.bias_type != SuffixTuningBiasType.SILU_Z_C:
                raise ValueError("AdaMix-SOT currently supports the original SOT bias_type SILU_Z_C only.")
            if self.num_experts < 1:
                raise ValueError("AdaMix-SOT requires num_experts >= 1.")
            if self.inference_merge != "mean":
                raise ValueError("AdaMix-SOT currently supports inference_merge='mean' only.")

        if self.use_sft:
            if self.sft_type not in ("channel", "scalar"):
                raise ValueError(f"Unsupported sft_type: {self.sft_type}")
            if self.sft_init != "zero":
                raise ValueError("SFT currently supports sft_init='zero' only.")
            if self.sft_apply_position != "after_softplus":
                raise ValueError("SFT currently supports sft_apply_position='after_softplus' only.")


@register_peft_tuner(MambaPeftType.SUFFIX_TUNING)
class SuffixTuningModel(MambaBaseTuner):
    prefix: str = "suffixtuning_"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        peft_config.target_modules = ["x_after_ssm"]

        if peft_config.C_scale_shape is not None or peft_config.C_bias_shape is not None:
            peft_config.target_modules += ["C"]

        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        finetune_parameters = self.peft_config[self.active_adapter].finetune_parameters

        if finetune_parameters is None:
            finetune_parameters = []

        for n, p in model.named_parameters():
            if (
                self.prefix in n
                or "sft_delta_bias" in n
                or "cf_sot_tf_coeff" in n
                or "cf_sot_context_alpha" in n
                or any(n.endswith("." + fp) for fp in finetune_parameters)
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _create_new_module(self, peft_config, adapter_name, target, target_name):
        new_module = None

        if target_name == "x_after_ssm":
            new_module = SuffixTuningBiasProcessor(
                target, adapter_name,
                bias_type=peft_config.bias_type,
                bias_init=peft_config.bias_init,
                r=peft_config.r, dropout=peft_config.dropout, r_ratio=peft_config.r_ratio,
                ksot_phi_dim=getattr(peft_config, "ksot_phi_dim", None),
                ksot_lambda_init=getattr(peft_config, "ksot_lambda_init", 1e-3),
                ksot_activation=getattr(peft_config, "ksot_activation", "silu"),
                method=getattr(peft_config, "method", None),
                num_experts=getattr(peft_config, "num_experts", 1),
                inference_merge=getattr(peft_config, "inference_merge", "mean"),
                use_sft=getattr(peft_config, "use_sft", False),
                sft_type=getattr(peft_config, "sft_type", "channel"),
                sft_delta_scale=getattr(peft_config, "sft_delta_scale", 0.01),
                sft_clamp=getattr(peft_config, "sft_clamp", 0.2),
                sft_init=getattr(peft_config, "sft_init", "zero"),
                sft_apply_position=getattr(peft_config, "sft_apply_position", "after_softplus"),
                use_tf_sot=getattr(peft_config, "use_tf_sot", False),
                tf_sot_num_freqs=getattr(peft_config, "tf_sot_num_freqs", 4),
                tf_sot_num_basis=getattr(peft_config, "tf_sot_num_basis", 8),
                tf_sot_eps=getattr(peft_config, "tf_sot_eps", 0.1),
                tf_sot_max_seq_len=getattr(peft_config, "tf_sot_max_seq_len", 256),
                tf_sot_freq_grid=getattr(peft_config, "tf_sot_freq_grid", "geometric"),
                tf_sot_normalize_basis=getattr(peft_config, "tf_sot_normalize_basis", True),
                tf_sot_init_coeff=getattr(peft_config, "tf_sot_init_coeff", "zero"),
                use_cf_sot=getattr(peft_config, "use_cf_sot", False),
                cf_sot_num_freqs=getattr(peft_config, "cf_sot_num_freqs", 4),
                cf_sot_num_basis=getattr(peft_config, "cf_sot_num_basis", 8),
                cf_sot_eps=getattr(peft_config, "cf_sot_eps", 0.1),
                cf_sot_max_seq_len=getattr(peft_config, "cf_sot_max_seq_len", 256),
                cf_sot_freq_grid=getattr(peft_config, "cf_sot_freq_grid", "geometric"),
                cf_sot_normalize_basis=getattr(peft_config, "cf_sot_normalize_basis", True),
                cf_sot_context_center=getattr(peft_config, "cf_sot_context_center", True),
                cf_sot_context_reduce=getattr(peft_config, "cf_sot_context_reduce", "mean"),
                cf_sot_init_tf_coeff=getattr(peft_config, "cf_sot_init_tf_coeff", "zero"),
                cf_sot_init_context_alpha=getattr(peft_config, "cf_sot_init_context_alpha", "zero"),
            )

        return new_module


def _causal_cumavg(x, dim=-1):
    """
    Causal cumulative average along given dim.
    c_t = (1/t) * sum_{i=1}^{t} x_i

    Args:
        x: tensor of arbitrary shape
        dim: dimension along which to compute cumavg
    Returns:
        c: same shape as x
    """
    cum = torch.cumsum(x, dim=dim)
    length = x.size(dim)
    denom = torch.arange(1, length + 1, device=x.device, dtype=x.dtype)
    shape = [1] * x.ndim
    shape[dim] = length
    denom = denom.view(shape)
    return cum / denom


def build_temporal_fourier_basis(
    max_seq_len,
    num_freqs,
    normalize=True,
    dtype=torch.float32,
    device=None,
    freq_grid="geometric",
):
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be >= 1.")
    if num_freqs < 1:
        raise ValueError("num_freqs must be >= 1.")
    if str(freq_grid).lower() != "geometric":
        raise ValueError("Only geometric temporal Fourier grids are currently supported.")

    num_basis = 2 * int(num_freqs)
    omega_min = math.pi / float(max_seq_len)
    omega_max = math.pi

    if num_freqs == 1:
        freqs = torch.tensor([omega_min], device=device, dtype=torch.float32)
    else:
        log_freqs = torch.linspace(
            math.log(omega_min),
            math.log(omega_max),
            int(num_freqs),
            device=device,
            dtype=torch.float32,
        )
        freqs = torch.exp(log_freqs)

    t = torch.arange(int(max_seq_len), device=device, dtype=torch.float32)
    phases = t[:, None] * freqs[None, :]
    phi = torch.stack((torch.sin(phases), torch.cos(phases)), dim=-1).reshape(int(max_seq_len), num_basis)

    if normalize:
        phi = phi / math.sqrt(float(num_basis))

    return phi.to(dtype=dtype)


def compute_cf_sot_scale(
    gate: torch.Tensor,
    phi: torch.Tensor,
    tf_coeff: torch.Tensor,
    context_alpha: torch.Tensor,
    eps: float = 0.1,
    attention_mask: Optional[torch.Tensor] = None,
    context_center: bool = True,
    context_reduce: str = "mean",
    time_dim: int = 1,
    return_debug: bool = False,
):
    """
    Compute the CF-SOT bounded scale.

    gate is [B, L, D] when time_dim=1, and [B, D, L] when time_dim=-1.
    The returned scale matches the input layout: [B, L, 1] or [B, 1, L].
    """
    if gate.ndim != 3:
        raise ValueError(f"CF-SOT expects gate with 3 dims, got shape {tuple(gate.shape)}")
    if str(context_reduce).lower() != "mean":
        raise ValueError("CF-SOT currently supports context_reduce='mean' only.")

    if time_dim in (-1, 2):
        gate_seq = gate.transpose(1, 2)
        return_bdl = True
    elif time_dim == 1:
        gate_seq = gate
        return_bdl = False
    else:
        raise ValueError("CF-SOT time_dim must be 1 or -1.")

    B, L, _ = gate_seq.shape
    work_dtype = torch.float32
    gate_seq = gate_seq.to(dtype=work_dtype)
    phi = phi[:L].to(device=gate.device, dtype=work_dtype)
    coeff = tf_coeff.to(device=gate.device, dtype=work_dtype)
    alpha = context_alpha.to(device=gate.device, dtype=work_dtype)

    freq_raw = phi @ coeff
    freq_raw = freq_raw.view(1, L, 1)

    cum_gate = torch.cumsum(gate_seq, dim=1)
    steps = torch.arange(1, L + 1, device=gate.device, dtype=work_dtype).view(1, L, 1)
    context = cum_gate / steps
    context_scalar = context.mean(dim=-1, keepdim=True)

    if attention_mask is not None:
        mask = attention_mask[:, :L].to(device=gate.device, dtype=work_dtype).view(B, L, 1)
        context_scalar = context_scalar * mask
        if context_center:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            context_mean = context_scalar.sum(dim=1, keepdim=True) / denom
            context_scalar = (context_scalar - context_mean) * mask
    elif context_center:
        context_scalar = context_scalar - context_scalar.mean(dim=1, keepdim=True)

    context_raw = alpha * context_scalar
    traj_raw = freq_raw + context_raw
    scale_seq = 1.0 + float(eps) * torch.tanh(traj_raw)
    scale_seq = scale_seq.to(device=gate.device, dtype=gate.dtype)
    scale = scale_seq.transpose(1, 2) if return_bdl else scale_seq

    if not return_debug:
        return scale

    debug = {
        "gate_shape": tuple(gate.shape),
        "phi_shape": tuple(phi.shape),
        "tf_coeff_shape": tuple(tf_coeff.shape),
        "context_alpha_shape": tuple(context_alpha.shape),
        "freq_raw_shape": tuple(freq_raw.shape),
        "context_scalar_shape": tuple(context_scalar.shape),
        "context_raw_shape": tuple(context_raw.shape),
        "traj_raw_shape": tuple(traj_raw.shape),
        "scale_shape": tuple(scale.shape),
        "scale_min": float(scale.detach().float().min().cpu()),
        "scale_max": float(scale.detach().float().max().cpu()),
        "scale_abs_dev_mean": float((scale.detach().float() - 1.0).abs().mean().cpu()),
        "scale_abs_dev_max": float((scale.detach().float() - 1.0).abs().max().cpu()),
        "scale_std": float(scale.detach().float().std(unbiased=False).cpu()),
    }
    return scale, debug


class KernelizedStateOffset(nn.Module):
    """Feature-space replacement for the SOT C_t^T h' readout."""

    def __init__(self, d_inner, d_state, phi_dim=None, lambda_init=1e-3, activation="silu", dtype=None, device=None):
        super().__init__()

        if activation != "silu":
            raise ValueError(f"Unsupported K-SOT activation: {activation}")

        self.d_inner = d_inner
        self.d_state = d_state
        self.phi_dim = d_state if phi_dim is None else phi_dim
        self.activation = activation

        self.W_phi = nn.Linear(d_state, self.phi_dim, bias=False, dtype=dtype, device=device)
        self.h_phi = nn.Parameter(torch.empty(d_inner, 2 * self.phi_dim, dtype=dtype, device=device))
        self.lambda_scale = nn.Parameter(torch.full((1,), float(lambda_init), dtype=torch.float32, device=device))

        nn.init.xavier_uniform_(self.W_phi.weight)
        nn.init.normal_(self.h_phi, mean=0.0, std=0.02)

    def forward(self, C):
        no_seqlen_dim = C.ndim == 2
        if no_seqlen_dim:
            C = C.unsqueeze(-1)

        if C.ndim != 3:
            raise ValueError(f"K-SOT expects C with shape [B, N, L] or [B, N], got {tuple(C.shape)}")

        z = self.W_phi(rearrange(C, "b n l -> b l n"))
        phi = torch.cat([F.silu(z), F.silu(-z)], dim=-1)
        offset = torch.einsum("blf,df->bdl", phi, self.h_phi.to(dtype=phi.dtype))
        offset = offset * self.lambda_scale.to(dtype=offset.dtype)

        if no_seqlen_dim:
            offset = offset.squeeze(-1)

        return offset


class SuffixTuningBiasProcessor(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer, adapter_name, r=None, **kwargs) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer

        self.suffixtuning_bias = nn.ParameterDict({}) if r is None else nn.ModuleDict({})
        self.suffixtuning_ia3 = nn.ParameterDict({})
        self.suffixtuning_scale = nn.ParameterDict({})
        self.suffixtuning_kernel = nn.ModuleDict({})
        self.suffixtuning_tf_coeff = nn.ParameterDict({})
        self.cf_sot_tf_coeff = nn.ParameterDict({})
        self.cf_sot_context_alpha = nn.ParameterDict({})
        self.sft_delta_bias = nn.ParameterDict({})
        self.suffixtuning_type = {}
        self.suffixtuning_method = {}
        self.suffixtuning_num_experts = {}
        self.suffixtuning_inference_merge = {}
        self.tf_sot_enabled = {}
        self.tf_sot_num_freqs = {}
        self.tf_sot_num_basis = {}
        self.tf_sot_eps = {}
        self.tf_sot_max_seq_len = {}
        self.tf_sot_freq_grid = {}
        self.tf_sot_normalize_basis = {}
        self.last_tf_sot_debug = {}
        self.cf_sot_enabled = {}
        self.cf_sot_num_freqs = {}
        self.cf_sot_num_basis = {}
        self.cf_sot_eps = {}
        self.cf_sot_max_seq_len = {}
        self.cf_sot_freq_grid = {}
        self.cf_sot_normalize_basis = {}
        self.cf_sot_context_center = {}
        self.cf_sot_context_reduce = {}
        self.cf_sot_phi_buffer_name = {}
        self.last_cf_sot_debug = {}
        self.sft_enabled = {}
        self.sft_type = {}
        self.sft_delta_scale = {}
        self.sft_clamp = {}
        self.sft_init = {}
        self.sft_apply_position = {}

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r=r,
            **kwargs
        )

    def _create_param(self, shape, init_type, dtype, device, r=None, dropout=None, r_ratio=None):
        if r is not None:
            d, n, = shape
            return LoraParam(d, n, r, dropout, ratio=r_ratio, dtype=dtype, device=device)
        else:
            match init_type:
                case SuffixTuningBiasInit.RANDOM:
                    data = torch.randn(shape, dtype=dtype, device=device) * 0.1
                case SuffixTuningBiasInit.ZERO:
                    data = torch.zeros(shape, dtype=dtype, device=device)

            return nn.Parameter(data)

    def _create_adamix_param(self, shape, init_type, dtype, device, num_experts):
        expert_shape = [num_experts, *shape]
        match init_type:
            case SuffixTuningBiasInit.RANDOM:
                data = torch.randn(expert_shape, dtype=dtype, device=device) * 0.1
            case SuffixTuningBiasInit.ZERO:
                data = torch.zeros(expert_shape, dtype=dtype, device=device)

        return nn.Parameter(data)

    def _select_adamix_offset(self, active_adapter, param):
        if param.ndim < 1:
            raise ValueError("AdaMix-SOT offset experts must have an expert dimension.")

        if self.training:
            expert_idx = torch.randint(param.shape[0], (1,), device=param.device).item()
            return param[expert_idx]

        merge_mode = self.suffixtuning_inference_merge[active_adapter]
        if merge_mode != "mean":
            raise ValueError(f"Unsupported AdaMix-SOT inference merge mode: {merge_mode}")
        return param.mean(dim=0)

    def _get_bias_param(self, active_adapter):
        param = self.suffixtuning_bias[active_adapter]
        if isinstance(param, nn.Module):
            param = param()

        if self.suffixtuning_method.get(active_adapter) == "adamix_sot":
            param = self._select_adamix_offset(active_adapter, param)

        return param

    @staticmethod
    def _adapter_buffer_name(prefix, adapter_name):
        safe_adapter = str(adapter_name).replace(".", "_")
        return f"{prefix}_{safe_adapter}"

    def _register_cf_sot_basis(self, adapter_name, max_seq_len, num_freqs, normalize, freq_grid):
        buffer_name = self._adapter_buffer_name("cf_sot_phi", adapter_name)
        phi = build_temporal_fourier_basis(
            max_seq_len=max_seq_len,
            num_freqs=num_freqs,
            normalize=normalize,
            dtype=torch.float32,
            device=self.base_layer.device,
            freq_grid=freq_grid,
        )
        if buffer_name in self._buffers:
            self._buffers[buffer_name] = phi
        else:
            self.register_buffer(buffer_name, phi, persistent=False)
        self.cf_sot_phi_buffer_name[adapter_name] = buffer_name

    def _get_cf_sot_phi(self, seq_len, active_adapter, device):
        max_seq_len = self.cf_sot_max_seq_len[active_adapter]
        num_freqs = self.cf_sot_num_freqs[active_adapter]
        normalize = self.cf_sot_normalize_basis[active_adapter]
        freq_grid = self.cf_sot_freq_grid[active_adapter]
        buffer_name = self.cf_sot_phi_buffer_name.get(active_adapter)
        phi = getattr(self, buffer_name, None) if buffer_name is not None else None

        if phi is None or seq_len > phi.shape[0]:
            phi = build_temporal_fourier_basis(
                max_seq_len=seq_len,
                num_freqs=num_freqs,
                normalize=normalize,
                dtype=torch.float32,
                device=device,
                freq_grid=freq_grid,
            )
        else:
            phi = phi[:seq_len].to(device=device, dtype=torch.float32)

        return phi

    def update_layer(
        self,
        adapter_name,
        bias_type,
        bias_init,
        r=None,
        dropout=None,
        r_ratio=None,
        ksot_phi_dim=None,
        ksot_lambda_init=1e-3,
        ksot_activation="silu",
        method=None,
        num_experts=1,
        inference_merge="mean",
        use_sft=False,
        sft_type="channel",
        sft_delta_scale=0.01,
        sft_clamp=0.2,
        sft_init="zero",
        sft_apply_position="after_softplus",
        use_tf_sot=False,
        tf_sot_num_freqs=4,
        tf_sot_num_basis=8,
        tf_sot_eps=0.1,
        tf_sot_max_seq_len=256,
        tf_sot_freq_grid="geometric",
        tf_sot_normalize_basis=True,
        tf_sot_init_coeff="zero",
        use_cf_sot=False,
        cf_sot_num_freqs=4,
        cf_sot_num_basis=8,
        cf_sot_eps=0.1,
        cf_sot_max_seq_len=256,
        cf_sot_freq_grid="geometric",
        cf_sot_normalize_basis=True,
        cf_sot_context_center=True,
        cf_sot_context_reduce="mean",
        cf_sot_init_tf_coeff="zero",
        cf_sot_init_context_alpha="zero",
    ):
        dims = self.base_layer.all_dims
        method = None if method is None else str(method).lower()
        num_experts = int(num_experts)
        inference_merge = str(inference_merge).lower()
        use_sft = bool(use_sft)
        sft_type = str(sft_type).lower()
        sft_delta_scale = float(sft_delta_scale)
        sft_clamp = None if sft_clamp is None else float(sft_clamp)
        sft_init = str(sft_init).lower()
        sft_apply_position = str(sft_apply_position).lower()
        use_tf_sot = bool(use_tf_sot)
        tf_sot_num_freqs = int(tf_sot_num_freqs)
        tf_sot_num_basis = int(tf_sot_num_basis)
        tf_sot_eps = float(tf_sot_eps)
        tf_sot_max_seq_len = int(tf_sot_max_seq_len)
        tf_sot_freq_grid = str(tf_sot_freq_grid).lower()
        tf_sot_normalize_basis = bool(tf_sot_normalize_basis)
        tf_sot_init_coeff = str(tf_sot_init_coeff).lower()
        use_cf_sot = bool(use_cf_sot)
        cf_sot_num_freqs = int(cf_sot_num_freqs)
        cf_sot_num_basis = int(cf_sot_num_basis)
        cf_sot_eps = float(cf_sot_eps)
        cf_sot_max_seq_len = int(cf_sot_max_seq_len)
        cf_sot_freq_grid = str(cf_sot_freq_grid).lower()
        cf_sot_normalize_basis = bool(cf_sot_normalize_basis)
        cf_sot_context_center = bool(cf_sot_context_center)
        cf_sot_context_reduce = str(cf_sot_context_reduce).lower()
        cf_sot_init_tf_coeff = str(cf_sot_init_tf_coeff).lower()
        cf_sot_init_context_alpha = str(cf_sot_init_context_alpha).lower()

        if use_tf_sot and use_cf_sot:
            raise ValueError("use_tf_sot and use_cf_sot cannot both be enabled.")

        if use_tf_sot:
            if bias_type != SuffixTuningBiasType.SILU_Z_C:
                raise ValueError("TF-SOT currently supports the original SOT bias_type SILU_Z_C only.")
            if method is not None:
                raise ValueError("TF-SOT should not be mixed with other SOT methods.")
            if use_sft:
                raise ValueError("TF-SOT should not be mixed with SOT+SFT in the same adapter.")
            if tf_sot_num_freqs < 1:
                raise ValueError("TF-SOT requires tf_sot_num_freqs >= 1.")
            if tf_sot_num_basis != 2 * tf_sot_num_freqs:
                raise ValueError("TF-SOT requires tf_sot_num_basis == 2 * tf_sot_num_freqs.")
            if tf_sot_eps < 0:
                raise ValueError("TF-SOT requires tf_sot_eps >= 0.")
            if tf_sot_max_seq_len < 1:
                raise ValueError("TF-SOT requires tf_sot_max_seq_len >= 1.")
            if tf_sot_freq_grid != "geometric":
                raise ValueError("TF-SOT currently supports tf_sot_freq_grid='geometric' only.")
            if tf_sot_init_coeff != "zero":
                raise ValueError("TF-SOT currently supports tf_sot_init_coeff='zero' only.")

        if use_cf_sot:
            if bias_type != SuffixTuningBiasType.SILU_Z_C:
                raise ValueError("CF-SOT currently supports the original SOT bias_type SILU_Z_C only.")
            if method is not None:
                raise ValueError("CF-SOT should not be mixed with other SOT methods.")
            if use_sft:
                raise ValueError("CF-SOT should not be mixed with SOT+SFT in the same adapter.")
            if cf_sot_num_freqs < 1:
                raise ValueError("CF-SOT requires cf_sot_num_freqs >= 1.")
            if cf_sot_num_basis != 2 * cf_sot_num_freqs:
                raise ValueError("CF-SOT requires cf_sot_num_basis == 2 * cf_sot_num_freqs.")
            if cf_sot_eps < 0:
                raise ValueError("CF-SOT requires cf_sot_eps >= 0.")
            if cf_sot_max_seq_len < 1:
                raise ValueError("CF-SOT requires cf_sot_max_seq_len >= 1.")
            if cf_sot_freq_grid != "geometric":
                raise ValueError("CF-SOT currently supports cf_sot_freq_grid='geometric' only.")
            if cf_sot_context_reduce != "mean":
                raise ValueError("CF-SOT currently supports cf_sot_context_reduce='mean' only.")
            if cf_sot_init_tf_coeff != "zero":
                raise ValueError("CF-SOT currently supports cf_sot_init_tf_coeff='zero' only.")
            if cf_sot_init_context_alpha != "zero":
                raise ValueError("CF-SOT currently supports cf_sot_init_context_alpha='zero' only.")

        if method == "adamix_sot":
            if bias_type != SuffixTuningBiasType.SILU_Z_C:
                raise ValueError("AdaMix-SOT currently supports the original SOT bias_type SILU_Z_C only.")
            if r is not None:
                raise ValueError("AdaMix-SOT does not support LoRA-style factorized SOT offsets.")
            if num_experts < 1:
                raise ValueError("AdaMix-SOT requires num_experts >= 1.")
            if inference_merge != "mean":
                raise ValueError("AdaMix-SOT currently supports inference_merge='mean' only.")

        if use_sft:
            if sft_type not in ("channel", "scalar"):
                raise ValueError(f"Unsupported sft_type: {sft_type}")
            if sft_init != "zero":
                raise ValueError("SFT currently supports sft_init='zero' only.")
            if sft_apply_position != "after_softplus":
                raise ValueError("SFT currently supports sft_apply_position='after_softplus' only.")

        # DSOT uses the same shape as SILU_Z_C, preserving the SOT parameter count.
        shape = {
            SuffixTuningBiasType.SILU_Z_C: [dims["d"], dims["n"]],
            SuffixTuningBiasType.SILU_Z_C_IA3: [dims["d"], dims["n"]],
            SuffixTuningBiasType.SILU_Z_C_SCALE: [dims["d"], dims["n"]],
            SuffixTuningBiasType.SILU_Z: [dims["d"]],
            SuffixTuningBiasType.DYNAMIC_SILU_Z_C: [dims["d"], dims["n"]],
            SuffixTuningBiasType.KEY_AWARE_DYNAMIC_SILU_Z_C: [dims["d"], dims["n"]],
        }.get(bias_type)

        if bias_type == SuffixTuningBiasType.KERNEL_SILU_Z_C:
            self.suffixtuning_kernel[adapter_name] = KernelizedStateOffset(
                d_inner=dims["d"],
                d_state=dims["n"],
                phi_dim=ksot_phi_dim,
                lambda_init=ksot_lambda_init,
                activation=ksot_activation,
                dtype=self.base_layer.dtype,
                device=self.base_layer.device,
            )
        else:
            if method == "adamix_sot":
                param = self._create_adamix_param(
                    shape,
                    bias_init,
                    self.base_layer.dtype,
                    self.base_layer.device,
                    num_experts=num_experts,
                )
            else:
                param = self._create_param(shape, bias_init, self.base_layer.dtype, self.base_layer.device,
                                           r=r, dropout=dropout, r_ratio=r_ratio)
            self.suffixtuning_bias[adapter_name] = param
            if bias_type == SuffixTuningBiasType.SILU_Z_C_IA3:
                self.suffixtuning_ia3[adapter_name] = nn.Parameter(
                    torch.ones(dims["n"], dtype=self.base_layer.dtype, device=self.base_layer.device)
                )
            if bias_type == SuffixTuningBiasType.SILU_Z_C_SCALE:
                self.suffixtuning_scale[adapter_name] = nn.Parameter(
                    torch.ones(dims["d"], dtype=self.base_layer.dtype, device=self.base_layer.device)
                )
                assert self.suffixtuning_scale[adapter_name].shape == (dims["d"],)
                assert self.suffixtuning_scale[adapter_name].requires_grad
                assert torch.allclose(
                    self.suffixtuning_scale[adapter_name].detach().float().mean(),
                    torch.ones((), dtype=torch.float32, device=self.base_layer.device),
                )

        if use_tf_sot:
            self.suffixtuning_tf_coeff[adapter_name] = nn.Parameter(
                torch.zeros(tf_sot_num_basis, dtype=torch.float32, device=self.base_layer.device)
            )

        if use_cf_sot:
            self.cf_sot_tf_coeff[adapter_name] = nn.Parameter(
                torch.zeros(cf_sot_num_basis, dtype=torch.float32, device=self.base_layer.device)
            )
            self.cf_sot_context_alpha[adapter_name] = nn.Parameter(
                torch.zeros(1, dtype=torch.float32, device=self.base_layer.device)
            )
            self._register_cf_sot_basis(
                adapter_name=adapter_name,
                max_seq_len=cf_sot_max_seq_len,
                num_freqs=cf_sot_num_freqs,
                normalize=cf_sot_normalize_basis,
                freq_grid=cf_sot_freq_grid,
            )

        if use_sft:
            if sft_type == "channel":
                sft_shape = (dims["d"],)
            elif sft_type == "scalar":
                sft_shape = (1,)
            else:
                raise ValueError(f"Unsupported sft_type: {sft_type}")
            self.sft_delta_bias[adapter_name] = nn.Parameter(
                torch.zeros(sft_shape, dtype=self.base_layer.dtype, device=self.base_layer.device)
            )

        self.suffixtuning_type[adapter_name] = bias_type
        self.suffixtuning_method[adapter_name] = method
        self.suffixtuning_num_experts[adapter_name] = num_experts
        self.suffixtuning_inference_merge[adapter_name] = inference_merge
        self.tf_sot_enabled[adapter_name] = use_tf_sot
        self.tf_sot_num_freqs[adapter_name] = tf_sot_num_freqs
        self.tf_sot_num_basis[adapter_name] = tf_sot_num_basis
        self.tf_sot_eps[adapter_name] = tf_sot_eps
        self.tf_sot_max_seq_len[adapter_name] = tf_sot_max_seq_len
        self.tf_sot_freq_grid[adapter_name] = tf_sot_freq_grid
        self.tf_sot_normalize_basis[adapter_name] = tf_sot_normalize_basis
        self.cf_sot_enabled[adapter_name] = use_cf_sot
        self.cf_sot_num_freqs[adapter_name] = cf_sot_num_freqs
        self.cf_sot_num_basis[adapter_name] = cf_sot_num_basis
        self.cf_sot_eps[adapter_name] = cf_sot_eps
        self.cf_sot_max_seq_len[adapter_name] = cf_sot_max_seq_len
        self.cf_sot_freq_grid[adapter_name] = cf_sot_freq_grid
        self.cf_sot_normalize_basis[adapter_name] = cf_sot_normalize_basis
        self.cf_sot_context_center[adapter_name] = cf_sot_context_center
        self.cf_sot_context_reduce[adapter_name] = cf_sot_context_reduce
        self.sft_enabled[adapter_name] = use_sft
        self.sft_type[adapter_name] = sft_type
        self.sft_delta_scale[adapter_name] = sft_delta_scale
        self.sft_clamp[adapter_name] = sft_clamp
        self.sft_init[adapter_name] = sft_init
        self.sft_apply_position[adapter_name] = sft_apply_position

        self.set_adapter(self.active_adapters)

        self.key_token_weights = None
        self.beta_key_context = 0.1
        self.key_context_eps = 1e-6
        self.last_keyaware_stats = {}

    def _build_tf_sot_basis(self, seq_len, device, dtype, active_adapter):
        num_freqs = self.tf_sot_num_freqs[active_adapter]
        num_basis = self.tf_sot_num_basis[active_adapter]
        max_seq_len = self.tf_sot_max_seq_len[active_adapter]
        freq_grid = self.tf_sot_freq_grid[active_adapter]

        if freq_grid != "geometric":
            raise RuntimeError(f"Unsupported TF-SOT frequency grid: {freq_grid}")

        omega_min = math.pi / float(max_seq_len)
        omega_max = math.pi
        if num_freqs == 1:
            freqs = torch.tensor([omega_min], device=device, dtype=torch.float32)
        else:
            log_freqs = torch.linspace(
                math.log(omega_min),
                math.log(omega_max),
                num_freqs,
                device=device,
                dtype=torch.float32,
            )
            freqs = torch.exp(log_freqs)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        phases = t[:, None] * freqs[None, :]
        phi = torch.stack((torch.sin(phases), torch.cos(phases)), dim=-1).reshape(seq_len, num_basis)

        if self.tf_sot_normalize_basis[active_adapter]:
            phi = phi / math.sqrt(float(num_basis))

        return phi.to(dtype=dtype)

    def compute_tf_sot_scale(self, seq_len, device, dtype, active_adapter):
        coeff = self.suffixtuning_tf_coeff[active_adapter]
        phi = self._build_tf_sot_basis(seq_len, device, torch.float32, active_adapter)
        raw = phi @ coeff.to(device=device, dtype=torch.float32)
        scale = 1.0 + self.tf_sot_eps[active_adapter] * torch.tanh(raw)
        scale = scale.to(device=device, dtype=dtype).view(1, 1, seq_len)

        if not hasattr(self, "last_tf_sot_debug"):
            self.last_tf_sot_debug = {}
        self.last_tf_sot_debug[active_adapter] = {
            "phi_shape": tuple(phi.shape),
            "coeff_shape": tuple(coeff.shape),
            "scale_shape": tuple(scale.shape),
            "scale_min": float(scale.detach().float().min().cpu()),
            "scale_max": float(scale.detach().float().max().cpu()),
        }

        return scale

    def compute_cf_sot_scale(self, gate, active_adapter, attention_mask=None):
        seq_len = gate.shape[-1]
        phi = self._get_cf_sot_phi(seq_len, active_adapter, gate.device)
        scale, debug = compute_cf_sot_scale(
            gate=gate,
            phi=phi,
            tf_coeff=self.cf_sot_tf_coeff[active_adapter],
            context_alpha=self.cf_sot_context_alpha[active_adapter],
            eps=self.cf_sot_eps[active_adapter],
            attention_mask=attention_mask,
            context_center=self.cf_sot_context_center[active_adapter],
            context_reduce=self.cf_sot_context_reduce[active_adapter],
            time_dim=-1,
            return_debug=True,
        )

        if not hasattr(self, "last_cf_sot_debug"):
            self.last_cf_sot_debug = {}
        self.last_cf_sot_debug[active_adapter] = debug
        return scale

    def has_sft_delta_tuning(self):
        return any(
            self.sft_enabled.get(active_adapter, False) and active_adapter in self.sft_delta_bias
            for active_adapter in self.active_adapters
        )

    def apply_sft_delta_tuning(self, delta, adapter_name=None):
        active_adapters = [adapter_name] if adapter_name is not None else self.active_adapters

        for active_adapter in active_adapters:
            if not self.sft_enabled.get(active_adapter, False):
                continue
            if active_adapter not in self.sft_delta_bias:
                continue
            if self.sft_apply_position.get(active_adapter) != "after_softplus":
                raise RuntimeError(
                    f"Unsupported SFT apply position: {self.sft_apply_position.get(active_adapter)}"
                )

            r = self.sft_delta_bias[active_adapter]
            log_multiplier = self.sft_delta_scale[active_adapter] * r.float()
            clamp = self.sft_clamp[active_adapter]
            if clamp is not None and clamp > 0:
                log_multiplier = torch.clamp(log_multiplier, min=-clamp, max=clamp)

            multiplier = torch.exp(log_multiplier).to(dtype=delta.dtype, device=delta.device)

            if r.numel() == 1:
                delta = delta * multiplier
                continue

            if delta.dim() == 3:
                if delta.shape[-1] == r.numel():
                    multiplier = multiplier.view(1, 1, -1)
                elif delta.shape[1] == r.numel():
                    multiplier = multiplier.view(1, -1, 1)
                elif delta.shape[0] == r.numel():
                    multiplier = multiplier.view(-1, 1, 1)
                else:
                    raise RuntimeError(
                        f"Cannot broadcast SFT delta bias: delta={tuple(delta.shape)}, r={tuple(r.shape)}"
                    )
            elif delta.dim() == 2:
                if delta.shape[-1] == r.numel():
                    multiplier = multiplier.view(1, -1)
                elif delta.shape[0] == r.numel():
                    multiplier = multiplier.view(-1, 1)
                else:
                    raise RuntimeError(
                        f"Cannot broadcast SFT delta bias: delta={tuple(delta.shape)}, r={tuple(r.shape)}"
                    )
            else:
                raise RuntimeError(f"Unsupported delta dim for SFT: delta={tuple(delta.shape)}, r={tuple(r.shape)}")

            delta = delta * multiplier

        return delta

    def set_keyaware_context(self, alpha, beta_key_context=0.1, eps=1e-6):
        self.key_token_weights = alpha
        self.beta_key_context = beta_key_context
        self.key_context_eps = eps

    def get_ia3_scale(self, dtype=None, device=None):
        ia3_adapters = [
            active_adapter
            for active_adapter in self.active_adapters
            if self.suffixtuning_type[active_adapter] == SuffixTuningBiasType.SILU_Z_C_IA3
        ]
        if not ia3_adapters:
            return None
        if len(self.active_adapters) > 1:
            raise NotImplementedError("SILU_Z_C_IA3 currently supports exactly one active adapter.")

        scale = self.suffixtuning_ia3[ia3_adapters[0]]
        if dtype is not None or device is not None:
            scale = scale.to(dtype=dtype or scale.dtype, device=device or scale.device)
        return scale

    def clear_keyaware_context(self):
        self.key_token_weights = None
        self.last_keyaware_stats = {}

    def _keyaware_causal_context(self, z_silu, vanilla_context):
        alpha = self.key_token_weights

        if alpha is None:
            self.last_keyaware_stats = {}
            return vanilla_context

        alpha = alpha.to(device=z_silu.device, dtype=z_silu.dtype)
        if alpha.ndim == 2:
            alpha = alpha.unsqueeze(1)
        elif alpha.ndim == 3 and alpha.shape[1] != 1:
            raise ValueError(f"Expected alpha shape [B, L] or [B, 1, L], got {tuple(alpha.shape)}")

        if alpha.shape[-1] != z_silu.shape[-1]:
            if alpha.shape[-1] < z_silu.shape[-1]:
                pad = z_silu.shape[-1] - alpha.shape[-1]
                alpha = F.pad(alpha, (0, pad), value=0)
            else:
                alpha = alpha[..., :z_silu.shape[-1]]

        weighted_sum = torch.cumsum(z_silu * alpha, dim=-1)
        weight_sum = torch.cumsum(alpha, dim=-1)
        key_context = weighted_sum / (weight_sum + self.key_context_eps)
        beta = float(self.beta_key_context)
        blended_context = (1.0 - beta) * vanilla_context + beta * key_context

        with torch.no_grad():
            self.last_keyaware_stats = {
                "c_key_norm": key_context.float().norm(dim=1).mean().detach(),
                "c_t_norm": vanilla_context.float().norm(dim=1).mean().detach(),
                "c_tilde_norm": blended_context.float().norm(dim=1).mean().detach(),
            }

        return blended_context

    def forward(self, x, z, A, B, C, D, dt):
        y = x

        for active_adapter in self.active_adapters:
            bias_type = self.suffixtuning_type[active_adapter]

            no_seqlen_dim = z.ndim == 2
            z_readout = z
            C_readout = C

            if no_seqlen_dim:
                z_readout = z.unsqueeze(2)
                C_readout = C.unsqueeze(2)

            match bias_type:
                case SuffixTuningBiasType.SILU_Z_C | SuffixTuningBiasType.SILU_Z_C_IA3 | SuffixTuningBiasType.SILU_Z_C_SCALE:
                    param = self._get_bias_param(active_adapter)
                    # SOT: y_add[b,d,l] = sum_n silu(z)[b,d,l] * C[b,n,l] * h'[d,n]
                    gate = F.silu(z_readout)
                    y_add = torch.einsum("bdl,bnl,dn -> bdl", gate, C_readout, param)
                    if getattr(self, "cf_sot_enabled", {}).get(active_adapter, False):
                        if bias_type != SuffixTuningBiasType.SILU_Z_C:
                            raise RuntimeError("CF-SOT is only valid for the original SILU_Z_C SOT path.")
                        scale = self.compute_cf_sot_scale(
                            gate=gate,
                            active_adapter=active_adapter,
                            attention_mask=None,
                        )
                        y_add = y_add * scale
                        self.last_cf_sot_debug[active_adapter]["h_prime_shape"] = tuple(param.shape)
                        self.last_cf_sot_debug[active_adapter]["offset_shape"] = tuple(y_add.shape)
                    elif getattr(self, "tf_sot_enabled", {}).get(active_adapter, False):
                        if bias_type != SuffixTuningBiasType.SILU_Z_C:
                            raise RuntimeError("TF-SOT is only valid for the original SILU_Z_C SOT path.")
                        scale = self.compute_tf_sot_scale(
                            seq_len=z_readout.shape[-1],
                            device=y_add.device,
                            dtype=y_add.dtype,
                            active_adapter=active_adapter,
                        )
                        y_add = y_add * scale
                    if bias_type == SuffixTuningBiasType.SILU_Z_C_SCALE:
                        alpha = self.suffixtuning_scale[active_adapter].to(
                            dtype=y_add.dtype,
                            device=y_add.device,
                        )
                        y_add_scaled = alpha.view(1, -1, 1) * y_add
                        assert y_add_scaled.shape == y_add.shape
                        y_add = y_add_scaled

                case SuffixTuningBiasType.SILU_Z:
                    param = self._get_bias_param(active_adapter)
                    y_add = torch.einsum("bdl,d -> bdl", F.silu(z_readout), param)

                case SuffixTuningBiasType.DYNAMIC_SILU_Z_C:
                    param = self._get_bias_param(active_adapter)
                    # DSOT: modulate SOT with causal context of silu(z).
                    z_silu = F.silu(z_readout)
                    c = _causal_cumavg(z_silu, dim=-1)
                    z_silu_modulated = z_silu * c
                    y_add = torch.einsum("bdl,bnl,dn -> bdl", z_silu_modulated, C_readout, param)

                case SuffixTuningBiasType.KERNEL_SILU_Z_C:
                    # K-SOT keeps the original Mamba/SOT gate and replaces only C_t^T h'
                    # with phi(C_t)^T h_phi.
                    offset = self.suffixtuning_kernel[active_adapter](C_readout)
                    y_add = F.silu(z_readout) * offset

                case SuffixTuningBiasType.KEY_AWARE_DYNAMIC_SILU_Z_C:
                    param = self._get_bias_param(active_adapter)
                    z_silu = F.silu(z_readout)
                    c = _causal_cumavg(z_silu, dim=-1)
                    c_tilde = self._keyaware_causal_context(z_silu, c)
                    z_silu_modulated = z_silu * c_tilde
                    y_add = torch.einsum("bdl,bnl,dn -> bdl", z_silu_modulated, C_readout, param)

            if no_seqlen_dim:
                y_add = y_add.squeeze(2)

            y = y + y_add

        return y


def set_keyaware_context(model, alpha, beta_key_context=0.1, eps=1e-6):
    for module in model.modules():
        if isinstance(module, SuffixTuningBiasProcessor):
            module.set_keyaware_context(alpha, beta_key_context=beta_key_context, eps=eps)


def clear_keyaware_context(model):
    for module in model.modules():
        if isinstance(module, SuffixTuningBiasProcessor):
            module.clear_keyaware_context()


def collect_keyaware_context_stats(model):
    stats = {}
    count = 0

    for module in model.modules():
        if isinstance(module, SuffixTuningBiasProcessor) and module.last_keyaware_stats:
            for key, value in module.last_keyaware_stats.items():
                stats[key] = stats.get(key, 0.0) + float(value.detach().cpu())
            count += 1

    if count == 0:
        return {}

    return {key: value / count for key, value in stats.items()}
