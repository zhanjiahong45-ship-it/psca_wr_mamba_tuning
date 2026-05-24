from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class SNOFTLayer(nn.Module):
    def __init__(
        self,
        d_inner: int,
        num_groups: int = 16,
        chunk_size: int = 32,
        router_rank: int = 8,
        tau_logit_init: float = 3.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_inner % num_groups == 0

        self.d_inner = int(d_inner)
        self.num_groups = int(num_groups)
        self.chunk_size = int(chunk_size)
        self.router_rank = int(router_rank)
        self.tau_logit_init = float(tau_logit_init)
        self.eps = float(eps)
        self.group_dim = self.d_inner // self.num_groups

        self.router = nn.Sequential(
            nn.Linear(self.d_inner, self.router_rank),
            nn.SiLU(),
            nn.Linear(self.router_rank, self.router_rank),
            nn.SiLU(),
        )
        self.gate_head = nn.Linear(self.router_rank, self.num_groups)
        self.u = nn.Parameter(torch.randn(self.num_groups, self.group_dim) * 0.02)
        self.alpha_bc = nn.Parameter(torch.zeros(1))
        self.alpha_delta = nn.Parameter(torch.zeros(1))
        self.tau_logit = nn.Parameter(torch.full((self.d_inner,), self.tau_logit_init))

        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def group_householder(self, v: torch.Tensor) -> torch.Tensor:
        B, L, D = v.shape
        G = self.num_groups
        dg = self.group_dim
        x = v.reshape(B, L, G, dg)
        u = self.u.to(dtype=v.dtype, device=v.device).view(1, 1, G, dg)
        uv = (x * u).sum(dim=-1, keepdim=True)
        uu = (u * u).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        y = x - 2.0 * uv / uu * u
        return y.reshape(B, L, D)

    def chunk_gate(self, v: torch.Tensor) -> torch.Tensor:
        B, L, D = v.shape
        C = self.chunk_size
        pad_len = (C - L % C) % C
        if pad_len > 0:
            pad = v[:, -1:, :].expand(B, pad_len, D)
            v_pad = torch.cat([v, pad], dim=1)
        else:
            v_pad = v

        L_pad = v_pad.shape[1]
        num_chunks = L_pad // C
        v_chunk = v_pad.reshape(B, num_chunks, C, D)
        s = v_chunk.mean(dim=2)
        r = self.router(s)
        g = torch.sigmoid(self.gate_head(r))
        g = g.unsqueeze(2).expand(B, num_chunks, C, self.num_groups)
        g = g.reshape(B, L_pad, self.num_groups)
        return g[:, :L, :]

    def ema_lpf(self, v: torch.Tensor) -> torch.Tensor:
        B, L, D = v.shape
        tau = torch.sigmoid(self.tau_logit).to(dtype=v.dtype, device=v.device).view(1, 1, D)
        outs = []
        y = v[:, 0:1, :]
        outs.append(y)
        # TODO: Replace this correctness-first loop with a scan-compatible or
        # causal-conv implementation if SNOFT-E becomes a training bottleneck.
        for t in range(1, L):
            y = (1.0 - tau) * y + tau * v[:, t : t + 1, :]
            outs.append(y)
        return torch.cat(outs, dim=1)

    def forward(self, v: torch.Tensor):
        B, L, D = v.shape
        g = self.chunk_gate(v)
        rv = self.group_householder(v)

        v_group = v.reshape(B, L, self.num_groups, self.group_dim)
        rv_group = rv.reshape(B, L, self.num_groups, self.group_dim)
        g = g.unsqueeze(-1)

        v_bc = v_group + self.alpha_bc * g * (rv_group - v_group)
        v_bc = v_bc.reshape(B, L, D)

        ema_v = self.ema_lpf(v_bc)
        v_delta = v_bc + self.alpha_delta * (ema_v - v_bc)
        return v_bc, v_delta


@dataclass
class SNOFTConfig:
    method: Optional[str] = None
    enabled: bool = False
    num_groups: int = 16
    chunk_size: int = 32
    router_rank: int = 8
    tau_logit_init: float = 3.0
    eps: float = 1e-6
    freeze_backbone: bool = True
    train_task_head: bool = True
    target_layers: Any = "all"
    sanity_check: bool = True

    def __post_init__(self):
        method_was_set = self.method is not None
        self.method = "snoft_e" if self.method is None else str(self.method).lower()
        if method_was_set and self.method in ("snoft", "snoft_e"):
            self.enabled = True
        self.enabled = bool(self.enabled)
        self.num_groups = int(self.num_groups)
        self.chunk_size = int(self.chunk_size)
        self.router_rank = int(self.router_rank)
        self.tau_logit_init = float(self.tau_logit_init)
        self.eps = float(self.eps)
        self.freeze_backbone = bool(self.freeze_backbone)
        self.train_task_head = bool(self.train_task_head)
        self.sanity_check = bool(self.sanity_check)
        if self.num_groups < 1:
            raise ValueError("SNOFT-E requires num_groups >= 1.")
        if self.chunk_size < 1:
            raise ValueError("SNOFT-E requires chunk_size >= 1.")
        if self.router_rank < 1:
            raise ValueError("SNOFT-E requires router_rank >= 1.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "enabled": self.enabled,
            "num_groups": self.num_groups,
            "chunk_size": self.chunk_size,
            "router_rank": self.router_rank,
            "tau_logit_init": self.tau_logit_init,
            "eps": self.eps,
            "freeze_backbone": self.freeze_backbone,
            "train_task_head": self.train_task_head,
            "target_layers": self.target_layers,
            "sanity_check": self.sanity_check,
        }


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, SimpleNamespace):
        return vars(obj)
    return {
        name: getattr(obj, name)
        for name in dir(obj)
        if not name.startswith("_") and not callable(getattr(obj, name))
    }


def is_snoft_config_dict(config: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(config, dict):
        return False
    method = str(config.get("method", "")).lower()
    if method in ("snoft", "snoft_e"):
        return True
    if bool(config.get("use_snoft", False)):
        return True
    snoft = config.get("snoft")
    return isinstance(snoft, dict) and bool(snoft.get("enabled", False))


def is_snoft_requested(peft_config: Optional[Dict[str, Any]], args: Any = None) -> bool:
    if is_snoft_config_dict(peft_config):
        return True
    args_dict = _as_dict(args)
    method = str(args_dict.get("method", "")).lower()
    if method in ("snoft", "snoft_e"):
        return True
    if bool(args_dict.get("use_snoft", False)):
        return True
    snoft = args_dict.get("snoft")
    return isinstance(snoft, dict) and bool(snoft.get("enabled", False))


def merge_snoft_config(peft_config: Optional[Dict[str, Any]] = None, args: Any = None) -> SNOFTConfig:
    values: Dict[str, Any] = {}

    if isinstance(peft_config, dict):
        if "method" in peft_config:
            values["method"] = peft_config["method"]
        if "use_snoft" in peft_config:
            values["enabled"] = peft_config["use_snoft"]
        nested = peft_config.get("snoft")
        if isinstance(nested, dict):
            values.update(nested)
        for key in SNOFTConfig.__dataclass_fields__.keys():
            if key in peft_config:
                values[key] = peft_config[key]

    args_dict = _as_dict(args)
    if args_dict:
        if "method" in args_dict and str(args_dict["method"]).lower() in ("snoft", "snoft_e"):
            values["method"] = args_dict["method"]
            values["enabled"] = True
        if "use_snoft" in args_dict and args_dict["use_snoft"] is not None:
            values["enabled"] = args_dict["use_snoft"]
        nested = args_dict.get("snoft")
        if isinstance(nested, dict):
            values.update(nested)
        for key in SNOFTConfig.__dataclass_fields__.keys():
            prefixed = f"snoft_{key}"
            if prefixed in args_dict and args_dict[prefixed] is not None:
                values[key] = args_dict[prefixed]

    return SNOFTConfig(**values)


def _resolve_target_layers(target_layers: Any, num_layers: int) -> List[int]:
    if target_layers is None or target_layers == "all":
        return list(range(num_layers))
    if isinstance(target_layers, str):
        if target_layers.lower() == "all":
            return list(range(num_layers))
        return [int(x.strip()) for x in target_layers.split(",") if x.strip()]
    return [int(x) for x in target_layers]


def _get_mamba_blocks(model: nn.Module) -> List[nn.Module]:
    if hasattr(model, "get_mamba_blocks"):
        return list(model.get_mamba_blocks())
    if hasattr(model, "model") and hasattr(model.model, "get_mamba_blocks"):
        return list(model.model.get_mamba_blocks())
    return [module for module in model.modules() if hasattr(module, "d_inner") and hasattr(module, "x_proj")]


def inject_snoft_adapters(model: nn.Module, config: SNOFTConfig) -> List[int]:
    blocks = _get_mamba_blocks(model)
    target_layers = _resolve_target_layers(config.target_layers, len(blocks))
    for layer_idx in target_layers:
        block = blocks[layer_idx]
        ref = block.conv1d.weight if hasattr(block, "conv1d") else next(block.parameters())
        block.snoft = SNOFTLayer(
            d_inner=block.d_inner,
            num_groups=config.num_groups,
            chunk_size=config.chunk_size,
            router_rank=config.router_rank,
            tau_logit_init=config.tau_logit_init,
            eps=config.eps,
        ).to(device=ref.device, dtype=ref.dtype)
        block.snoft_enabled = True
        block.snoft_sanity_check = config.sanity_check
        block.snoft_sanity_checked = False
        block.use_fast_path = False

    setattr(model, "use_snoft", True)
    setattr(model, "snoft_config", config.to_dict())
    setattr(model, "snoft_target_layers", target_layers)
    return target_layers


def iter_snoft_layers(model: nn.Module) -> Iterable[SNOFTLayer]:
    for module in model.modules():
        if isinstance(module, SNOFTLayer):
            yield module


def has_snoft_adapters(model: nn.Module) -> bool:
    return any(True for _ in iter_snoft_layers(model))


def _is_small_classifier_param(name: str) -> bool:
    lowered = name.lower()
    if "lm_head" in lowered or "embedding" in lowered:
        return False
    return any(token in lowered for token in ("classifier", "classification_head", "score", "predictor"))


def mark_only_snoft_as_trainable(model: nn.Module, train_task_head: bool = True) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        lowered = name.lower()
        if "snoft" in lowered:
            param.requires_grad = True
        elif train_task_head and _is_small_classifier_param(name):
            param.requires_grad = True

    freeze_lm_head_weight_for_snoft(model)


def freeze_lm_head_weight_for_snoft(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("lm_head.weight") and param.requires_grad:
            print("[SNOFT-E][warning] lm_head.weight was trainable; freezing it to avoid full LM-head tuning.")
            param.requires_grad = False


def count_snoft_parameters(model: nn.Module) -> int:
    return sum(param.numel() for layer in iter_snoft_layers(model) for param in layer.parameters())


def print_snoft_summary(model: nn.Module, config: SNOFTConfig, target_layers: List[int]) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params > 0 else 0.0
    snoft_params = count_snoft_parameters(model)
    lm_head_trainable = any(
        name.endswith("lm_head.weight") and param.requires_grad
        for name, param in model.named_parameters()
    )

    print("SNOFT-E configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print(f"  target_layers_resolved: {target_layers}")
    print(f"  adapter_count: {len(list(iter_snoft_layers(model)))}")
    print(f"  snoft parameters: {snoft_params:,}")
    print(f"  lm_head_trainable: {lm_head_trainable}")
    print("trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name}: shape={tuple(param.shape)}, params={param.numel():,}")
    print(f"trainable params count: {trainable_params:,}")
    print(f"trainable params ratio: {ratio:.6%}")
    if ratio > 0.003:
        print("[SNOFT-E][warning] trainable ratio exceeds 0.3%; check task head and frozen backbone settings.")
