from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SDFTConfig:
    use_sdft: bool = False
    sdft_rank: int = 4
    sdft_rho_init: float = 0.05
    sdft_gate_mode: str = "none"
    sdft_dropout: float = 0.05
    sdft_target_layers: Any = "all"
    sdft_freeze_base_model: bool = True
    sdft_train_classifier: bool = True
    sdft_log_stats: bool = True
    sdft_log_interval: Optional[int] = None
    sdft_log_per_layer: bool = False
    sdft_log_grad: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "SDFTConfig":
        cfg = dict(cfg or {})
        if str(cfg.get("method", "")).lower() == "sdft":
            cfg["use_sdft"] = True

        allowed = set(cls.__dataclass_fields__.keys())
        values = {key: cfg[key] for key in allowed if key in cfg}
        out = cls(**values)
        out.use_sdft = bool(out.use_sdft)
        out.sdft_rank = int(out.sdft_rank)
        out.sdft_rho_init = float(out.sdft_rho_init)
        out.sdft_gate_mode = str(out.sdft_gate_mode).lower()
        out.sdft_dropout = float(out.sdft_dropout)
        out.sdft_freeze_base_model = bool(out.sdft_freeze_base_model)
        out.sdft_train_classifier = bool(out.sdft_train_classifier)
        out.sdft_log_stats = bool(out.sdft_log_stats)
        out.sdft_log_interval = None if out.sdft_log_interval is None else int(out.sdft_log_interval)
        out.sdft_log_per_layer = bool(out.sdft_log_per_layer)
        out.sdft_log_grad = bool(out.sdft_log_grad)

        if out.sdft_rank <= 0:
            raise ValueError("sdft_rank must be positive.")
        if out.sdft_gate_mode not in ("none", "z"):
            raise ValueError(f"Unsupported sdft_gate_mode: {out.sdft_gate_mode}")
        if out.sdft_dropout < 0:
            raise ValueError("sdft_dropout must be non-negative.")
        if out.sdft_log_interval is not None and out.sdft_log_interval <= 0:
            raise ValueError("sdft_log_interval must be positive when set.")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


_SOT_KEYS = (
    "bias_type",
    "bias_init",
    "use_sft",
    "use_tf_sot",
    "use_cf_sot",
    "C_scale_shape",
    "C_bias_shape",
)


def is_sdft_config_dict(cfg: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get("use_sdft", False)) or str(cfg.get("method", "")).lower() == "sdft"


def validate_sdft_config_dict(cfg: Optional[Dict[str, Any]]) -> None:
    if not is_sdft_config_dict(cfg):
        return
    if str((cfg or {}).get("peft_type", "")).upper() == "SUFFIX_TUNING":
        raise ValueError("SDFT must not use the SUFFIX_TUNING PEFT path.")
    forbidden = [key for key in _SOT_KEYS if cfg.get(key) not in (None, False)]
    if forbidden:
        raise ValueError(f"SDFT config must not enable SOT/suffix fields: {forbidden}")


def merge_sdft_config(peft_cfg: Optional[Dict[str, Any]], overrides: Optional[Any] = None) -> SDFTConfig:
    cfg = dict(peft_cfg or {})
    if overrides is not None:
        for key in SDFTConfig.__dataclass_fields__.keys():
            if hasattr(overrides, key):
                value = getattr(overrides, key)
                if value is not None:
                    cfg[key] = value
        if str(getattr(overrides, "method", "")).lower() == "sdft":
            cfg["use_sdft"] = True
    validate_sdft_config_dict(cfg)
    return SDFTConfig.from_dict(cfg)


class SDFTDriverAdapter(nn.Module):
    def __init__(
        self,
        d_inner: int,
        rank: int = 4,
        rho_init: float = 0.05,
        gate_mode: str = "none",
        dropout: float = 0.05,
        layer_idx: Optional[int] = None,
        stats_enabled: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        gate_mode = str(gate_mode).lower()
        if gate_mode not in ("none", "z"):
            raise ValueError(f"Unsupported SDFT gate_mode: {gate_mode}")

        self.d_inner = int(d_inner)
        self.rank = int(rank)
        self.layer_idx = layer_idx
        self.gate_mode = gate_mode
        self.stats_enabled = bool(stats_enabled)
        self.eps = 1e-8

        self.ln = nn.LayerNorm(self.d_inner, device=device, dtype=dtype)
        self.down = nn.Linear(self.d_inner, self.rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(self.rank, self.d_inner, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(float(dropout))
        self.rho = nn.Parameter(torch.tensor(float(rho_init), device=device, dtype=dtype))

        if self.gate_mode == "z":
            self.gate_scale = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
            self.gate_bias = nn.Parameter(torch.zeros(self.d_inner, device=device, dtype=dtype))

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-4)

        self._stats_accumulator = defaultdict(list)

    def _push_stat(self, name: str, value: float) -> None:
        self._stats_accumulator[name].append(float(value))

    def _as_bld(self, x: torch.Tensor, layout: Optional[str] = None) -> tuple[torch.Tensor, str]:
        if layout is not None:
            layout = layout.lower()
            if layout == "bd":
                if x.ndim != 2 or x.shape[-1] != self.d_inner:
                    raise RuntimeError(f"SDFT expected [B,D] with D={self.d_inner}, got {tuple(x.shape)}")
                return x.unsqueeze(1), "bd"
            if layout == "bld":
                if x.ndim != 3 or x.shape[-1] != self.d_inner:
                    raise RuntimeError(f"SDFT expected [B,L,D] with D={self.d_inner}, got {tuple(x.shape)}")
                return x, "bld"
            if layout == "bdl":
                if x.ndim != 3 or x.shape[1] != self.d_inner:
                    raise RuntimeError(f"SDFT expected [B,D,L] with D={self.d_inner}, got {tuple(x.shape)}")
                return x.transpose(1, 2), "bdl"
            raise RuntimeError(f"Unknown SDFT layout hint: {layout}")

        if x.ndim == 2:
            if x.shape[-1] != self.d_inner:
                raise RuntimeError(f"SDFT expected last dim {self.d_inner}, got {tuple(x.shape)}")
            return x.unsqueeze(1), "bd"
        if x.ndim != 3:
            raise RuntimeError(f"SDFT expects [B,D,L], [B,L,D], or [B,D], got {tuple(x.shape)}")
        if x.shape[-1] == self.d_inner:
            return x, "bld"
        if x.shape[1] == self.d_inner:
            return x.transpose(1, 2), "bdl"
        raise RuntimeError(f"SDFT cannot infer channel dimension from shape {tuple(x.shape)}")

    @staticmethod
    def _restore_layout(x: torch.Tensor, layout: str) -> torch.Tensor:
        if layout == "bld":
            return x
        if layout == "bdl":
            return x.transpose(1, 2)
        if layout == "bd":
            return x.squeeze(1)
        raise RuntimeError(f"Unknown SDFT layout: {layout}")

    def _compute_gate(self, z: Optional[torch.Tensor], target_layout: str) -> Optional[torch.Tensor]:
        if self.gate_mode == "none":
            return None
        if z is None:
            raise RuntimeError("SDFT gate_mode='z' requires z branch input.")
        try:
            z_bld, _ = self._as_bld(z, target_layout)
        except RuntimeError:
            if target_layout == "bd" and z.ndim == 3:
                z_bld, _ = self._as_bld(z)
            elif target_layout != "bd":
                z_bld, _ = self._as_bld(z)
            else:
                raise
        gate = torch.sigmoid(
            self.gate_scale.to(dtype=z_bld.dtype, device=z_bld.device) * F.silu(z_bld)
            + self.gate_bias.to(dtype=z_bld.dtype, device=z_bld.device).view(1, 1, -1)
        )
        if target_layout == "bd" and gate.shape[1] != 1:
            gate = gate[:, -1:, :]
        return gate

    def _collect_forward_stats(self, v_bld: torch.Tensor, delta_bld: torch.Tensor, gate: Optional[torch.Tensor]) -> None:
        if not self.stats_enabled:
            return
        with torch.no_grad():
            v_f = v_bld.detach().float()
            d_f = delta_bld.detach().float()
            v_rms = torch.sqrt(torch.mean(v_f ** 2)).item()
            d_rms = torch.sqrt(torch.mean(d_f ** 2)).item()
            v_abs = torch.mean(torch.abs(v_f)).item()
            d_abs = torch.mean(torch.abs(d_f)).item()

            self._push_stat("v_rms", v_rms)
            self._push_stat("v_mean", torch.mean(v_f).item())
            self._push_stat("v_std", torch.std(v_f, unbiased=False).item())
            self._push_stat("v_abs_mean", v_abs)
            self._push_stat("delta_rms", d_rms)
            self._push_stat("delta_mean", torch.mean(d_f).item())
            self._push_stat("delta_std", torch.std(d_f, unbiased=False).item())
            self._push_stat("delta_abs_mean", d_abs)
            self._push_stat("delta_to_v_rms_ratio", d_rms / (v_rms + self.eps))
            self._push_stat("delta_to_v_abs_ratio", d_abs / (v_abs + self.eps))
            self._push_stat("rho", self.rho.detach().float().item())

            if self.gate_mode == "z" and gate is not None:
                g_f = gate.detach().float()
                self._push_stat("gate_mean", torch.mean(g_f).item())
                self._push_stat("gate_std", torch.std(g_f, unbiased=False).item())
                self._push_stat("gate_min", torch.min(g_f).item())
                self._push_stat("gate_max", torch.max(g_f).item())
                self._push_stat("gate_saturation_low", torch.mean((g_f < 0.05).float()).item())
                self._push_stat("gate_saturation_high", torch.mean((g_f > 0.95).float()).item())
                self._push_stat("gate_scale", self.gate_scale.detach().float().item())
                bias_f = self.gate_bias.detach().float()
                self._push_stat("gate_bias_mean", torch.mean(bias_f).item())
                self._push_stat("gate_bias_std", torch.std(bias_f, unbiased=False).item())

    def forward(self, v: torch.Tensor, z: Optional[torch.Tensor] = None, layout: Optional[str] = None) -> torch.Tensor:
        v_bld, layout = self._as_bld(v, layout)
        hidden = self.down(self.ln(v_bld))
        hidden = self.dropout(F.silu(hidden))
        update = self.up(hidden)

        gate = self._compute_gate(z, layout)
        if gate is None:
            delta = self.rho.to(dtype=v_bld.dtype, device=v_bld.device) * update
        else:
            delta = self.rho.to(dtype=v_bld.dtype, device=v_bld.device) * gate.to(dtype=v_bld.dtype, device=v_bld.device) * update
        delta = delta.to(dtype=v_bld.dtype, device=v_bld.device)
        self._collect_forward_stats(v_bld, delta, gate)
        return self._restore_layout(v_bld + delta, layout)

    def peek_stats(self) -> Dict[str, float]:
        return {
            key: sum(values) / len(values)
            for key, values in self._stats_accumulator.items()
            if len(values) > 0
        }

    def pop_stats(self) -> Dict[str, float]:
        stats = self.peek_stats()
        self.clear_stats()
        return stats

    def clear_stats(self) -> None:
        self._stats_accumulator.clear()


def _find_mamba_blocks(model: nn.Module) -> List[nn.Module]:
    candidate = model
    while hasattr(candidate, "module"):
        candidate = candidate.module

    for obj in (candidate, getattr(candidate, "model", None), getattr(candidate, "base_model", None)):
        if obj is not None and hasattr(obj, "get_mamba_blocks"):
            return list(obj.get_mamba_blocks())

    blocks = []
    for module in candidate.modules():
        if all(hasattr(module, attr) for attr in ("d_inner", "conv1d", "x_proj", "dt_proj")):
            blocks.append(module)
    return blocks


def _resolve_target_layers(target_layers: Any, num_layers: int) -> List[int]:
    if target_layers is None or target_layers == "all":
        return list(range(num_layers))
    if isinstance(target_layers, str):
        value = target_layers.strip().lower()
        if value == "all":
            return list(range(num_layers))
        raw = [item.strip() for item in value.split(",") if item.strip()]
        layers = [int(item) for item in raw]
    elif isinstance(target_layers, Iterable):
        layers = [int(item) for item in target_layers]
    else:
        layers = [int(target_layers)]

    invalid = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Invalid SDFT target layer(s) {invalid}; model has {num_layers} layers.")
    return sorted(set(layers))


def inject_sdft_adapters(model: nn.Module, config: SDFTConfig | Dict[str, Any] | SimpleNamespace) -> List[int]:
    if not isinstance(config, SDFTConfig):
        if isinstance(config, SimpleNamespace):
            config = merge_sdft_config(None, config)
        else:
            config = SDFTConfig.from_dict(dict(config))
    if not config.use_sdft:
        return []

    blocks = _find_mamba_blocks(model)
    if not blocks:
        raise RuntimeError("Could not find Mamba blocks for SDFT injection.")

    target_layers = _resolve_target_layers(config.sdft_target_layers, len(blocks))
    for layer_idx in target_layers:
        block = blocks[layer_idx]
        ref = block.conv1d.weight if hasattr(block, "conv1d") else next(block.parameters())
        block.sdft_adapter = SDFTDriverAdapter(
            d_inner=block.d_inner,
            rank=config.sdft_rank,
            rho_init=config.sdft_rho_init,
            gate_mode=config.sdft_gate_mode,
            dropout=config.sdft_dropout,
            layer_idx=layer_idx,
            stats_enabled=config.sdft_log_stats,
            device=ref.device,
            dtype=ref.dtype,
        )
    setattr(model, "use_sdft", True)
    setattr(model, "sdft_config", config.to_dict())
    setattr(model, "sdft_target_layers", target_layers)
    return target_layers


def iter_sdft_adapters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, SDFTDriverAdapter):
            yield module


def has_sdft_adapters(model: nn.Module) -> bool:
    return any(True for _ in iter_sdft_adapters(model))


def _is_small_classifier_param(name: str) -> bool:
    lowered = name.lower()
    if "lm_head" in lowered or "embedding" in lowered:
        return False
    return any(token in lowered for token in ("classifier", "classification_head", "score", "predictor"))


def freeze_for_sdft(model: nn.Module, train_classifier: bool = True) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        lowered = name.lower()
        if "sdft" in lowered:
            param.requires_grad = True
        elif train_classifier and _is_small_classifier_param(name):
            param.requires_grad = True

    for name, param in model.named_parameters():
        if name.endswith("lm_head.weight") and param.requires_grad:
            print("[SDFT][warning] lm_head.weight was trainable; freezing it to avoid full LM-head tuning.")
            param.requires_grad = False


def freeze_lm_head_weight_for_sdft(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("lm_head.weight") and param.requires_grad:
            print("[SDFT][warning] lm_head.weight was trainable; freezing it to avoid full LM-head tuning.")
            param.requires_grad = False


def collect_sdft_stats(model: nn.Module, clear: bool = True, log_per_layer: bool = True) -> Dict[str, float]:
    adapters = list(iter_sdft_adapters(model))
    if not adapters:
        return {}

    logs: Dict[str, float] = {}
    ratio_values = []
    gate_values = []
    rho_values = []

    for idx, adapter in enumerate(adapters):
        layer_idx = adapter.layer_idx if adapter.layer_idx is not None else idx
        stats = adapter.pop_stats() if clear else adapter.peek_stats()
        if log_per_layer:
            for key, value in stats.items():
                logs[f"sdft/layer_{layer_idx}/{key}"] = float(value)
        if "delta_to_v_rms_ratio" in stats:
            ratio_values.append(float(stats["delta_to_v_rms_ratio"]))
        if "gate_mean" in stats:
            gate_values.append(float(stats["gate_mean"]))
        rho_values.append(float(adapter.rho.detach().float().cpu()))

    if ratio_values:
        logs["sdft/global/mean_delta_to_v_rms_ratio"] = sum(ratio_values) / len(ratio_values)
        logs["sdft/global/max_delta_to_v_rms_ratio"] = max(ratio_values)
        logs["sdft/global/min_delta_to_v_rms_ratio"] = min(ratio_values)
    if gate_values:
        logs["sdft/global/mean_gate"] = sum(gate_values) / len(gate_values)
    if rho_values:
        logs["sdft/global/mean_rho"] = sum(rho_values) / len(rho_values)
    logs["sdft/global/trainable_param_count"] = float(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logs["sdft/global/adapter_count"] = float(len(adapters))
    return logs


def _grad_norm(param: Optional[torch.nn.Parameter]) -> Optional[float]:
    if param is None or param.grad is None:
        return None
    return float(param.grad.detach().float().norm().cpu())


def _grad_abs(param: Optional[torch.nn.Parameter]) -> Optional[float]:
    if param is None or param.grad is None:
        return None
    return float(param.grad.detach().float().abs().mean().cpu())


def collect_sdft_grad_stats(model: nn.Module, log_per_layer: bool = True) -> Dict[str, float]:
    adapters = list(iter_sdft_adapters(model))
    if not adapters:
        return {}

    logs: Dict[str, float] = {}
    up_norms = []
    down_norms = []

    for idx, adapter in enumerate(adapters):
        layer_idx = adapter.layer_idx if adapter.layer_idx is not None else idx
        values = {
            "up_grad_norm": _grad_norm(adapter.up.weight),
            "down_grad_norm": _grad_norm(adapter.down.weight),
            "rho_grad_abs": _grad_abs(adapter.rho),
        }
        if getattr(adapter, "gate_mode", "none") == "z":
            values["gate_scale_grad_abs"] = _grad_abs(getattr(adapter, "gate_scale", None))
            values["gate_bias_grad_norm"] = _grad_norm(getattr(adapter, "gate_bias", None))

        for key, value in values.items():
            if value is None:
                continue
            if log_per_layer:
                logs[f"sdft/layer_{layer_idx}/{key}"] = value
            if key == "up_grad_norm":
                up_norms.append(value)
            elif key == "down_grad_norm":
                down_norms.append(value)

    if up_norms:
        logs["sdft/global/mean_up_grad_norm"] = sum(up_norms) / len(up_norms)
    if down_norms:
        logs["sdft/global/mean_down_grad_norm"] = sum(down_norms) / len(down_norms)
    return logs


def print_sdft_summary(model: nn.Module, config: SDFTConfig, target_layers: List[int]) -> None:
    adapters = list(iter_sdft_adapters(model))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params > 0 else 0.0

    print("SDFT configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print(f"  target_layers_resolved: {target_layers}")
    print(f"  adapter_count: {len(adapters)}")
    print("SDFT trainable parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
    print(f"SDFT trainable params: {trainable_params:,} / {total_params:,} ({ratio:.6%})")
