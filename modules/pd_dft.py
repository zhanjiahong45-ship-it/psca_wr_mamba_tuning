from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PDDFTConfig:
    use_pd_dft: bool = False
    pd_dft_rank: int = 4
    pd_dft_dropout: float = 0.05
    pd_dft_rho_param_init: float = 0.05
    pd_dft_rho_scan_init: float = 0.05
    pd_dft_learnable_rho: bool = True
    pd_dft_mode: str = "both"
    pd_dft_target_layers: Any = "all"
    pd_dft_share_down: bool = True
    pd_dft_max_delta_ratio_param: Optional[float] = None
    pd_dft_max_delta_ratio_scan: Optional[float] = None
    pd_dft_log_stats: bool = True
    pd_dft_log_per_layer: bool = False
    pd_dft_log_grad: bool = True
    pd_dft_log_interval: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "PDDFTConfig":
        cfg = dict(cfg or {})
        if str(cfg.get("method", "")).lower() in ("pd_dft", "path_decoupled_dft"):
            cfg["use_pd_dft"] = True

        allowed = set(cls.__dataclass_fields__.keys())
        values = {key: cfg[key] for key in allowed if key in cfg}
        out = cls(**values)
        out.use_pd_dft = bool(out.use_pd_dft)
        out.pd_dft_rank = int(out.pd_dft_rank)
        out.pd_dft_dropout = float(out.pd_dft_dropout)
        out.pd_dft_rho_param_init = float(out.pd_dft_rho_param_init)
        out.pd_dft_rho_scan_init = float(out.pd_dft_rho_scan_init)
        out.pd_dft_learnable_rho = bool(out.pd_dft_learnable_rho)
        out.pd_dft_mode = str(out.pd_dft_mode).lower()
        out.pd_dft_share_down = bool(out.pd_dft_share_down)
        out.pd_dft_log_stats = bool(out.pd_dft_log_stats)
        out.pd_dft_log_per_layer = bool(out.pd_dft_log_per_layer)
        out.pd_dft_log_grad = bool(out.pd_dft_log_grad)
        out.pd_dft_log_interval = None if out.pd_dft_log_interval is None else int(out.pd_dft_log_interval)
        if out.pd_dft_max_delta_ratio_param is not None:
            out.pd_dft_max_delta_ratio_param = float(out.pd_dft_max_delta_ratio_param)
        if out.pd_dft_max_delta_ratio_scan is not None:
            out.pd_dft_max_delta_ratio_scan = float(out.pd_dft_max_delta_ratio_scan)

        if out.pd_dft_rank <= 0:
            raise ValueError("pd_dft_rank must be positive.")
        if out.pd_dft_dropout < 0:
            raise ValueError("pd_dft_dropout must be non-negative.")
        if out.pd_dft_mode not in ("param_only", "scan_only", "both"):
            raise ValueError(f"Unsupported pd_dft_mode: {out.pd_dft_mode}")
        if not out.pd_dft_share_down:
            raise ValueError("The first PD-DFT version requires pd_dft_share_down=True.")
        if out.pd_dft_log_interval is not None and out.pd_dft_log_interval <= 0:
            raise ValueError("pd_dft_log_interval must be positive when set.")
        if out.pd_dft_max_delta_ratio_param is not None and out.pd_dft_max_delta_ratio_param <= 0:
            raise ValueError("pd_dft_max_delta_ratio_param must be positive when set.")
        if out.pd_dft_max_delta_ratio_scan is not None and out.pd_dft_max_delta_ratio_scan <= 0:
            raise ValueError("pd_dft_max_delta_ratio_scan must be positive when set.")
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


def is_pd_dft_config_dict(cfg: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(cfg, dict):
        return False
    method = str(cfg.get("method", "")).lower()
    return bool(cfg.get("use_pd_dft", False)) or method in ("pd_dft", "path_decoupled_dft")


def validate_pd_dft_config_dict(cfg: Optional[Dict[str, Any]]) -> None:
    if not is_pd_dft_config_dict(cfg):
        return
    if str((cfg or {}).get("peft_type", "")).upper() == "SUFFIX_TUNING":
        raise ValueError("PD-DFT must not use the SUFFIX_TUNING PEFT path.")
    forbidden = [key for key in _SOT_KEYS if cfg.get(key) not in (None, False)]
    if forbidden:
        raise ValueError(f"PD-DFT config must not enable SOT/suffix fields: {forbidden}")


def merge_pd_dft_config(peft_cfg: Optional[Dict[str, Any]], overrides: Optional[Any] = None) -> PDDFTConfig:
    cfg = dict(peft_cfg or {})
    if overrides is not None:
        for key in PDDFTConfig.__dataclass_fields__.keys():
            if hasattr(overrides, key):
                value = getattr(overrides, key)
                if value is not None:
                    cfg[key] = value
        if str(getattr(overrides, "method", "")).lower() in ("pd_dft", "path_decoupled_dft"):
            cfg["use_pd_dft"] = True
    validate_pd_dft_config_dict(cfg)
    return PDDFTConfig.from_dict(cfg)


class PathDecoupledDFTAdapter(nn.Module):
    def __init__(
        self,
        d_inner: int,
        rank: int = 4,
        dropout: float = 0.05,
        rho_param_init: float = 0.05,
        rho_scan_init: float = 0.05,
        learnable_rho: bool = True,
        mode: str = "both",
        max_delta_ratio_param: Optional[float] = None,
        max_delta_ratio_scan: Optional[float] = None,
        layer_idx: Optional[int] = None,
        log_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in ("param_only", "scan_only", "both"):
            raise ValueError(f"Unsupported PD-DFT mode: {mode}")

        self.d_inner = int(d_inner)
        self.rank = int(rank)
        self.layer_idx = layer_idx
        self.mode = mode
        self.log_stats = bool(log_stats)
        self.max_delta_ratio_param = max_delta_ratio_param
        self.max_delta_ratio_scan = max_delta_ratio_scan
        self.eps = 1e-8

        self.ln = nn.LayerNorm(self.d_inner, device=device, dtype=dtype)
        self.down = nn.Linear(self.d_inner, self.rank, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(float(dropout))
        self.up_param = nn.Linear(self.rank, self.d_inner, bias=False, device=device, dtype=dtype)
        self.up_scan = nn.Linear(self.rank, self.d_inner, bias=False, device=device, dtype=dtype)

        rho_param = torch.tensor(float(rho_param_init), device=device, dtype=dtype)
        rho_scan = torch.tensor(float(rho_scan_init), device=device, dtype=dtype)
        if learnable_rho:
            self.rho_param = nn.Parameter(rho_param)
            self.rho_scan = nn.Parameter(rho_scan)
        else:
            self.register_buffer("rho_param", rho_param)
            self.register_buffer("rho_scan", rho_scan)

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_param.weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.up_scan.weight, mean=0.0, std=1e-4)

        if not self.param_active:
            self.up_param.requires_grad_(False)
            if isinstance(self.rho_param, nn.Parameter):
                self.rho_param.requires_grad = False
        if not self.scan_active:
            self.up_scan.requires_grad_(False)
            if isinstance(self.rho_scan, nn.Parameter):
                self.rho_scan.requires_grad = False

        self._stats_accumulator = defaultdict(list)

    @property
    def param_active(self) -> bool:
        return self.mode in ("param_only", "both")

    @property
    def scan_active(self) -> bool:
        return self.mode in ("scan_only", "both")

    def _push_stat(self, name: str, value: float) -> None:
        self._stats_accumulator[name].append(float(value))

    def _to_bld(self, x: torch.Tensor, layout: Optional[str] = None) -> tuple[torch.Tensor, str]:
        if layout is not None:
            layout = layout.lower()
            if layout == "bd":
                if x.ndim != 2 or x.shape[-1] != self.d_inner:
                    raise RuntimeError(f"PD-DFT expected [B,D] with D={self.d_inner}, got {tuple(x.shape)}")
                return x.unsqueeze(1), "bd"
            if layout == "bld":
                if x.ndim != 3 or x.shape[-1] != self.d_inner:
                    raise RuntimeError(f"PD-DFT expected [B,L,D] with D={self.d_inner}, got {tuple(x.shape)}")
                return x, "bld"
            if layout == "bdl":
                if x.ndim != 3 or x.shape[1] != self.d_inner:
                    raise RuntimeError(f"PD-DFT expected [B,D,L] with D={self.d_inner}, got {tuple(x.shape)}")
                return x.transpose(1, 2), "bdl"
            raise RuntimeError(f"Unknown PD-DFT layout hint: {layout}")

        if x.ndim == 2:
            if x.shape[-1] != self.d_inner:
                raise RuntimeError(f"PD-DFT expected last dim {self.d_inner}, got {tuple(x.shape)}")
            return x.unsqueeze(1), "bd"
        if x.ndim != 3:
            raise RuntimeError(f"PD-DFT expects [B,D,L], [B,L,D], or [B,D], got {tuple(x.shape)}")
        if x.shape[-1] == self.d_inner:
            return x, "bld"
        if x.shape[1] == self.d_inner:
            return x.transpose(1, 2), "bdl"
        raise RuntimeError(f"PD-DFT cannot infer channel dimension from shape {tuple(x.shape)}")

    @staticmethod
    def _restore_layout(x: torch.Tensor, layout: str) -> torch.Tensor:
        if layout == "bld":
            return x
        if layout == "bdl":
            return x.transpose(1, 2)
        if layout == "bd":
            return x.squeeze(1)
        raise RuntimeError(f"Unknown PD-DFT layout: {layout}")

    def _apply_rms_cap(
        self,
        v_bld: torch.Tensor,
        delta: torch.Tensor,
        max_delta_ratio: Optional[float],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if max_delta_ratio is None:
            return delta, None
        v_rms = torch.sqrt(torch.mean(v_bld.float() ** 2, dim=-1, keepdim=True) + self.eps).to(delta.dtype)
        d_rms = torch.sqrt(torch.mean(delta.float() ** 2, dim=-1, keepdim=True) + self.eps).to(delta.dtype)
        scale = torch.clamp(float(max_delta_ratio) * v_rms / (d_rms + self.eps), max=1.0)
        return delta * scale, scale

    def _collect_forward_stats(
        self,
        v_bld: torch.Tensor,
        delta_param: torch.Tensor,
        delta_scan: torch.Tensor,
        cap_param_scale: Optional[torch.Tensor],
        cap_scan_scale: Optional[torch.Tensor],
    ) -> None:
        if not self.log_stats:
            return
        with torch.no_grad():
            v_f = v_bld.detach().float()
            p_f = delta_param.detach().float()
            s_f = delta_scan.detach().float()
            v_rms = torch.sqrt(torch.mean(v_f ** 2)).item()
            p_rms = torch.sqrt(torch.mean(p_f ** 2)).item()
            s_rms = torch.sqrt(torch.mean(s_f ** 2)).item()

            v_token_rms = torch.sqrt(torch.mean(v_f ** 2, dim=-1) + self.eps)
            p_token_rms = torch.sqrt(torch.mean(p_f ** 2, dim=-1) + self.eps)
            s_token_rms = torch.sqrt(torch.mean(s_f ** 2, dim=-1) + self.eps)

            self._push_stat("delta_param_to_v_ratio", p_rms / (v_rms + self.eps))
            self._push_stat("delta_scan_to_v_ratio", s_rms / (v_rms + self.eps))
            self._push_stat("delta_param_to_v_max", torch.max(p_token_rms / (v_token_rms + self.eps)).item())
            self._push_stat("delta_scan_to_v_max", torch.max(s_token_rms / (v_token_rms + self.eps)).item())
            self._push_stat("rho_param", self.rho_param.detach().float().item())
            self._push_stat("rho_scan", self.rho_scan.detach().float().item())
            if cap_param_scale is not None:
                self._push_stat("cap_param_active_rate", torch.mean((cap_param_scale.detach().float() < 1.0).float()).item())
            if cap_scan_scale is not None:
                self._push_stat("cap_scan_active_rate", torch.mean((cap_scan_scale.detach().float() < 1.0).float()).item())

    def forward(self, v: torch.Tensor, layout: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
        v_bld, layout = self._to_bld(v, layout)
        q = self.dropout(F.silu(self.down(self.ln(v_bld))))

        if self.param_active:
            rho = self.rho_param.to(dtype=v_bld.dtype, device=v_bld.device)
            delta_param = rho * self.up_param(q)
            delta_param, cap_param_scale = self._apply_rms_cap(v_bld, delta_param, self.max_delta_ratio_param)
        else:
            delta_param = torch.zeros_like(v_bld)
            cap_param_scale = None

        if self.scan_active:
            rho = self.rho_scan.to(dtype=v_bld.dtype, device=v_bld.device)
            delta_scan = rho * self.up_scan(q)
            delta_scan, cap_scan_scale = self._apply_rms_cap(v_bld, delta_scan, self.max_delta_ratio_scan)
        else:
            delta_scan = torch.zeros_like(v_bld)
            cap_scan_scale = None

        delta_param = delta_param.to(dtype=v_bld.dtype, device=v_bld.device)
        delta_scan = delta_scan.to(dtype=v_bld.dtype, device=v_bld.device)
        self._collect_forward_stats(v_bld, delta_param, delta_scan, cap_param_scale, cap_scan_scale)
        return (
            self._restore_layout(v_bld + delta_param, layout),
            self._restore_layout(v_bld + delta_scan, layout),
        )

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
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    elif isinstance(target_layers, Iterable):
        values = list(target_layers)
        if len(values) == 1 and str(values[0]).lower() == "all":
            return list(range(num_layers))
        layers = [int(item) for item in values]
    else:
        layers = [int(target_layers)]

    invalid = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Invalid PD-DFT target layer(s) {invalid}; model has {num_layers} layers.")
    return sorted(set(layers))


def inject_pd_dft_adapters(model: nn.Module, config: PDDFTConfig | Dict[str, Any] | SimpleNamespace) -> List[int]:
    if not isinstance(config, PDDFTConfig):
        if isinstance(config, SimpleNamespace):
            config = merge_pd_dft_config(None, config)
        else:
            config = PDDFTConfig.from_dict(dict(config))
    if not config.use_pd_dft:
        return []

    blocks = _find_mamba_blocks(model)
    if not blocks:
        raise RuntimeError("Could not find Mamba blocks for PD-DFT injection.")

    target_layers = _resolve_target_layers(config.pd_dft_target_layers, len(blocks))
    for layer_idx in target_layers:
        block = blocks[layer_idx]
        ref = block.conv1d.weight if hasattr(block, "conv1d") else next(block.parameters())
        block.pd_dft_adapter = PathDecoupledDFTAdapter(
            d_inner=block.d_inner,
            rank=config.pd_dft_rank,
            dropout=config.pd_dft_dropout,
            rho_param_init=config.pd_dft_rho_param_init,
            rho_scan_init=config.pd_dft_rho_scan_init,
            learnable_rho=config.pd_dft_learnable_rho,
            mode=config.pd_dft_mode,
            max_delta_ratio_param=config.pd_dft_max_delta_ratio_param,
            max_delta_ratio_scan=config.pd_dft_max_delta_ratio_scan,
            layer_idx=layer_idx,
            log_stats=config.pd_dft_log_stats,
            device=ref.device,
            dtype=ref.dtype,
        )
    setattr(model, "use_pd_dft", True)
    setattr(model, "pd_dft_config", config.to_dict())
    setattr(model, "pd_dft_target_layers", target_layers)
    return target_layers


def iter_pd_dft_adapters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, PathDecoupledDFTAdapter):
            yield module


def has_pd_dft_adapters(model: nn.Module) -> bool:
    return any(True for _ in iter_pd_dft_adapters(model))


def _is_small_classifier_param(name: str) -> bool:
    lowered = name.lower()
    if "lm_head" in lowered or "embedding" in lowered:
        return False
    return any(token in lowered for token in ("classifier", "classification_head", "score", "predictor"))


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad = value


def mark_only_pd_dft_as_trainable(model: nn.Module, train_classifier: bool = True) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for adapter in iter_pd_dft_adapters(model):
        _set_requires_grad(adapter.ln, True)
        _set_requires_grad(adapter.down, True)
        _set_requires_grad(adapter.up_param, adapter.param_active)
        _set_requires_grad(adapter.up_scan, adapter.scan_active)
        if isinstance(adapter.rho_param, nn.Parameter):
            adapter.rho_param.requires_grad = adapter.param_active
        if isinstance(adapter.rho_scan, nn.Parameter):
            adapter.rho_scan.requires_grad = adapter.scan_active

    if train_classifier:
        for name, param in model.named_parameters():
            if _is_small_classifier_param(name):
                param.requires_grad = True

    freeze_lm_head_weight_for_pd_dft(model)


def freeze_lm_head_weight_for_pd_dft(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("lm_head.weight") and param.requires_grad:
            print("[PD-DFT][warning] lm_head.weight was trainable; freezing it to avoid full LM-head tuning.")
            param.requires_grad = False


def collect_pd_dft_stats(model: nn.Module, clear: bool = True, log_per_layer: bool = False) -> Dict[str, float]:
    adapters = list(iter_pd_dft_adapters(model))
    if not adapters:
        return {}

    logs: Dict[str, float] = {}
    param_ratios = []
    scan_ratios = []
    param_maxes = []
    scan_maxes = []
    rho_params = []
    rho_scans = []
    cap_param_rates = []
    cap_scan_rates = []

    for idx, adapter in enumerate(adapters):
        layer_idx = adapter.layer_idx if adapter.layer_idx is not None else idx
        stats = adapter.pop_stats() if clear else adapter.peek_stats()
        if log_per_layer:
            for key, value in stats.items():
                logs[f"pd_dft/layer_{layer_idx}/{key}"] = float(value)
        if adapter.param_active:
            if "delta_param_to_v_ratio" in stats:
                param_ratios.append(float(stats["delta_param_to_v_ratio"]))
            if "delta_param_to_v_max" in stats:
                param_maxes.append(float(stats["delta_param_to_v_max"]))
            rho_params.append(float(adapter.rho_param.detach().float().cpu()))
            if "cap_param_active_rate" in stats:
                cap_param_rates.append(float(stats["cap_param_active_rate"]))
        if adapter.scan_active:
            if "delta_scan_to_v_ratio" in stats:
                scan_ratios.append(float(stats["delta_scan_to_v_ratio"]))
            if "delta_scan_to_v_max" in stats:
                scan_maxes.append(float(stats["delta_scan_to_v_max"]))
            rho_scans.append(float(adapter.rho_scan.detach().float().cpu()))
            if "cap_scan_active_rate" in stats:
                cap_scan_rates.append(float(stats["cap_scan_active_rate"]))

    if param_ratios:
        logs["pd_dft/global/mean_delta_param_to_v_ratio"] = sum(param_ratios) / len(param_ratios)
        logs["pd_dft/global/max_delta_param_to_v_ratio"] = max(param_maxes) if param_maxes else max(param_ratios)
    else:
        logs["pd_dft/global/mean_delta_param_to_v_ratio"] = 0.0
        logs["pd_dft/global/max_delta_param_to_v_ratio"] = 0.0
    if scan_ratios:
        logs["pd_dft/global/mean_delta_scan_to_v_ratio"] = sum(scan_ratios) / len(scan_ratios)
        logs["pd_dft/global/max_delta_scan_to_v_ratio"] = max(scan_maxes) if scan_maxes else max(scan_ratios)
    else:
        logs["pd_dft/global/mean_delta_scan_to_v_ratio"] = 0.0
        logs["pd_dft/global/max_delta_scan_to_v_ratio"] = 0.0
    if rho_params:
        logs["pd_dft/global/mean_rho_param"] = sum(rho_params) / len(rho_params)
    if rho_scans:
        logs["pd_dft/global/mean_rho_scan"] = sum(rho_scans) / len(rho_scans)
    if cap_param_rates:
        logs["pd_dft/global/mean_cap_param_active_rate"] = sum(cap_param_rates) / len(cap_param_rates)
    if cap_scan_rates:
        logs["pd_dft/global/mean_cap_scan_active_rate"] = sum(cap_scan_rates) / len(cap_scan_rates)
    logs["pd_dft/global/trainable_param_count"] = float(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logs["pd_dft/global/adapter_count"] = float(len(adapters))
    return logs


def _grad_norm(param: Optional[torch.nn.Parameter]) -> Optional[float]:
    if param is None or param.grad is None:
        return None
    return float(param.grad.detach().float().norm().cpu())


def _grad_abs(param: Optional[torch.nn.Parameter]) -> Optional[float]:
    if param is None or param.grad is None:
        return None
    return float(param.grad.detach().float().abs().mean().cpu())


def collect_pd_dft_grad_stats(model: nn.Module, log_per_layer: bool = False) -> Dict[str, float]:
    adapters = list(iter_pd_dft_adapters(model))
    if not adapters:
        return {}

    logs: Dict[str, float] = {}
    up_param_norms = []
    up_scan_norms = []
    down_norms = []

    for idx, adapter in enumerate(adapters):
        layer_idx = adapter.layer_idx if adapter.layer_idx is not None else idx
        values = {
            "down_grad_norm": _grad_norm(adapter.down.weight),
            "up_param_grad_norm": _grad_norm(adapter.up_param.weight) if adapter.param_active else None,
            "up_scan_grad_norm": _grad_norm(adapter.up_scan.weight) if adapter.scan_active else None,
            "rho_param_grad_abs": _grad_abs(adapter.rho_param) if adapter.param_active and isinstance(adapter.rho_param, nn.Parameter) else None,
            "rho_scan_grad_abs": _grad_abs(adapter.rho_scan) if adapter.scan_active and isinstance(adapter.rho_scan, nn.Parameter) else None,
        }
        for key, value in values.items():
            if value is None:
                continue
            if log_per_layer:
                logs[f"pd_dft/layer_{layer_idx}/{key}"] = value
            if key == "up_param_grad_norm":
                up_param_norms.append(value)
            elif key == "up_scan_grad_norm":
                up_scan_norms.append(value)
            elif key == "down_grad_norm":
                down_norms.append(value)

    if up_param_norms:
        logs["pd_dft/global/mean_up_param_grad_norm"] = sum(up_param_norms) / len(up_param_norms)
    if up_scan_norms:
        logs["pd_dft/global/mean_up_scan_grad_norm"] = sum(up_scan_norms) / len(up_scan_norms)
    if down_norms:
        logs["pd_dft/global/mean_down_grad_norm"] = sum(down_norms) / len(down_norms)
    return logs


def print_pd_dft_summary(model: nn.Module, config: PDDFTConfig, target_layers: List[int]) -> None:
    adapters = list(iter_pd_dft_adapters(model))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params > 0 else 0.0
    classifier_trainable = any(
        param.requires_grad and _is_small_classifier_param(name)
        for name, param in model.named_parameters()
    )
    lm_head_trainable = any(
        name.endswith("lm_head.weight") and param.requires_grad
        for name, param in model.named_parameters()
    )

    print("PD-DFT enabled: True")
    print(f"  mode: {config.pd_dft_mode}")
    print(f"  rank: {config.pd_dft_rank}")
    print(f"  target_layers: {target_layers}")
    print(f"  adapter_count: {len(adapters)}")
    print(f"  trainable_param_count: {trainable_params:,}")
    print(f"  trainable_param_percent: {ratio:.6%}")
    print(f"  classifier_trainable: {classifier_trainable}")
    print(f"  lm_head_trainable: {lm_head_trainable}")
