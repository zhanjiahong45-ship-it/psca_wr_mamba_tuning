import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
    return bool(value)


@dataclass
class PSCAWRConfig:
    use_psca_wr: bool = False
    psca_rank: int = 8
    psca_alpha: float = 1.0
    psca_dropout: float = 0.0
    psca_target_modules: Any = "all"
    psca_init_zero: bool = True
    psca_adapt_b: bool = True
    psca_adapt_c: bool = True
    psca_use_projector_shift: bool = True
    psca_projector_residual: bool = True
    psca_projector_scale: float = 0.01
    psca_fallback_lite: bool = False
    psca_random_gate: bool = False
    psca_independent_gate: bool = False
    psca_debug: bool = False

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "PSCAWRConfig":
        cfg = dict(cfg or {})
        if isinstance(cfg.get("psca"), dict):
            nested = dict(cfg["psca"])
            nested.update({key: value for key, value in cfg.items() if key != "psca"})
            cfg = nested

        method = str(cfg.get("method", "")).lower()
        if method in ("psca_wr", "psca-wr", "psca_lite", "psca-lite"):
            cfg["use_psca_wr"] = True
        if method in ("psca_lite", "psca-lite"):
            cfg["psca_fallback_lite"] = True

        allowed = set(cls.__dataclass_fields__.keys())
        values = {key: cfg[key] for key in allowed if key in cfg}
        out = cls(**values)
        out.use_psca_wr = _as_bool(out.use_psca_wr)
        out.psca_rank = int(out.psca_rank)
        out.psca_alpha = float(out.psca_alpha)
        out.psca_dropout = float(out.psca_dropout)
        out.psca_init_zero = _as_bool(out.psca_init_zero)
        out.psca_adapt_b = _as_bool(out.psca_adapt_b)
        out.psca_adapt_c = _as_bool(out.psca_adapt_c)
        out.psca_use_projector_shift = _as_bool(out.psca_use_projector_shift)
        out.psca_projector_residual = _as_bool(out.psca_projector_residual)
        out.psca_projector_scale = float(out.psca_projector_scale)
        out.psca_fallback_lite = _as_bool(out.psca_fallback_lite)
        out.psca_random_gate = _as_bool(out.psca_random_gate)
        out.psca_independent_gate = _as_bool(out.psca_independent_gate)
        out.psca_debug = _as_bool(out.psca_debug)

        if out.psca_rank <= 0:
            raise ValueError("psca_rank must be positive.")
        if out.psca_dropout < 0:
            raise ValueError("psca_dropout must be non-negative.")
        if out.psca_projector_scale < 0:
            raise ValueError("psca_projector_scale must be non-negative.")
        if not out.psca_adapt_b and not out.psca_adapt_c and not out.psca_fallback_lite:
            raise ValueError("Full PSCA-WR needs at least one of psca_adapt_b or psca_adapt_c.")
        if out.psca_random_gate and out.psca_independent_gate:
            raise ValueError("psca_random_gate and psca_independent_gate are mutually exclusive.")
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


def is_psca_wr_config_dict(cfg: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(cfg, dict):
        return False
    if isinstance(cfg.get("psca"), dict) and is_psca_wr_config_dict(cfg.get("psca")):
        return True
    method = str(cfg.get("method", "")).lower()
    return bool(cfg.get("use_psca_wr", False)) or method in ("psca_wr", "psca-wr", "psca_lite", "psca-lite")


def validate_psca_wr_config_dict(cfg: Optional[Dict[str, Any]]) -> None:
    if not is_psca_wr_config_dict(cfg):
        return
    if str((cfg or {}).get("peft_type", "")).upper() == "SUFFIX_TUNING":
        raise ValueError("PSCA-WR must not use the SUFFIX_TUNING/SOT PEFT path.")
    forbidden = [key for key in _SOT_KEYS if cfg.get(key) not in (None, False)]
    if forbidden:
        raise ValueError(f"PSCA-WR config must not enable SOT/suffix fields: {forbidden}")


def merge_psca_wr_config(peft_cfg: Optional[Dict[str, Any]], overrides: Optional[Any] = None) -> PSCAWRConfig:
    cfg = dict(peft_cfg or {})
    if overrides is not None:
        for key in PSCAWRConfig.__dataclass_fields__.keys():
            if hasattr(overrides, key):
                value = getattr(overrides, key)
                if value is not None:
                    cfg[key] = value
        method = str(getattr(overrides, "method", "")).lower()
        if method in ("psca_wr", "psca-wr", "psca_lite", "psca-lite"):
            cfg["use_psca_wr"] = True
        if method in ("psca_lite", "psca-lite"):
            cfg["psca_fallback_lite"] = True
    validate_psca_wr_config_dict(cfg)
    return PSCAWRConfig.from_dict(cfg)


class PSCAWRAdapter(nn.Module):
    def __init__(
        self,
        d_inner: int,
        d_state: int,
        rank: int = 8,
        alpha_scale: float = 1.0,
        dropout: float = 0.0,
        adapt_b: bool = True,
        adapt_c: bool = True,
        use_projector_shift: bool = True,
        projector_residual: bool = True,
        projector_scale: float = 1e-3,
        fallback_lite: bool = False,
        init_zero: bool = True,
        random_gate: bool = False,
        independent_gate: bool = False,
        layer_idx: Optional[int] = None,
        debug: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_inner = int(d_inner)
        self.d_state = int(d_state)
        self.rank = int(rank)
        self.alpha_scale = float(alpha_scale)
        self.adapt_b = bool(adapt_b)
        self.adapt_c = bool(adapt_c)
        self.use_projector_shift = bool(use_projector_shift)
        self.projector_residual = bool(projector_residual)
        self.projector_scale = float(projector_scale)
        self.fallback_lite = bool(fallback_lite)
        self.random_gate = bool(random_gate)
        self.independent_gate = bool(independent_gate)
        self.layer_idx = layer_idx
        self.debug = bool(debug)
        self.debug_printed = False

        self.psca_proj_down = nn.Linear(self.d_inner, self.rank, bias=False, device=device, dtype=dtype)
        self.psca_proj_up = nn.Linear(self.rank, self.d_inner, bias=False, device=device, dtype=dtype)
        self.psca_down = nn.Linear(self.d_inner, self.rank, bias=False, device=device, dtype=dtype)
        self.psca_up = nn.Linear(self.rank, self.d_inner, bias=False, device=device, dtype=dtype)
        self.psca_B = nn.Linear(self.d_inner, self.d_state, bias=False, device=device, dtype=dtype)
        self.psca_C = nn.Linear(self.d_inner, self.d_state, bias=False, device=device, dtype=dtype)
        self.psca_alpha = nn.Parameter(torch.ones(1, device=device, dtype=dtype) * 1e-3)
        self.psca_lite_vector = nn.Parameter(torch.empty(self.d_inner, device=device, dtype=dtype))
        self.psca_independent_gate = nn.Parameter(torch.zeros(self.d_inner, device=device, dtype=dtype))
        self.dropout = nn.Dropout(float(dropout))

        self.reset_parameters(init_zero=bool(init_zero))

        if not self.adapt_b:
            self.psca_B.requires_grad_(False)
        if not self.adapt_c:
            self.psca_C.requires_grad_(False)

    def reset_parameters(self, init_zero: bool = True) -> None:
        nn.init.kaiming_uniform_(self.psca_proj_down.weight, a=math.sqrt(5))
        if self.psca_proj_down.bias is not None:
            nn.init.zeros_(self.psca_proj_down.bias)
        nn.init.kaiming_uniform_(self.psca_proj_up.weight, a=math.sqrt(5))
        if self.psca_proj_up.bias is not None:
            nn.init.zeros_(self.psca_proj_up.bias)
        nn.init.kaiming_uniform_(self.psca_down.weight, a=math.sqrt(5))
        if self.psca_down.bias is not None:
            nn.init.zeros_(self.psca_down.bias)
        nn.init.kaiming_uniform_(self.psca_up.weight, a=math.sqrt(5))
        if self.psca_up.bias is not None:
            nn.init.zeros_(self.psca_up.bias)
        if init_zero:
            nn.init.zeros_(self.psca_B.weight)
            if self.psca_B.bias is not None:
                nn.init.zeros_(self.psca_B.bias)
            nn.init.zeros_(self.psca_C.weight)
            if self.psca_C.bias is not None:
                nn.init.zeros_(self.psca_C.bias)
            nn.init.zeros_(self.psca_lite_vector)
            self.psca_alpha.data.fill_(1e-3)
        else:
            nn.init.xavier_uniform_(self.psca_B.weight)
            nn.init.xavier_uniform_(self.psca_C.weight)
            nn.init.normal_(self.psca_lite_vector, mean=0.0, std=1e-4)
            self.psca_alpha.data.fill_(1.0)
        nn.init.zeros_(self.psca_independent_gate)

    def compute_delta_u(self, v_bld: torch.Tensor) -> torch.Tensor:
        if v_bld.ndim != 3 or v_bld.shape[-1] != self.d_inner:
            raise RuntimeError(f"PSCA-WR expected projector input [B,L,{self.d_inner}], got {tuple(v_bld.shape)}")

        if self.random_gate or self.independent_gate:
            return torch.zeros_like(v_bld)

        if self.use_projector_shift:
            proj = self.psca_proj_down(v_bld)
            proj = self.dropout(F.silu(proj))
            return self.psca_proj_up(proj)
        return v_bld

    def compute_gate(self, delta_u: torch.Tensor) -> torch.Tensor:
        if delta_u.ndim != 3 or delta_u.shape[-1] != self.d_inner:
            raise RuntimeError(f"PSCA-WR expected delta_u [B,L,{self.d_inner}], got {tuple(delta_u.shape)}")

        if self.random_gate:
            return torch.rand_like(delta_u)

        if self.independent_gate:
            gate = torch.sigmoid(self.psca_independent_gate.to(dtype=delta_u.dtype, device=delta_u.device))
            return gate.view(1, 1, -1).expand_as(delta_u)

        hidden = self.psca_down(delta_u)
        hidden = self.dropout(F.silu(hidden))
        return torch.sigmoid(self.psca_up(hidden))

    def _layout_update(self, update_bld: torch.Tensor, target: torch.Tensor, layout: str) -> torch.Tensor:
        layout = layout.lower()
        update_bld = update_bld.to(dtype=target.dtype, device=target.device)
        if layout == "bdl":
            update = update_bld.transpose(1, 2)
        elif layout == "bd":
            update = update_bld[:, -1, :]
        elif layout == "bld":
            update = update_bld
        else:
            raise RuntimeError(f"Unknown PSCA layout: {layout}")
        if update.shape != target.shape:
            raise RuntimeError(
                f"PSCA projector residual shape mismatch: update {tuple(update.shape)}, target {tuple(target.shape)}"
            )
        return update.contiguous()

    def apply_projector_residual(self, x: torch.Tensor, delta_u: torch.Tensor, layout: str) -> torch.Tensor:
        if (
            not self.projector_residual
            or self.projector_scale == 0
            or not self.use_projector_shift
            or self.random_gate
            or self.independent_gate
        ):
            return x
        update = self._layout_update(delta_u, x, layout)
        return (x + self.projector_scale * update).contiguous()

    def _match_update_to_bc(self, update_bln: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
        update_bln = update_bln.to(dtype=target.dtype, device=target.device)
        if target.ndim == 2:
            if update_bln.shape[1] != 1:
                update_bln = update_bln[:, -1:, :]
            if update_bln.shape[-1] != target.shape[-1]:
                raise RuntimeError(
                    f"PSCA-WR {name} step update dim mismatch: got {tuple(update_bln.shape)}, "
                    f"target {tuple(target.shape)}"
                )
            return update_bln.squeeze(1).contiguous()

        if target.ndim == 3:
            if target.shape[1] == self.d_state:
                update = update_bln.transpose(1, 2)
                if update.shape[-1] != target.shape[-1]:
                    if update.shape[-1] == 1:
                        update = update.expand(-1, -1, target.shape[-1])
                    else:
                        raise RuntimeError(
                            f"PSCA-WR {name} sequence length mismatch: got {tuple(update.shape)}, "
                            f"target {tuple(target.shape)}"
                        )
                return update.contiguous()

            if target.shape[-1] == self.d_state:
                if update_bln.shape[1] != 1:
                    update_bln = update_bln[:, -1:, :]
                update = update_bln.squeeze(1).unsqueeze(1)
                return update.expand(-1, target.shape[1], -1).contiguous()

        if target.ndim == 4:
            update = update_bln.transpose(1, 2).unsqueeze(1)
            if update.shape[-1] != target.shape[-1]:
                if update.shape[-1] == 1:
                    update = update.expand(-1, -1, -1, target.shape[-1])
                else:
                    raise RuntimeError(
                        f"PSCA-WR {name} 4D sequence length mismatch: got {tuple(update.shape)}, "
                        f"target {tuple(target.shape)}"
                    )
            if update.shape[1] != target.shape[1]:
                update = update.expand(-1, target.shape[1], -1, -1)
            return update.contiguous()

        raise RuntimeError(f"PSCA-WR cannot adapt {name} with shape {tuple(target.shape)}")

    def _alpha_for(self, target: torch.Tensor) -> torch.Tensor:
        return (self.psca_alpha * self.alpha_scale).to(dtype=target.dtype, device=target.device)

    def adapt_bc(
        self,
        B: torch.Tensor,
        C: torch.Tensor,
        v_bld: torch.Tensor,
        delta_u: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B_before_shape = tuple(B.shape)
        C_before_shape = tuple(C.shape)
        if delta_u is None:
            delta_u = self.compute_delta_u(v_bld)
        if gate is None:
            gate = self.compute_gate(delta_u)

        delta_B = self.psca_B(gate) if self.adapt_b else None
        delta_C = self.psca_C(gate) if self.adapt_c else None

        if delta_B is not None:
            B = B + self._alpha_for(B) * self._match_update_to_bc(delta_B, B, "B")
        if delta_C is not None:
            C = C + self._alpha_for(C) * self._match_update_to_bc(delta_C, C, "C")

        if self.debug and not self.debug_printed:
            delta_B_norm = 0.0 if delta_B is None else float(delta_B.detach().float().norm().cpu())
            delta_C_norm = 0.0 if delta_C is None else float(delta_C.detach().float().norm().cpu())
            print(
                "[PSCA-WR debug]"
                f" layer={self.layer_idx} delta_u shape: {tuple(delta_u.shape)}"
                f" gate shape: {tuple(gate.shape)}"
                f" B shape before/after: {B_before_shape}->{tuple(B.shape)}"
                f" C shape before/after: {C_before_shape}->{tuple(C.shape)}"
                f" delta_B norm: {delta_B_norm:.6g}"
                f" delta_C norm: {delta_C_norm:.6g}"
                f" psca_alpha: {float(self.psca_alpha.detach().float().cpu()):.6g}"
            )
            self.debug_printed = True

        return B.contiguous(), C.contiguous()

    def apply_lite(
        self,
        x: torch.Tensor,
        v_bld: torch.Tensor,
        layout: str,
        delta_u: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if delta_u is None:
            delta_u = self.compute_delta_u(v_bld)
        if gate is None:
            gate = self.compute_gate(delta_u)
        update = gate * self.psca_lite_vector.to(dtype=gate.dtype, device=gate.device).view(1, 1, -1)

        layout = layout.lower()
        if layout == "bdl":
            update = update.transpose(1, 2)
        elif layout == "bd":
            update = update[:, -1, :]
        elif layout != "bld":
            raise RuntimeError(f"Unknown PSCA_LITE layout: {layout}")

        out = x + self._alpha_for(x) * update.to(dtype=x.dtype, device=x.device)
        if self.debug and not self.debug_printed:
            print(
                "[PSCA_LITE debug]"
                f" layer={self.layer_idx} delta_u shape: {tuple(delta_u.shape)}"
                f" gate shape: {tuple(gate.shape)}"
                f" x shape before/after: {tuple(x.shape)}->{tuple(out.shape)}"
                f" update norm: {float(update.detach().float().norm().cpu()):.6g}"
                f" psca_alpha: {float(self.psca_alpha.detach().float().cpu()):.6g}"
            )
            self.debug_printed = True
        return out.contiguous()


def _find_mamba_blocks(model: nn.Module) -> List[nn.Module]:
    candidate = model
    while hasattr(candidate, "module"):
        candidate = candidate.module

    for obj in (candidate, getattr(candidate, "model", None), getattr(candidate, "base_model", None)):
        if obj is not None and hasattr(obj, "get_mamba_blocks"):
            return list(obj.get_mamba_blocks())

    blocks = []
    for module in candidate.modules():
        if all(hasattr(module, attr) for attr in ("d_inner", "d_state", "conv1d", "x_proj", "dt_proj")):
            blocks.append(module)
    return blocks


def _resolve_target_modules(target_modules: Any, num_layers: int) -> List[int]:
    if target_modules is None or target_modules == "all":
        return list(range(num_layers))
    if isinstance(target_modules, str):
        value = target_modules.strip().lower()
        if value == "all":
            return list(range(num_layers))
        raw = [item.strip() for item in value.split(",") if item.strip()]
        layers = [int(item) for item in raw]
    elif isinstance(target_modules, Iterable):
        values = list(target_modules)
        if len(values) == 1 and str(values[0]).strip().lower() == "all":
            return list(range(num_layers))
        layers = [int(item) for item in values]
    else:
        layers = [int(target_modules)]

    invalid = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Invalid PSCA-WR target layer(s) {invalid}; model has {num_layers} layers.")
    return sorted(set(layers))


def inject_psca_wr_adapters(model: nn.Module, config: PSCAWRConfig | Dict[str, Any] | SimpleNamespace) -> List[int]:
    if not isinstance(config, PSCAWRConfig):
        if isinstance(config, SimpleNamespace):
            config = merge_psca_wr_config(None, config)
        else:
            config = PSCAWRConfig.from_dict(dict(config))
    if not config.use_psca_wr:
        return []

    blocks = _find_mamba_blocks(model)
    if not blocks:
        raise RuntimeError("Could not find Mamba blocks for PSCA-WR injection.")

    target_layers = _resolve_target_modules(config.psca_target_modules, len(blocks))
    for layer_idx in target_layers:
        block = blocks[layer_idx]
        ref = block.conv1d.weight if hasattr(block, "conv1d") else next(block.parameters())
        block.psca_wr = PSCAWRAdapter(
            d_inner=block.d_inner,
            d_state=block.d_state,
            rank=config.psca_rank,
            alpha_scale=config.psca_alpha,
            dropout=config.psca_dropout,
            adapt_b=config.psca_adapt_b,
            adapt_c=config.psca_adapt_c,
            use_projector_shift=config.psca_use_projector_shift,
            projector_residual=config.psca_projector_residual,
            projector_scale=config.psca_projector_scale,
            fallback_lite=config.psca_fallback_lite,
            init_zero=config.psca_init_zero,
            random_gate=config.psca_random_gate,
            independent_gate=config.psca_independent_gate,
            layer_idx=layer_idx,
            debug=config.psca_debug,
            device=ref.device,
            dtype=ref.dtype,
        )

    setattr(model, "use_psca_wr", True)
    setattr(model, "psca_wr_config", config.to_dict())
    setattr(model, "psca_wr_target_layers", target_layers)
    return target_layers


def iter_psca_wr_adapters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, PSCAWRAdapter):
            yield module


def has_psca_wr_adapters(model: nn.Module) -> bool:
    return any(True for _ in iter_psca_wr_adapters(model))


def _is_small_classifier_param(name: str) -> bool:
    lowered = name.lower()
    if "lm_head" in lowered or "embedding" in lowered:
        return False
    return any(token in lowered for token in ("classifier", "classification_head", "score", "predictor"))


def _is_lora_param_name(name: str) -> bool:
    return "lora_" in name.lower()


def mark_only_psca_wr_as_trainable(
    model: nn.Module,
    train_classifier: bool = True,
    train_lora: bool = False,
) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        lowered = name.lower()
        if ".psca_wr." in lowered or "psca_" in lowered:
            param.requires_grad = True
        elif train_lora and _is_lora_param_name(name):
            param.requires_grad = True
        elif train_classifier and _is_small_classifier_param(name):
            param.requires_grad = True

    for adapter in iter_psca_wr_adapters(model):
        if adapter.fallback_lite:
            adapter.psca_B.requires_grad_(False)
            adapter.psca_C.requires_grad_(False)
        else:
            if not adapter.adapt_b:
                adapter.psca_B.requires_grad_(False)
            if not adapter.adapt_c:
                adapter.psca_C.requires_grad_(False)
            adapter.psca_lite_vector.requires_grad = False
        if adapter.random_gate:
            adapter.psca_proj_down.requires_grad_(False)
            adapter.psca_proj_up.requires_grad_(False)
            adapter.psca_down.requires_grad_(False)
            adapter.psca_up.requires_grad_(False)
            adapter.psca_independent_gate.requires_grad = False
        elif adapter.independent_gate:
            adapter.psca_proj_down.requires_grad_(False)
            adapter.psca_proj_up.requires_grad_(False)
            adapter.psca_down.requires_grad_(False)
            adapter.psca_up.requires_grad_(False)
        elif not adapter.use_projector_shift:
            adapter.psca_proj_down.requires_grad_(False)
            adapter.psca_proj_up.requires_grad_(False)
        if not adapter.independent_gate:
            adapter.psca_independent_gate.requires_grad = False

    freeze_lm_head_weight_for_psca_wr(model)


def freeze_lm_head_weight_for_psca_wr(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("lm_head.weight") and param.requires_grad:
            print("[PSCA-WR][warning] lm_head.weight was trainable; freezing it to avoid full LM-head tuning.")
            param.requires_grad = False


def print_psca_wr_summary(model: nn.Module, config: PSCAWRConfig, target_layers: List[int]) -> None:
    adapters = list(iter_psca_wr_adapters(model))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params > 0 else 0.0

    if config.psca_fallback_lite:
        print("Using PSCA_LITE fallback, not full PSCA_WR.")

    print("==== PSCA-WR Config ====")
    print(f"use_psca_wr: {config.use_psca_wr}")
    print(f"psca_rank: {config.psca_rank}")
    print(f"psca_alpha: {config.psca_alpha}")
    print(f"adapt_b: {config.psca_adapt_b}")
    print(f"adapt_c: {config.psca_adapt_c}")
    print(f"use_projector_shift: {config.psca_use_projector_shift}")
    print(f"projector_residual: {config.psca_projector_residual}")
    print(f"projector_scale: {config.psca_projector_scale}")
    print(f"fallback_lite: {config.psca_fallback_lite}")
    print(f"random_gate: {config.psca_random_gate}")
    print(f"independent_gate: {config.psca_independent_gate}")
    print(f"target_layers_resolved: {target_layers}")
    print(f"adapter_count: {len(adapters)}")
    print(f"trainable_params: {trainable_params:,}")
    print(f"trainable_ratio: {ratio:.6%}")
    print("========================")
    print("PSCA-WR trainable parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: shape={tuple(param.shape)}, params={param.numel():,}")
