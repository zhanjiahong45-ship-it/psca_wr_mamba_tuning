import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


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


def normalize_lora_target_modules(target_modules: Any) -> List[str]:
    if target_modules is None:
        return ["in_proj_x", "in_proj_z", "out_proj"]
    raw_items: List[str] = []
    if isinstance(target_modules, str):
        raw_items = [target_modules]
    elif isinstance(target_modules, Iterable):
        raw_items = [str(item) for item in target_modules]
    else:
        raw_items = [str(target_modules)]

    out: List[str] = []
    for item in raw_items:
        for token in item.split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


@dataclass
class LinearLoRAConfig:
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    lora_target_modules: Any = "in_proj_x,in_proj_z,out_proj"

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "LinearLoRAConfig":
        cfg = dict(cfg or {})
        if isinstance(cfg.get("lora"), dict):
            nested = dict(cfg["lora"])
            nested.update({key: value for key, value in cfg.items() if key != "lora"})
            cfg = nested

        if "r" in cfg and "lora_rank" not in cfg:
            cfg["lora_rank"] = cfg["r"]
        if "target_modules" in cfg and "lora_target_modules" not in cfg:
            cfg["lora_target_modules"] = cfg["target_modules"]
        if "bias" in cfg and "lora_bias" not in cfg:
            cfg["lora_bias"] = cfg["bias"]

        peft_type = str(cfg.get("peft_type", "")).upper()
        method = str(cfg.get("method", "")).lower()
        if peft_type == "LORA" or "lora" in method:
            cfg["use_lora"] = True

        allowed = set(cls.__dataclass_fields__.keys())
        values = {key: cfg[key] for key in allowed if key in cfg}
        out = cls(**values)
        out.use_lora = _as_bool(out.use_lora)
        out.lora_rank = int(out.lora_rank)
        out.lora_alpha = float(out.lora_alpha)
        out.lora_dropout = float(out.lora_dropout)
        out.lora_bias = str(out.lora_bias).lower()
        out.lora_target_modules = normalize_lora_target_modules(out.lora_target_modules)

        if out.lora_rank <= 0:
            raise ValueError("lora_rank must be positive.")
        if out.lora_dropout < 0:
            raise ValueError("lora_dropout must be non-negative.")
        if out.lora_bias != "none":
            raise ValueError("Only lora_bias='none' is supported for LoRA(inoutproj).")
        return out

    def to_dict(self) -> Dict[str, Any]:
        values = dict(self.__dict__)
        values["lora_target_modules"] = list(normalize_lora_target_modules(values.get("lora_target_modules")))
        return values


def is_lora_config_dict(cfg: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(cfg, dict):
        return False
    if isinstance(cfg.get("lora"), dict) and is_lora_config_dict(cfg.get("lora")):
        return True
    peft_type = str(cfg.get("peft_type", "")).upper()
    method = str(cfg.get("method", "")).lower()
    return bool(cfg.get("use_lora", False)) or peft_type == "LORA" or "lora" in method


def merge_lora_config(peft_cfg: Optional[Dict[str, Any]], overrides: Optional[Any] = None) -> LinearLoRAConfig:
    cfg = dict(peft_cfg or {})
    if overrides is not None:
        for key in LinearLoRAConfig.__dataclass_fields__.keys():
            if hasattr(overrides, key):
                value = getattr(overrides, key)
                if value is not None:
                    cfg[key] = value
    return LinearLoRAConfig.from_dict(cfg)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear can only wrap nn.Linear, got {type(base_layer)}")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.lora_alpha = float(alpha)
        self.scaling = self.lora_alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False, device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False, device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.reset_lora_parameters()
        self.base_layer.requires_grad_(False)

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base + update.to(dtype=base.dtype) * self.scaling


_ALLOWED_TARGETS = {"in_proj_x", "in_proj_z", "out_proj"}
_FORBIDDEN_TARGETS = {
    "x_proj",
    "x_proj_B",
    "x_proj_C",
    "x_proj_dt",
    "dt_proj",
    "A_log",
    "conv1d",
    "psca_B",
    "psca_C",
    "psca_proj_down",
    "psca_proj_up",
    "psca_down",
    "psca_up",
}


def _get_parent_module(root: nn.Module, module_name: str) -> nn.Module:
    parent = root
    if not module_name:
        return parent
    for token in module_name.split("."):
        parent = getattr(parent, token)
    return parent


def inject_lora_linear_adapters(
    model: nn.Module,
    config: LinearLoRAConfig | Dict[str, Any] | SimpleNamespace,
) -> List[str]:
    if not isinstance(config, LinearLoRAConfig):
        if isinstance(config, SimpleNamespace):
            config = merge_lora_config(None, config)
        else:
            config = LinearLoRAConfig.from_dict(dict(config))
    if not config.use_lora:
        return []

    targets = set(normalize_lora_target_modules(config.lora_target_modules))
    forbidden = sorted(targets & _FORBIDDEN_TARGETS)
    if forbidden:
        raise ValueError(f"LoRA(inoutproj) must not target S6/PSCA internals: {forbidden}")
    unsupported = sorted(targets - _ALLOWED_TARGETS)
    if unsupported:
        raise ValueError(f"Unsupported LoRA target_modules for this experiment: {unsupported}")

    if hasattr(model, "split_layers"):
        model.split_layers()

    resolved: List[str] = []
    modules = list(model.named_modules())
    for full_name, module in modules:
        if not full_name:
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        if isinstance(module, LoRALinear):
            resolved.append(full_name)
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target {full_name} is {type(module)}, expected nn.Linear.")
        parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        parent = _get_parent_module(model, parent_name)
        setattr(
            parent,
            leaf,
            LoRALinear(
                module,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            ),
        )
        resolved.append(full_name)

    missing = sorted(targets - {name.rsplit(".", 1)[-1] for name in resolved})
    if missing:
        raise RuntimeError(f"LoRA target_modules not found after split_layers(): {missing}")

    setattr(model, "use_lora", True)
    setattr(model, "lora_config", config.to_dict())
    setattr(model, "lora_target_modules_resolved", resolved)
    return resolved


def is_lora_param_name(name: str) -> bool:
    return "lora_" in name.lower()


def _is_small_classifier_param(name: str) -> bool:
    lowered = name.lower()
    if "lm_head" in lowered or "embedding" in lowered:
        return False
    return any(token in lowered for token in ("classifier", "classification_head", "score", "predictor"))


def mark_only_lora_as_trainable(model: nn.Module, train_classifier: bool = True) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if is_lora_param_name(name):
            param.requires_grad = True
        elif train_classifier and _is_small_classifier_param(name):
            param.requires_grad = True
    freeze_lm_head_weight_for_lora(model)


def freeze_lm_head_weight_for_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("lm_head.weight") and param.requires_grad:
            print("[LoRA][warning] lm_head.weight was trainable; freezing it to avoid full LM-head tuning.")
            param.requires_grad = False


def get_lora_trainable_parameter_counts(model: nn.Module) -> Dict[str, int]:
    counts = {"lora": 0, "psca_wr": 0, "classifier": 0, "other": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if is_lora_param_name(name):
            counts["lora"] += param.numel()
        elif ".psca_wr." in lowered or "psca_" in lowered:
            counts["psca_wr"] += param.numel()
        elif _is_small_classifier_param(name):
            counts["classifier"] += param.numel()
        else:
            counts["other"] += param.numel()
    return counts


def validate_lora_trainable_parameters(model: nn.Module) -> None:
    bad_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if "base_layer.weight" in lowered or "base_layer.bias" in lowered:
            bad_names.append(name)
        elif "embedding" in lowered or name.endswith("lm_head.weight"):
            bad_names.append(name)
        elif any(token in lowered for token in ("x_proj", "dt_proj", "a_log", "conv1d")) and not is_lora_param_name(name):
            bad_names.append(name)
    if bad_names:
        raise RuntimeError("Unexpected trainable parameters for LoRA(inoutproj): " + ", ".join(bad_names))


def print_lora_summary(model: nn.Module, config: LinearLoRAConfig, target_modules: Sequence[str]) -> None:
    validate_lora_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params > 0 else 0.0
    group_counts = get_lora_trainable_parameter_counts(model)

    print("==== LoRA LinProj Config ====")
    print(f"use_lora: {config.use_lora}")
    print(f"lora_rank: {config.lora_rank}")
    print(f"lora_alpha: {config.lora_alpha}")
    print(f"lora_dropout: {config.lora_dropout}")
    print(f"lora_bias: {config.lora_bias}")
    print(f"target_modules_requested: {normalize_lora_target_modules(config.lora_target_modules)}")
    print(f"target_modules_resolved: {list(target_modules)}")
    print(f"total_params: {total_params:,}")
    print(f"trainable_params: {trainable_params:,}")
    print(f"trainable_ratio: {ratio:.6%}")
    print(f"lora_trainable_params: {group_counts['lora']:,}")
    print(f"psca_wr_trainable_params: {group_counts['psca_wr']:,}")
    print(f"classifier_trainable_params: {group_counts['classifier']:,}")
    print(f"other_trainable_params: {group_counts['other']:,}")
    print("=============================")
