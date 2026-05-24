import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _as_int_list(values) -> List[int]:
    if values is None:
        return []
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    return [int(v) for v in values]


def _iter_model_candidates(model) -> Iterable[nn.Module]:
    queue = [model]
    seen = set()

    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate

        if hasattr(candidate, "get_base_model"):
            try:
                queue.append(candidate.get_base_model())
            except Exception:
                pass

        for attr in ("module", "model", "base_model"):
            child = getattr(candidate, attr, None)
            if child is not None:
                queue.append(child)


def _get_ds_host(model) -> nn.Module:
    for candidate in _iter_model_candidates(model):
        if hasattr(candidate, "use_deep_supervision") or hasattr(candidate, "auxiliary_heads"):
            return candidate
    return model.module if hasattr(model, "module") else model


def get_mamba_backbone_model(model) -> nn.Module:
    for candidate in _iter_model_candidates(model):
        if hasattr(candidate, "backbone") and hasattr(candidate.backbone, "layers"):
            return candidate
    raise AttributeError("Could not find a Mamba backbone with layers on the model.")


def is_deep_supervision_enabled(model) -> bool:
    host = _get_ds_host(model)
    return bool(getattr(host, "use_deep_supervision", False) or getattr(host, "ds_adaptive", False))


def get_deep_supervision_required_layers(model) -> List[int]:
    host = _get_ds_host(model)
    if not bool(getattr(host, "use_deep_supervision", False)):
        return []

    layers = set()
    if bool(getattr(host, "ds_adaptive", False)):
        layers.update(int(layer) for layer in getattr(host, "candidate_aux_layers", []))
        layers.update(int(layer) for layer in getattr(host, "selected_aux_layers", []))
    layers.update(int(layer) for layer in getattr(host, "aux_layers", []))

    heads = getattr(host, "auxiliary_heads", None)
    if heads is not None:
        for layer in heads.keys():
            layers.add(int(layer))

    return sorted(layer for layer in layers if layer > 0)


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _scheduled_aux_weight(host, progress: float) -> float:
    progress = max(0.0, min(1.0, float(progress)))
    schedule = str(getattr(host, "ds_schedule", "constant")).lower()
    aux_loss_weight = float(
        getattr(host, "effective_aux_loss_weight", getattr(host, "aux_loss_weight", 0.1))
    )

    if schedule == "constant":
        return aux_loss_weight

    if schedule != "linear_warmup":
        raise ValueError(f"Unsupported ds_schedule: {schedule}")

    start = float(getattr(host, "ds_start_ratio", 0.0))
    warmup = float(getattr(host, "ds_warmup_ratio", 0.0))
    if progress < start:
        return 0.0
    if warmup <= 0:
        return aux_loss_weight
    if progress < start + warmup:
        return aux_loss_weight * ((progress - start) / warmup)
    return aux_loss_weight


def get_deep_supervision_loss_weight(model, progress: Optional[float] = None) -> float:
    host = _get_ds_host(model)
    progress = 0.0 if progress is None else float(progress)

    if bool(getattr(host, "ds_adaptive", False)):
        if not bool(getattr(host, "ds_selection_done", False)) and progress < float(getattr(host, "probe_ratio", 0.15)):
            return float(getattr(host, "probe_aux_weight", 0.05))
        if bool(getattr(host, "adaptive_ds_disabled", False)):
            return 0.0

    return _scheduled_aux_weight(host, progress)


def _normalize_aux_layers(aux_layers, n_layers: int) -> List[int]:
    valid_layers = []
    seen = set()
    for layer in sorted(_as_int_list(aux_layers)):
        if layer in seen:
            continue
        seen.add(layer)
        if layer < 1 or layer > n_layers:
            warnings.warn(
                f"Ignoring aux layer {layer}: valid 1-based range is [1, {n_layers}].",
                stacklevel=2,
            )
            print(f"Warning: ignoring aux layer {layer}; valid 1-based range is [1, {n_layers}].")
            continue
        valid_layers.append(layer)
    return valid_layers


def _build_aux_weights(aux_layers: List[int], scheme: str) -> Dict[int, float]:
    if not aux_layers:
        return {}
    scheme = str(scheme).lower()
    if scheme == "uniform":
        raw = torch.ones(len(aux_layers), dtype=torch.float32)
    elif scheme == "linear_increase":
        raw = torch.arange(1, len(aux_layers) + 1, dtype=torch.float32)
    elif scheme == "adaptive":
        raw = torch.ones(len(aux_layers), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported aux_weight_scheme: {scheme}")

    weights = raw / raw.sum()
    return {layer: float(weight.item()) for layer, weight in zip(aux_layers, weights)}


def _is_tied_to_embedding(lm_head: nn.Module, embedding: nn.Module) -> bool:
    if not hasattr(lm_head, "weight") or not hasattr(embedding, "weight"):
        return False
    if lm_head.weight is embedding.weight:
        return True
    return lm_head.weight.data_ptr() == embedding.weight.data_ptr()


def _freeze_lm_head_and_embedding(backbone_model: nn.Module) -> List[str]:
    """Keep the LM vocabulary projection frozen for PEFT-fair SOT+DS training."""
    lm_head = getattr(backbone_model, "lm_head", None)
    embedding = getattr(getattr(backbone_model, "backbone", None), "embedding", None)

    names = []
    if embedding is not None:
        for name, param in embedding.named_parameters(prefix="backbone.embedding"):
            param.requires_grad = False
            names.append(name)

    if lm_head is not None:
        for name, param in lm_head.named_parameters(prefix="lm_head"):
            param.requires_grad = False
            names.append(name)
    else:
        print("Warning: no lm_head found; final LM vocabulary head freeze was skipped.")

    return names


def _assert_lm_head_and_embedding_frozen(backbone_model: nn.Module) -> None:
    leaked = []
    for name, param in backbone_model.named_parameters():
        if param.requires_grad and (
            name == "lm_head.weight"
            or name.startswith("lm_head.")
            or name == "backbone.embedding.weight"
            or name.startswith("backbone.embedding.")
        ):
            leaked.append(name)
    if leaked:
        raise RuntimeError(
            "SOT+DS must not train the full LM vocabulary head or token embeddings. "
            f"Still-trainable parameters: {leaked}"
        )


def configure_deep_supervision(
    model,
    use_deep_supervision: bool = False,
    aux_layers: Optional[List[int]] = None,
    aux_loss_weight: float = 0.1,
    aux_weight_scheme: str = "linear_increase",
    aux_pooling: str = "last_token",
    ds_adaptive: bool = False,
    ds_adaptive_strategy: str = "loss_drop",
    candidate_aux_layers: Optional[List[int]] = None,
    probe_ratio: float = 0.15,
    probe_aux_weight: float = 0.05,
    probe_start_window_ratio: float = 0.3,
    probe_end_window_ratio: float = 0.3,
    probe_loss_stat: str = "window_mean",
    adaptive_top_k: int = 3,
    adaptive_late_bias_gamma: float = 1.0,
    adaptive_score_mode: str = "drop_plus_confidence",
    adaptive_confidence_weight: float = 0.1,
    adaptive_score_threshold: float = 0.02,
    adaptive_min_layer: int = 0,
    adaptive_disable_if_low_score: bool = False,
    fallback_aux_layers: Optional[List[int]] = None,
    fallback_aux_weight_scheme: str = "linear_increase",
    fallback_aux_loss_weight_scale: float = 0.5,
    fallback_confidence_threshold: float = 0.0,
    ds_schedule: str = "constant",
    ds_start_ratio: Optional[float] = None,
    ds_warmup_ratio: float = 0.3,
    num_labels: Optional[int] = None,
    label_token_ids: Optional[List[int]] = None,
    train_final_head: bool = False,
    task_name: Optional[str] = None,
    selected_aux_layers: Optional[List[int]] = None,
    adaptive_aux_weights: Optional[Dict[int, float]] = None,
    ds_selection_done: Optional[bool] = None,
    adaptive_ds_disabled: bool = False,
    no_print: bool = False,
):
    host = _get_ds_host(model)
    host.use_deep_supervision = bool(use_deep_supervision)

    if not host.use_deep_supervision:
        return model

    if num_labels is None or int(num_labels) <= 0:
        raise ValueError("Deep supervision requires a positive num_labels value.")
    if label_token_ids is None or len(label_token_ids) != int(num_labels):
        raise ValueError("Deep supervision requires one label token id per class.")

    aux_pooling = str(aux_pooling).lower()
    if aux_pooling != "last_token":
        raise ValueError("SOT+DS currently supports aux_pooling='last_token' only.")

    ds_adaptive = bool(ds_adaptive)
    ds_adaptive_strategy = str(ds_adaptive_strategy).lower()
    if ds_adaptive and ds_adaptive_strategy != "loss_drop":
        raise ValueError("Probe-then-Adapt DS-SOT currently supports ds_adaptive_strategy='loss_drop' only.")

    aux_loss_weight = float(aux_loss_weight)
    if aux_loss_weight < 0:
        raise ValueError("aux_loss_weight must be non-negative.")
    if aux_loss_weight > 1.0:
        warnings.warn(
            f"aux_loss_weight={aux_loss_weight} is large; values around 0.05-0.1 are recommended.",
            stacklevel=2,
        )
        print(f"Warning: aux_loss_weight={aux_loss_weight} is large; 0.05-0.1 is recommended.")

    probe_ratio = max(0.0, min(1.0, float(probe_ratio)))
    probe_aux_weight = float(probe_aux_weight)
    probe_start_window_ratio = max(0.0, min(1.0, float(probe_start_window_ratio)))
    probe_end_window_ratio = max(0.0, min(1.0, float(probe_end_window_ratio)))
    probe_loss_stat = str(probe_loss_stat).lower()
    if probe_loss_stat not in ("window_mean", "ema"):
        raise ValueError("probe_loss_stat must be one of: window_mean, ema.")
    adaptive_top_k = max(0, int(adaptive_top_k))
    adaptive_late_bias_gamma = float(adaptive_late_bias_gamma)
    adaptive_score_mode = str(adaptive_score_mode).lower()
    if adaptive_score_mode not in ("loss_drop", "drop_plus_confidence"):
        raise ValueError("adaptive_score_mode must be one of: loss_drop, drop_plus_confidence.")
    adaptive_confidence_weight = float(adaptive_confidence_weight)
    adaptive_score_threshold = float(adaptive_score_threshold)
    adaptive_min_layer = int(adaptive_min_layer)
    adaptive_disable_if_low_score = bool(adaptive_disable_if_low_score)
    fallback_aux_weight_scheme = str(fallback_aux_weight_scheme).lower()
    if fallback_aux_weight_scheme not in ("uniform", "linear_increase"):
        raise ValueError("fallback_aux_weight_scheme must be one of: uniform, linear_increase.")
    fallback_aux_loss_weight_scale = max(0.0, float(fallback_aux_loss_weight_scale))
    fallback_confidence_threshold = float(fallback_confidence_threshold)
    ds_schedule = str(ds_schedule).lower()
    if ds_schedule not in ("constant", "linear_warmup"):
        raise ValueError("ds_schedule must be one of: constant, linear_warmup.")
    if ds_start_ratio is None:
        ds_start_ratio = probe_ratio if ds_adaptive else 0.0
    ds_start_ratio = max(0.0, min(1.0, float(ds_start_ratio)))
    ds_warmup_ratio = max(0.0, float(ds_warmup_ratio))

    backbone_model = get_mamba_backbone_model(model)
    n_layers = len(backbone_model.backbone.layers)
    hidden_size = int(getattr(backbone_model.config, "d_model", backbone_model.backbone.embedding.weight.shape[1]))
    embedding = backbone_model.backbone.embedding
    device = embedding.weight.device
    dtype = embedding.weight.dtype

    if ds_adaptive:
        candidate_layers = candidate_aux_layers if candidate_aux_layers is not None else [4, 8, 12, 16, 20, 24]
        valid_candidate_layers = _normalize_aux_layers(candidate_layers, n_layers)
        requested_fallback_layers = fallback_aux_layers if fallback_aux_layers is not None else [16, 20, 24]
        valid_fallback_layers = _normalize_aux_layers(requested_fallback_layers, n_layers)
        fallback_head_layers = valid_fallback_layers or ([n_layers] if n_layers > 0 else [])
        head_layers = sorted(set(valid_candidate_layers) | set(fallback_head_layers))
        restored_selected_layers = _normalize_aux_layers(selected_aux_layers, n_layers) if selected_aux_layers else None
        if restored_selected_layers is not None:
            valid_layers = restored_selected_layers
            aux_weights = {
                int(layer): float(weight)
                for layer, weight in (adaptive_aux_weights or {}).items()
                if int(layer) in valid_layers
            }
            if not aux_weights and valid_layers:
                aux_weights = _build_aux_weights(valid_layers, "uniform")
        else:
            valid_layers = []
            aux_weights = {}
    else:
        valid_candidate_layers = []
        valid_fallback_layers = []
        valid_layers = _normalize_aux_layers(aux_layers, n_layers)
        head_layers = valid_layers
        if str(aux_weight_scheme).lower() == "adaptive":
            raise ValueError("aux_weight_scheme='adaptive' requires ds_adaptive=True.")
        aux_weights = _build_aux_weights(valid_layers, aux_weight_scheme)

    host.auxiliary_heads = nn.ModuleDict(
        {
            str(layer): nn.Linear(hidden_size, int(num_labels), device=device, dtype=dtype)
            for layer in head_layers
        }
    )
    for param in host.auxiliary_heads.parameters():
        param.requires_grad = True

    if train_final_head:
        print(
            "Warning: train_final_head=True is ignored for SOT+DS; "
            "the full LM vocabulary head stays frozen to preserve PEFT parameter counts."
        )
    frozen_lm_head_params = _freeze_lm_head_and_embedding(backbone_model)
    _assert_lm_head_and_embedding_frozen(backbone_model)
    train_final_head = False

    host.aux_layers = valid_layers
    host.aux_layer_indices_0based = [layer - 1 for layer in valid_layers]
    host.aux_loss_weight = aux_loss_weight
    host.aux_weight_scheme = str(aux_weight_scheme).lower()
    host.aux_loss_weights_by_layer = aux_weights
    host.aux_pooling = aux_pooling
    host.aux_num_labels = int(num_labels)
    host.aux_label_token_ids = [int(x) for x in label_token_ids]
    host.ds_adaptive = ds_adaptive
    host.ds_adaptive_strategy = ds_adaptive_strategy
    host.candidate_aux_layers = valid_candidate_layers
    host.probe_ratio = probe_ratio
    host.probe_aux_weight = probe_aux_weight
    host.probe_start_window_ratio = probe_start_window_ratio
    host.probe_end_window_ratio = probe_end_window_ratio
    host.probe_loss_stat = probe_loss_stat
    host.adaptive_top_k = adaptive_top_k
    host.adaptive_late_bias_gamma = adaptive_late_bias_gamma
    host.adaptive_score_mode = adaptive_score_mode
    host.adaptive_confidence_weight = adaptive_confidence_weight
    host.adaptive_score_threshold = adaptive_score_threshold
    host.adaptive_min_layer = adaptive_min_layer
    host.adaptive_disable_if_low_score = adaptive_disable_if_low_score
    host.fallback_aux_layers = list(valid_fallback_layers)
    host.fallback_aux_weight_scheme = fallback_aux_weight_scheme
    host.fallback_aux_loss_weight_scale = fallback_aux_loss_weight_scale
    host.fallback_confidence_threshold = fallback_confidence_threshold
    host.ds_schedule = ds_schedule
    host.ds_start_ratio = ds_start_ratio
    host.ds_warmup_ratio = ds_warmup_ratio
    host.ds_total_num_layers = n_layers
    host.probe_loss_start = {}
    host.probe_loss_ema = {}
    host.probe_loss_count = {}
    host.probe_loss_seen = {}
    host.probe_start_loss_sum = {}
    host.probe_start_loss_count = {}
    host.probe_end_loss_sum = {}
    host.probe_end_loss_count = {}
    host.probe_ema_beta = 0.9
    host.adaptive_scores = {}
    host.adaptive_layer_stats = {}
    host.adaptive_ds_disabled = bool(adaptive_ds_disabled)
    host.adaptive_used_fallback = False
    host.effective_aux_loss_weight = 0.0 if bool(adaptive_ds_disabled) else aux_loss_weight
    if ds_selection_done is None:
        ds_selection_done = bool(ds_adaptive and selected_aux_layers)
    host.ds_selection_done = bool(ds_selection_done)
    host.selected_aux_layers = list(valid_layers) if host.ds_selection_done else []
    host.latest_loss_dict = {}
    _sync_deep_supervision_config(host, train_final_head=False, task_name=task_name)

    if not no_print:
        print("Deep supervision enabled:")
        print(f"  task_name: {task_name}")
        print(f"  ds_adaptive: {ds_adaptive}")
        if ds_adaptive:
            print(f"  method: Probe-then-Adapt DS-SOT")
            print(f"  requested candidate aux layers (1-based): {_as_int_list(candidate_layers)}")
            print(f"  enabled candidate aux layers (1-based): {valid_candidate_layers}")
            print(f"  probing layer_idx (0-based): {[layer - 1 for layer in valid_candidate_layers]}")
            print(f"  probe_ratio: {probe_ratio}")
            print(f"  probe_aux_weight: {probe_aux_weight}")
            print(f"  probe_start_window_ratio: {probe_start_window_ratio}")
            print(f"  probe_end_window_ratio: {probe_end_window_ratio}")
            print(f"  probe_loss_stat: {probe_loss_stat}")
            print(f"  adaptive_top_k: {adaptive_top_k}")
            print(f"  adaptive_score_mode: {adaptive_score_mode}")
            print(f"  adaptive_confidence_weight: {adaptive_confidence_weight}")
            print(f"  adaptive_score_threshold: {adaptive_score_threshold}")
            print(f"  adaptive_disable_if_low_score: {adaptive_disable_if_low_score}")
            print(f"  fallback aux layers (1-based): {valid_fallback_layers}")
            print(f"  fallback_aux_weight_scheme: {fallback_aux_weight_scheme}")
            print(f"  fallback_aux_loss_weight_scale: {fallback_aux_loss_weight_scale}")
            print(f"  fallback_confidence_threshold: {fallback_confidence_threshold}")
            print(f"  ds_schedule: {ds_schedule}")
            print(f"  ds_start_ratio: {ds_start_ratio}")
            print(f"  ds_warmup_ratio: {ds_warmup_ratio}")
        else:
            print(f"  requested aux layers (1-based): {_as_int_list(aux_layers)}")
            print(f"  enabled aux layers (1-based): {valid_layers}")
            print(f"  internal Mamba layer_idx (0-based): {[layer - 1 for layer in valid_layers]}")
        print(f"  aux loss weights: {aux_weights}")
        print(f"  aux_loss_weight: {aux_loss_weight}")
        print(f"  aux_pooling: {aux_pooling}")
        print(f"  label_token_ids: {host.aux_label_token_ids}")
        if frozen_lm_head_params:
            print(f"  frozen LM head / embedding params: {frozen_lm_head_params}")
        if ds_adaptive and not valid_candidate_layers:
            print("Warning: ds_adaptive=True but no valid candidate aux layers were enabled.")
        elif not ds_adaptive and not valid_layers:
            print("Warning: use_deep_supervision=True but no valid aux layers were enabled.")

    return model


def _sync_deep_supervision_config(host, train_final_head: bool, task_name: Optional[str]) -> None:
    host.deep_supervision_config = {
        "use_deep_supervision": True,
        "aux_layers": list(getattr(host, "aux_layers", [])),
        "aux_loss_weight": float(getattr(host, "aux_loss_weight", 0.1)),
        "aux_weight_scheme": str(getattr(host, "aux_weight_scheme", "linear_increase")),
        "aux_pooling": str(getattr(host, "aux_pooling", "last_token")),
        "ds_adaptive": bool(getattr(host, "ds_adaptive", False)),
        "ds_adaptive_strategy": str(getattr(host, "ds_adaptive_strategy", "loss_drop")),
        "candidate_aux_layers": list(getattr(host, "candidate_aux_layers", [])),
        "probe_ratio": float(getattr(host, "probe_ratio", 0.15)),
        "probe_aux_weight": float(getattr(host, "probe_aux_weight", 0.05)),
        "probe_start_window_ratio": float(getattr(host, "probe_start_window_ratio", 0.3)),
        "probe_end_window_ratio": float(getattr(host, "probe_end_window_ratio", 0.3)),
        "probe_loss_stat": str(getattr(host, "probe_loss_stat", "window_mean")),
        "adaptive_top_k": int(getattr(host, "adaptive_top_k", 3)),
        "adaptive_late_bias_gamma": float(getattr(host, "adaptive_late_bias_gamma", 1.0)),
        "adaptive_score_mode": str(getattr(host, "adaptive_score_mode", "drop_plus_confidence")),
        "adaptive_confidence_weight": float(getattr(host, "adaptive_confidence_weight", 0.1)),
        "adaptive_score_threshold": float(getattr(host, "adaptive_score_threshold", 0.02)),
        "adaptive_min_layer": int(getattr(host, "adaptive_min_layer", 0)),
        "adaptive_disable_if_low_score": bool(getattr(host, "adaptive_disable_if_low_score", False)),
        "fallback_aux_layers": list(getattr(host, "fallback_aux_layers", [])),
        "fallback_aux_weight_scheme": str(getattr(host, "fallback_aux_weight_scheme", "linear_increase")),
        "fallback_aux_loss_weight_scale": float(getattr(host, "fallback_aux_loss_weight_scale", 0.5)),
        "fallback_confidence_threshold": float(getattr(host, "fallback_confidence_threshold", 0.0)),
        "adaptive_used_fallback": bool(getattr(host, "adaptive_used_fallback", False)),
        "effective_aux_loss_weight": float(getattr(host, "effective_aux_loss_weight", getattr(host, "aux_loss_weight", 0.1))),
        "ds_schedule": str(getattr(host, "ds_schedule", "constant")),
        "ds_start_ratio": float(getattr(host, "ds_start_ratio", 0.0)),
        "ds_warmup_ratio": float(getattr(host, "ds_warmup_ratio", 0.3)),
        "selected_aux_layers": list(getattr(host, "selected_aux_layers", [])),
        "adaptive_aux_weights": {
            int(layer): float(weight)
            for layer, weight in getattr(host, "aux_loss_weights_by_layer", {}).items()
        },
        "ds_selection_done": bool(getattr(host, "ds_selection_done", False)),
        "adaptive_ds_disabled": bool(getattr(host, "adaptive_ds_disabled", False)),
        "num_labels": int(getattr(host, "aux_num_labels", 0)),
        "label_token_ids": [int(x) for x in getattr(host, "aux_label_token_ids", [])],
        "train_final_head": bool(train_final_head),
        "task_name": task_name,
    }

    peft_args = getattr(host, "peft_args", None)
    if isinstance(peft_args, dict):
        peft_args["deep_supervision"] = host.deep_supervision_config


def _targets_from_label_ids(
    label_ids: torch.Tensor,
    label_token_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_label = label_ids != -100
    has_label = valid_label.any(dim=1)
    reverse_positions = valid_label.flip(1).long().argmax(dim=1)
    positions = label_ids.shape[1] - 1 - reverse_positions
    batch_idx = torch.arange(label_ids.shape[0], device=label_ids.device)
    token_targets = label_ids[batch_idx, positions]

    class_targets = torch.full(
        (label_ids.shape[0],),
        -100,
        dtype=torch.long,
        device=label_ids.device,
    )
    for class_idx, token_id in enumerate(label_token_ids):
        class_targets[token_targets == int(token_id)] = class_idx

    valid_class = has_label & (class_targets != -100)
    return positions, class_targets, valid_class


def _layers_to_text(layers: List[int]) -> str:
    return ",".join(str(layer) for layer in layers)


def _compute_aux_layer_losses(host, hidden_states, label_ids, layers: List[int]) -> Dict[int, torch.Tensor]:
    positions, class_targets, valid_class = _targets_from_label_ids(
        label_ids,
        getattr(host, "aux_label_token_ids", []),
    )
    if not bool(valid_class.any()):
        return {}

    batch_idx = torch.arange(label_ids.shape[0], device=label_ids.device)[valid_class]
    positions = positions[valid_class]
    class_targets = class_targets[valid_class]
    heads = getattr(host, "auxiliary_heads", None)
    losses = {}

    for layer in layers:
        if heads is None or str(layer) not in heads:
            continue
        if layer >= len(hidden_states):
            raise RuntimeError(
                f"Deep supervision requires hidden_states for 1-based aux layer {layer}, "
                f"but hidden_states has length {len(hidden_states)}. "
                "Check that output_hidden_states=True reaches the Mamba backbone or that Trainer hooks are active."
            )

        head = heads[str(layer)]
        layer_hidden = hidden_states[layer]
        if layer_hidden is None:
            raise RuntimeError(
                f"Deep supervision requires hidden_states[{layer}] for 1-based aux layer {layer}, "
                "but that slot is None."
            )
        if layer_hidden.shape[1] != label_ids.shape[1]:
            max_pos = layer_hidden.shape[1] - 1
            layer_positions = positions.clamp(max=max_pos)
        else:
            layer_positions = positions

        selected = layer_hidden[batch_idx, layer_positions]
        logits = head(selected).float()
        losses[layer] = F.cross_entropy(logits, class_targets)

    return losses


def _update_probe_stats(host, layer_losses: Dict[int, torch.Tensor], progress: float) -> None:
    beta = float(getattr(host, "probe_ema_beta", 0.9))
    probe_ratio = float(getattr(host, "probe_ratio", 0.15))
    probe_progress = 1.0 if probe_ratio <= 0 else max(0.0, min(1.0, float(progress) / probe_ratio))
    start_window = float(getattr(host, "probe_start_window_ratio", 0.3))
    end_window = float(getattr(host, "probe_end_window_ratio", 0.3))
    use_window_stats = str(getattr(host, "probe_loss_stat", "window_mean")).lower() == "window_mean"

    for layer, loss in layer_losses.items():
        value = float(loss.detach().cpu())
        count = int(getattr(host, "probe_loss_seen", {}).get(layer, 0))
        if count == 0:
            host.probe_loss_start[layer] = value
            host.probe_loss_ema[layer] = value
        else:
            host.probe_loss_ema[layer] = beta * float(host.probe_loss_ema[layer]) + (1.0 - beta) * value
        host.probe_loss_seen[layer] = count + 1
        host.probe_loss_count[layer] = count + 1

        if use_window_stats and probe_progress <= start_window:
            host.probe_start_loss_sum[layer] = float(host.probe_start_loss_sum.get(layer, 0.0)) + value
            host.probe_start_loss_count[layer] = int(host.probe_start_loss_count.get(layer, 0)) + 1

        if use_window_stats and probe_progress >= 1.0 - end_window:
            host.probe_end_loss_sum[layer] = float(host.probe_end_loss_sum.get(layer, 0.0)) + value
            host.probe_end_loss_count[layer] = int(host.probe_end_loss_count.get(layer, 0)) + 1


def _probe_window_loss(host, layer: int, phase: str) -> Tuple[Optional[float], bool]:
    use_window_stats = str(getattr(host, "probe_loss_stat", "window_mean")).lower() == "window_mean"
    if phase == "start":
        count = int(getattr(host, "probe_start_loss_count", {}).get(layer, 0))
        total = float(getattr(host, "probe_start_loss_sum", {}).get(layer, 0.0))
        fallback = getattr(host, "probe_loss_start", {}).get(layer)
    else:
        count = int(getattr(host, "probe_end_loss_count", {}).get(layer, 0))
        total = float(getattr(host, "probe_end_loss_sum", {}).get(layer, 0.0))
        fallback = getattr(host, "probe_loss_ema", {}).get(layer)

    if count > 0:
        return total / float(count), False

    if fallback is not None:
        if use_window_stats:
            print(
                f"Warning: probing {phase} window for layer {layer} has no samples; "
                "falling back to EMA/first-loss statistics."
            )
        return float(fallback), True

    return None, True


def _select_adaptive_aux_layers(host) -> None:
    if bool(getattr(host, "ds_selection_done", False)):
        return

    eps = 1e-8
    total_layers = max(1, int(getattr(host, "ds_total_num_layers", 1)))
    min_layer = int(getattr(host, "adaptive_min_layer", 0))
    gamma = float(getattr(host, "adaptive_late_bias_gamma", 1.0))
    score_mode = str(getattr(host, "adaptive_score_mode", "drop_plus_confidence")).lower()
    confidence_weight = float(getattr(host, "adaptive_confidence_weight", 0.1))
    threshold = float(getattr(host, "adaptive_score_threshold", 0.02))
    top_k = int(getattr(host, "adaptive_top_k", 3))
    disable_if_low_score = bool(getattr(host, "adaptive_disable_if_low_score", False))

    stats = {}
    confidence_raw_by_layer = {}
    candidate_layers = list(getattr(host, "candidate_aux_layers", []))
    for layer in candidate_layers:
        start, used_start_fallback = _probe_window_loss(host, layer, "start")
        end, used_end_fallback = _probe_window_loss(host, layer, "end")

        if start is None or end is None:
            drop_ratio = 0.0
            confidence_raw = 0.0
        else:
            drop_ratio = (float(start) - float(end)) / (float(start) + eps)
            confidence_raw = 1.0 / (float(end) + eps)

        confidence_raw_by_layer[layer] = confidence_raw
        stats[layer] = {
            "start_loss": None if start is None else float(start),
            "end_loss": None if end is None else float(end),
            "drop_ratio": float(drop_ratio),
            "drop_score": float(max(drop_ratio, 0.0)),
            "confidence_raw": float(confidence_raw),
            "confidence_norm": 0.0,
            "late_bias": float((float(layer) / float(total_layers)) ** gamma),
            "final_score": 0.0,
            "used_start_fallback": bool(used_start_fallback),
            "used_end_fallback": bool(used_end_fallback),
        }

    if confidence_raw_by_layer:
        min_conf = min(confidence_raw_by_layer.values())
        max_conf = max(confidence_raw_by_layer.values())
    else:
        min_conf = 0.0
        max_conf = 0.0

    eligible_scores = {}
    for layer, item in stats.items():
        confidence_norm = (item["confidence_raw"] - min_conf) / (max_conf - min_conf + eps)
        item["confidence_norm"] = float(confidence_norm)

        if score_mode == "loss_drop":
            score = item["drop_score"] * item["late_bias"]
        else:
            score = (item["drop_score"] + confidence_weight * confidence_norm) * item["late_bias"]

        item["final_score"] = float(score)
        if layer >= min_layer:
            eligible_scores[layer] = float(score)

    host.adaptive_layer_stats = stats
    host.adaptive_scores = eligible_scores

    max_score = max(eligible_scores.values()) if eligible_scores else 0.0
    max_confidence = max((stats[layer]["confidence_norm"] for layer in eligible_scores), default=0.0)
    can_select = bool(eligible_scores) and top_k > 0 and max_score >= threshold

    if can_select:
        selected = sorted(
            [layer for layer, _ in sorted(eligible_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        )
        score_sum = sum(eligible_scores[layer] for layer in selected)
        if score_sum <= eps:
            weights = _build_aux_weights(selected, "uniform")
        else:
            weights = {layer: eligible_scores[layer] / score_sum for layer in selected}

        host.aux_layers = selected
        host.selected_aux_layers = selected
        host.aux_loss_weights_by_layer = weights
        host.ds_selection_done = True
        host.adaptive_ds_disabled = False
        host.adaptive_used_fallback = False
        host.effective_aux_loss_weight = float(getattr(host, "aux_loss_weight", 0.1))
        _sync_deep_supervision_config(
            host,
            train_final_head=False,
            task_name=getattr(host, "deep_supervision_config", {}).get("task_name"),
        )
    elif (
        disable_if_low_score
        or top_k <= 0
        or max_confidence < float(getattr(host, "fallback_confidence_threshold", 0.0))
    ):
        host.aux_layers = []
        host.selected_aux_layers = []
        host.aux_loss_weights_by_layer = {}
        host.ds_selection_done = True
        host.adaptive_ds_disabled = True
        host.adaptive_used_fallback = False
        host.effective_aux_loss_weight = 0.0
        _sync_deep_supervision_config(
            host,
            train_final_head=False,
            task_name=getattr(host, "deep_supervision_config", {}).get("task_name"),
        )
        print("Warning: Adaptive DS disabled because probing scores are too low.")
    else:
        fallback_layers = list(getattr(host, "fallback_aux_layers", []))
        if not fallback_layers:
            fallback_layers = [total_layers]
        heads = getattr(host, "auxiliary_heads", None)
        fallback_layers = [layer for layer in fallback_layers if heads is None or str(layer) in heads]
        if not fallback_layers:
            fallback_layers = [total_layers]
        fallback_layers = sorted(set(fallback_layers))
        weights = _build_aux_weights(
            fallback_layers,
            str(getattr(host, "fallback_aux_weight_scheme", "linear_increase")),
        )

        host.aux_layers = fallback_layers
        host.selected_aux_layers = fallback_layers
        host.aux_loss_weights_by_layer = weights
        host.ds_selection_done = True
        host.adaptive_ds_disabled = False
        host.adaptive_used_fallback = True
        host.effective_aux_loss_weight = (
            float(getattr(host, "aux_loss_weight", 0.1))
            * float(getattr(host, "fallback_aux_loss_weight_scale", 0.5))
        )
        _sync_deep_supervision_config(
            host,
            train_final_head=False,
            task_name=getattr(host, "deep_supervision_config", {}).get("task_name"),
        )
        print("Adaptive scores are low; using fallback late-layer weak supervision instead of disabling DS.")

    print("Probe-then-Adapt DS-SOT selection:")
    print(f"  selected_aux_layers: {getattr(host, 'selected_aux_layers', [])}")
    print(f"  adaptive_aux_weights: {getattr(host, 'aux_loss_weights_by_layer', {})}")
    print(f"  adaptive_used_fallback: {getattr(host, 'adaptive_used_fallback', False)}")
    print(f"  adaptive_ds_disabled: {getattr(host, 'adaptive_ds_disabled', False)}")
    print(f"  effective_aux_loss_weight: {getattr(host, 'effective_aux_loss_weight', 0.0)}")
    print("  layer | start_loss | end_loss | drop_ratio | confidence_raw | confidence_norm | late_bias | final_score | adaptive_weight")
    for layer in candidate_layers:
        item = stats.get(layer, {})
        print(
            f"  {layer} | "
            f"{item.get('start_loss')} | "
            f"{item.get('end_loss')} | "
            f"{item.get('drop_ratio')} | "
            f"{item.get('confidence_raw')} | "
            f"{item.get('confidence_norm')} | "
            f"{item.get('late_bias')} | "
            f"{item.get('final_score')} | "
            f"{getattr(host, 'aux_loss_weights_by_layer', {}).get(layer, 0.0)}"
        )


def _adaptive_log_items(host) -> Dict[str, float]:
    logs = {
        "adaptive_used_fallback": 1.0 if bool(getattr(host, "adaptive_used_fallback", False)) else 0.0,
        "adaptive_ds_disabled": 1.0 if bool(getattr(host, "adaptive_ds_disabled", False)) else 0.0,
        "effective_aux_loss_weight": float(
            getattr(host, "effective_aux_loss_weight", getattr(host, "aux_loss_weight", 0.1))
        ),
    }
    for layer, score in getattr(host, "adaptive_scores", {}).items():
        logs[f"adaptive_score_layer_{layer}"] = float(score)
    for layer, weight in getattr(host, "aux_loss_weights_by_layer", {}).items():
        logs[f"adaptive_weight_layer_{layer}"] = float(weight)
    return logs


def compute_deep_supervision_loss(model, hidden_states, label_ids, progress: Optional[float] = None):
    host = _get_ds_host(model)
    if not bool(getattr(host, "use_deep_supervision", False)):
        return None, {}
    if hidden_states is None:
        raise RuntimeError(
            "Deep supervision is enabled but hidden_states were not returned or captured. "
            "Set output_hidden_states=True for the Mamba forward path, or use the Trainer's Mamba block hook fallback."
        )
    if label_ids is None:
        return None, {}

    heads = getattr(host, "auxiliary_heads", None)
    if heads is None:
        return None, {}

    progress = 0.0 if progress is None else max(0.0, min(1.0, float(progress)))
    logs = {
        "progress": progress,
        "selected_aux_layers": _layers_to_text(list(getattr(host, "selected_aux_layers", []))),
    }

    if bool(getattr(host, "ds_adaptive", False)):
        logs.update(_adaptive_log_items(host))
        probe_ratio = float(getattr(host, "probe_ratio", 0.15))
        if not bool(getattr(host, "ds_selection_done", False)) and progress >= probe_ratio:
            _select_adaptive_aux_layers(host)
            logs["selected_aux_layers"] = _layers_to_text(list(getattr(host, "selected_aux_layers", [])))
            logs.update(_adaptive_log_items(host))

        if not bool(getattr(host, "ds_selection_done", False)):
            layers = list(getattr(host, "candidate_aux_layers", []))
            logs["active_aux_layers"] = _layers_to_text(layers)
            logs["probe_progress"] = 1.0 if probe_ratio <= 0 else max(0.0, min(1.0, progress / probe_ratio))
            layer_losses = _compute_aux_layer_losses(host, hidden_states, label_ids, layers)
            if not layer_losses:
                return None, logs

            _update_probe_stats(host, layer_losses, progress)
            aux_total = sum(layer_losses.values()) / len(layer_losses)
            logs["probe_aux_loss"] = aux_total.detach()
            logs["aux_loss_total"] = aux_total.detach()
            for layer, loss in layer_losses.items():
                logs[f"aux_loss_layer_{layer}"] = loss.detach()
                logs[f"probe_loss_ema_layer_{layer}"] = float(host.probe_loss_ema[layer])
            return aux_total, logs

        if bool(getattr(host, "adaptive_ds_disabled", False)):
            logs["active_aux_layers"] = ""
            logs["adaptive_ds_disabled"] = 1.0
            logs.update(_adaptive_log_items(host))
            return None, logs

    aux_layers = list(getattr(host, "aux_layers", []))
    logs["active_aux_layers"] = _layers_to_text(aux_layers)
    logs["selected_aux_layers"] = _layers_to_text(list(getattr(host, "selected_aux_layers", aux_layers)))
    if not aux_layers:
        return None, logs

    layer_losses = _compute_aux_layer_losses(host, hidden_states, label_ids, aux_layers)
    if not layer_losses:
        return None, logs

    aux_total = None
    for layer, layer_loss in layer_losses.items():
        weight = float(getattr(host, "aux_loss_weights_by_layer", {}).get(layer, 0.0))
        weighted_loss = layer_loss * weight
        aux_total = weighted_loss if aux_total is None else aux_total + weighted_loss
        logs[f"aux_loss_layer_{layer}"] = layer_loss.detach()

    if aux_total is None:
        return None, logs

    logs["aux_loss_total"] = aux_total.detach()
    if bool(getattr(host, "ds_adaptive", False)):
        logs.update(_adaptive_log_items(host))
    return aux_total, logs


def set_latest_loss_dict(model, values: Dict[str, torch.Tensor]) -> None:
    host = _get_ds_host(model)
    host.latest_loss_dict = {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in values.items()
    }
