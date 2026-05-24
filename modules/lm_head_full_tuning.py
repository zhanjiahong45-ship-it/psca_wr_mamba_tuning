from typing import Iterable, Optional, Tuple

from torch import nn


_LM_HEAD_PARENT_PATHS = (
    "",
    "backbone",
    "model",
    "base_model",
    "model.model",
    "model.base_model",
    "base_model.model",
    "base_model.model.base_model",
)

_EMBEDDING_PATHS = (
    "backbone.embedding",
    "embedding",
    "word_embeddings",
    "model.backbone.embedding",
    "model.embedding",
    "model.word_embeddings",
    "base_model.backbone.embedding",
    "base_model.embedding",
    "base_model.word_embeddings",
    "base_model.model.backbone.embedding",
    "base_model.model.embedding",
    "base_model.model.word_embeddings",
)


def _get_attr_path(root: nn.Module, path: str):
    if path == "":
        return root

    current = root
    for attr in path.split("."):
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _module_name(model: nn.Module, module: nn.Module) -> Optional[str]:
    for name, candidate in model.named_modules():
        if candidate is module:
            return name
    return None


def _iter_lm_head_candidates(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    seen = set()
    for parent_path in _LM_HEAD_PARENT_PATHS:
        parent = _get_attr_path(model, parent_path)
        if parent is None:
            continue

        getter = getattr(parent, "get_output_embeddings", None)
        if callable(getter):
            candidate = getter()
            if candidate is not None and hasattr(candidate, "weight") and id(candidate) not in seen:
                seen.add(id(candidate))
                name = _module_name(model, candidate)
                yield name or f"{parent_path}.get_output_embeddings()", candidate

        candidate = getattr(parent, "lm_head", None)
        if candidate is not None and hasattr(candidate, "weight") and id(candidate) not in seen:
            seen.add(id(candidate))
            name = _module_name(model, candidate)
            yield name or f"{parent_path}.lm_head".strip("."), candidate


def _find_lm_head(model: nn.Module) -> Tuple[str, nn.Module]:
    for name, module in _iter_lm_head_candidates(model):
        if isinstance(module.weight, nn.Parameter):
            return name, module
    raise RuntimeError(
        "[LM_HEAD_FULL] Could not find an explicit lm_head.weight. "
        "Checked model.lm_head, model.backbone.lm_head, model.model.lm_head, "
        "and model.base_model.lm_head style paths."
    )


def _find_input_embedding(model: nn.Module) -> Optional[nn.Module]:
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embedding = getter()
        if embedding is not None and hasattr(embedding, "weight"):
            return embedding

    for path in _EMBEDDING_PATHS:
        candidate = _get_attr_path(model, path)
        if candidate is not None and hasattr(candidate, "weight"):
            return candidate
    return None


def _is_tied_weight(lm_head: nn.Module, embedding: Optional[nn.Module]) -> bool:
    if embedding is None or not hasattr(embedding, "weight"):
        return False
    if lm_head.weight is embedding.weight:
        return True
    return lm_head.weight.data_ptr() == embedding.weight.data_ptr()


def _trainable_parameters(model: nn.Module):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def enable_lm_head_full_tuning(model: nn.Module) -> nn.Module:
    for _, param in model.named_parameters():
        param.requires_grad = False

    _, lm_head = _find_lm_head(model)
    embedding = _find_input_embedding(model)

    if _is_tied_weight(lm_head, embedding):
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())

    if embedding is not None:
        embedding.weight.requires_grad = False

    if getattr(lm_head, "bias", None) is not None:
        lm_head.bias.requires_grad = False

    lm_head.weight.requires_grad = True

    trainable_params = _trainable_parameters(model)
    trainable_names = [name for name, _ in trainable_params]
    if len(trainable_names) != 1 or not trainable_names[0].endswith("lm_head.weight"):
        print("[LM_HEAD_FULL] Invalid trainable parameters:")
        for name, param in trainable_params:
            print(f"    {name} {tuple(param.shape)} {param.numel()}")
        raise RuntimeError(
            "[LM_HEAD_FULL] Expected only lm_head.weight to be trainable, "
            f"got {trainable_names}."
        )

    total_params = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for _, param in trainable_params)
    trainable_ratio = (100.0 * trainable_count / total_params) if total_params > 0 else 0.0

    print("[LM_HEAD_FULL] Enabled lm_head-only full tuning.")
    print("[LM_HEAD_FULL] Trainable parameters:")
    for name, param in trainable_params:
        print(f"    {name} {tuple(param.shape)} {param.numel()}")
    print(f"[LM_HEAD_FULL] Total trainable params: {trainable_count}")
    print(f"[LM_HEAD_FULL] Total params: {total_params}")
    print(f"[LM_HEAD_FULL] Trainable ratio: {trainable_ratio:.6f}%")

    return model
