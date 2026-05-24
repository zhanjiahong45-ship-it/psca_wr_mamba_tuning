import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import concatenate_datasets, load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler

from modules import (
    MambaLMHeadModelPeft,
    MambaPeft,
    get_mamba_peft_model,
    get_trainable_parameters_ratio,
    load_mamba,
    print_trainable_parameter_names,
)
from modules.suffix_tuning import (
    clear_keyaware_context,
    collect_keyaware_context_stats,
    set_keyaware_context,
)


PROMPTS = {
    "rte": "Determine if the given pair of sentences displays entailment or not_entailment. Respond with '0' if entailment or '1' if not_entailment: ",
    "cola": "Review the sentence below and identify whether its grammar is Unacceptable or Acceptable. Respond with '0' if Unacceptable or '1' if Acceptable: ",
    "mrpc": "Can the given sentences be considered semantically identical? Respond with '0' if not_equivalent or '1' if equivalent: ",
    "sst2": "Read the provided excerpt and choose between negative and positive to describe its sentiment. Respond with '0' if negative or '1' if positive: ",
    "qnli": "Consider the context and question, and indicate if the answer can be logically deduced from the context. Respond with '0' if entailment or '1' if not_entailment: ",
    "qqp": "Can these two statements be considered equal in meaning? Respond with '0' if not_equivalent or '1' if equivalent: ",
    "mnli": "Assess the connection between the following sentences and classify it as entailment, neutral, or contradiction. Respond with '0' if entailment, '1' if neutral or '2' if contradiction: ",
    "stsb": "Rate the semantic similarity of the following sentence pair from '0' to '5'. Respond with one number: ",
}

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 6,
}

DEFAULT_LR = {
    "cola": 2e-4,
    "mrpc": 2e-4,
    "mnli": 4e-4,
    "qnli": 1e-4,
    "qqp": 4e-5,
    "rte": 1e-3,
    "sst2": 1e-4,
    "stsb": 2e-4,
}

SOT_CONFIG_ROOT = ROOT / "cfg" / "final" / "exps" / "mamba-130m"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_sot_config_path(task_name: str) -> Path:
    return SOT_CONFIG_ROOT / f"glue_{task_name}" / "state_tuning.yaml"


def load_sot_training_config(args):
    cfg_path = Path(args.sot_config_path) if args.sot_config_path else default_sot_config_path(args.task_name)
    if not cfg_path.exists():
        if args.task_name == "stsb":
            print(f"Warning: no SOT config found for STS-B at {cfg_path}; using script defaults for this task.")
            return None, None
        raise FileNotFoundError(f"SOT config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg_path


def apply_sot_training_config(args):
    cfg, cfg_path = load_sot_training_config(args)
    if cfg is None:
        if args.learning_rate is None:
            args.learning_rate = DEFAULT_LR[args.task_name]
        return

    if "model" in cfg:
        args.model_name_or_path = cfg["model"]
    if "learning_rate" in cfg:
        args.learning_rate = float(cfg["learning_rate"])
    if "batch_size" in cfg:
        args.batch_size = int(cfg["batch_size"])
    if "num_epochs" in cfg:
        args.num_train_epochs = int(cfg["num_epochs"])
    if "prec" in cfg:
        args.prec = cfg["prec"]
    args.aligned_sot_config_path = str(cfg_path)
    print(
        "Aligned training config with SOT "
        f"({cfg_path}): lr={args.learning_rate}, batch_size={args.batch_size}, "
        f"num_train_epochs={args.num_train_epochs}, model={args.model_name_or_path}, prec={args.prec}"
    )


def resolve_device(device: str) -> torch.device:
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(device)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def get_label_token_ids(tokenizer, task_name: str) -> List[int]:
    ids = []
    for label in range(NUM_LABELS[task_name]):
        token_ids = tokenizer.encode(str(label))
        if len(token_ids) != 1:
            raise ValueError(f"Expected label {label!r} to map to one token, got {token_ids}")
        ids.append(token_ids[0])
    return ids


def label_to_text(task_name: str, label) -> str:
    if task_name == "stsb":
        return str(int(round(float(label))))
    return str(int(label))


@dataclass
class PromptExample:
    input_ids: torch.Tensor
    label_ids: torch.Tensor
    gold_label: float
    split_name: str


class PromptGlueDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        task_name: str,
        split: str,
        max_length: int,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.split = split
        self.max_length = max_length
        self.examples = self._build_examples(max_samples=max_samples)

        if len(self.examples) == 0:
            raise RuntimeError(f"No examples left for {task_name}/{split}; increase --max_length.")

    def _load_hf_split(self):
        if self.task_name == "mnli" and self.split == "validation":
            matched = load_dataset("nyu-mll/glue", "mnli_matched")["validation"]
            mismatched = load_dataset("nyu-mll/glue", "mnli_mismatched")["validation"]
            matched = matched.add_column("_split_name", ["validation_matched"] * len(matched))
            mismatched = mismatched.add_column("_split_name", ["validation_mismatched"] * len(mismatched))
            return concatenate_datasets([matched, mismatched])

        split_name = {"train": "train", "validation": "validation", "test": "test"}[self.split]
        dataset = load_dataset("nyu-mll/glue", self.task_name)[split_name]
        return dataset.add_column("_split_name", [split_name] * len(dataset))

    def _format_input(self, sample) -> str:
        key1, key2 = TASK_TO_KEYS[self.task_name]
        text = sample[key1]
        if key2 is not None:
            sep = self.tokenizer.sep_token if self.tokenizer.sep_token is not None else "\n"
            text = text + sep + sample[key2]
        return PROMPTS[self.task_name] + text + self.tokenizer.sep_token

    def _build_examples(self, max_samples: Optional[int]) -> List[PromptExample]:
        hf_dataset = self._load_hf_split()
        examples = []

        for idx, sample in enumerate(hf_dataset):
            if max_samples is not None and len(examples) >= max_samples:
                break

            prompt_ids = torch.LongTensor(self.tokenizer.encode(self._format_input(sample)))
            label_ids_raw = torch.LongTensor(self.tokenizer.encode(label_to_text(self.task_name, sample["label"])))
            ids = torch.cat([prompt_ids, label_ids_raw])

            if ids.numel() <= 1 or (self.max_length is not None and ids.numel() - 1 > self.max_length):
                continue

            input_ids = ids[:-1]
            label_ids = torch.full_like(input_ids, -100)
            label_ids[-label_ids_raw.numel():] = ids[-label_ids_raw.numel():]

            examples.append(
                PromptExample(
                    input_ids=input_ids,
                    label_ids=label_ids,
                    gold_label=float(sample["label"]),
                    split_name=sample["_split_name"],
                )
            )

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class PromptCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: List[PromptExample]) -> Dict:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [ex.input_ids for ex in examples],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        label_ids = torch.nn.utils.rnn.pad_sequence(
            [ex.label_ids for ex in examples],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "gold_labels": torch.tensor([ex.gold_label for ex in examples], dtype=torch.float32),
            "split_names": [ex.split_name for ex in examples],
        }


class RecallProbe(nn.Module):
    def __init__(self, hidden_dim: int, embedding_weight: torch.Tensor):
        super().__init__()
        self.proj = nn.Linear(
            hidden_dim,
            embedding_weight.shape[1],
            bias=False,
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
        )
        self.embedding_weight = embedding_weight.detach()
        self.embedding_weight.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.proj(h)
        embedding_weight = self.embedding_weight.to(device=z.device, dtype=z.dtype)
        return (z @ embedding_weight.t()).float()


def get_embedding_weight(model) -> torch.Tensor:
    for module in model.modules():
        if hasattr(module, "backbone") and hasattr(module.backbone, "embedding"):
            return module.backbone.embedding.weight
    if hasattr(model, "model") and hasattr(model.model, "word_embeddings"):
        return model.model.word_embeddings.weight
    raise AttributeError("Could not find Mamba embedding weight on the model.")


def get_hidden_dim(model) -> int:
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "d_model"):
            return module.config.d_model
    return get_embedding_weight(model).shape[1]


def freeze_model(model) -> Dict[str, bool]:
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad = False
    return requires_grad


def restore_requires_grad(model, requires_grad: Dict[str, bool]):
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad.get(name, param.requires_grad)


def get_mamba_backbone_model(model):
    candidates = [model]
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:
            pass

    idx = 0
    while idx < len(candidates):
        candidate = candidates[idx]
        if hasattr(candidate, "backbone") and hasattr(candidate.backbone, "layers"):
            return candidate
        for attr in ("model", "base_model"):
            child = getattr(candidate, attr, None)
            if child is not None and child not in candidates:
                candidates.append(child)
        idx += 1
    raise AttributeError("Could not find a Mamba backbone with layers on the model.")


def get_target_hidden_module(model, target_layer: int):
    backbone_model = get_mamba_backbone_model(model)
    if target_layer == 0:
        return backbone_model.backbone.embedding

    layers = backbone_model.backbone.layers
    layer_idx = target_layer - 1
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"target_layer={target_layer} is invalid for {len(layers)} Mamba layers.")
    return layers[layer_idx]


def forward_with_hidden(model, input_ids, target_layer: int):
    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = get_target_hidden_module(model, target_layer).register_forward_hook(hook)
    try:
        outputs = model(input_ids)
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError(f"Forward hook did not capture target_layer={target_layer}.")
    return outputs, captured["hidden"]


def valid_token_mask(input_ids, attention_mask, special_token_ids: List[int]) -> torch.Tensor:
    mask = attention_mask.bool()
    if special_token_ids:
        special = torch.tensor(special_token_ids, device=input_ids.device, dtype=input_ids.dtype)
        mask = mask & ~torch.isin(input_ids, special)
    return mask


def token_cross_entropy(logits, target_ids, mask):
    if mask.sum().item() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target_ids[mask])


def probe_eval(
    model,
    probe: RecallProbe,
    dataloader: DataLoader,
    args,
    special_token_ids: List[int],
    unigram_probs: Optional[torch.Tensor],
):
    model.eval()
    probe.eval()
    total_loss = 0.0
    total_tokens = 0
    total_top1 = 0
    total_top5 = 0
    total_unigram = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Probe eval")):
            if args.probe_eval_batches and batch_idx >= args.probe_eval_batches:
                break
            batch = move_batch_to_device(batch, args.torch_device)
            outputs, h = forward_with_hidden(model, batch["input_ids"], args.target_layer)
            logits = probe(h)
            mask = valid_token_mask(batch["input_ids"], batch["attention_mask"], special_token_ids)
            if mask.sum().item() == 0:
                continue

            targets = batch["input_ids"][mask]
            logits_valid = logits[mask]
            loss_sum = F.cross_entropy(logits_valid, targets, reduction="sum")
            total_loss += float(loss_sum.detach().cpu())
            total_tokens += int(targets.numel())

            top5 = logits_valid.topk(k=min(5, logits_valid.shape[-1]), dim=-1).indices
            total_top1 += int((top5[:, 0] == targets).sum().detach().cpu())
            total_top5 += int((top5 == targets.unsqueeze(-1)).any(dim=-1).sum().detach().cpu())

            if unigram_probs is not None:
                token_probs = unigram_probs.to(targets.device)[targets].clamp_min(1e-12)
                total_unigram += float((-token_probs.log()).sum().detach().cpu())

    val_ce = total_loss / max(total_tokens, 1)
    unigram_ce = total_unigram / max(total_tokens, 1) if unigram_probs is not None else None
    return {
        "val_token_ce": val_ce,
        "random_ce": math.log(probe.embedding_weight.shape[0]),
        "unigram_ce": unigram_ce if unigram_ce is not None else math.log(probe.embedding_weight.shape[0]),
        "val_top1_acc": total_top1 / max(total_tokens, 1),
        "val_top5_acc": total_top5 / max(total_tokens, 1),
    }


def train_recall_probe_for_task(model, train_loader, val_loader, args, output_dir: Path, tokenizer):
    probe_dir = output_dir / "recall_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / "recall_probe.pt"
    metrics_path = probe_dir / "probe_metrics.json"

    if args.skip_probe_training:
        if not probe_path.exists():
            raise FileNotFoundError(f"--skip_probe_training true but {probe_path} does not exist.")
        embedding_weight = get_embedding_weight(model)
        probe = RecallProbe(get_hidden_dim(model), embedding_weight)
        probe.load_state_dict(torch.load(probe_path, map_location=args.torch_device))
        probe.eval()
        return probe

    if probe_path.exists() and not args.always_train_probe:
        embedding_weight = get_embedding_weight(model)
        probe = RecallProbe(get_hidden_dim(model), embedding_weight)
        probe.load_state_dict(torch.load(probe_path, map_location=args.torch_device))
        probe.eval()
        return probe

    print("Training recall probe from scratch for this task.")
    original_requires_grad = freeze_model(model)
    model.eval()

    embedding_weight = get_embedding_weight(model)
    probe = RecallProbe(get_hidden_dim(model), embedding_weight).to(args.torch_device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr)
    special_token_ids = list(set(tokenizer.all_special_ids + [tokenizer.pad_token_id]))
    vocab_size = embedding_weight.shape[0]
    unigram_counts = torch.zeros(vocab_size, dtype=torch.float64)

    train_iter = iter(train_loader)
    for step in tqdm(range(args.probe_steps), desc="Probe train"):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = move_batch_to_device(batch, args.torch_device)
        with torch.no_grad():
            outputs, h = forward_with_hidden(model, batch["input_ids"], args.target_layer)

        logits = probe(h)
        mask = valid_token_mask(batch["input_ids"], batch["attention_mask"], special_token_ids)
        loss = token_cross_entropy(logits, batch["input_ids"], mask)

        if mask.sum().item() > 0:
            ids = batch["input_ids"][mask].detach().cpu()
            unigram_counts += torch.bincount(ids, minlength=vocab_size).double()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    unigram_probs = (unigram_counts + 1.0) / (unigram_counts.sum() + vocab_size)
    metrics = probe_eval(model, probe, val_loader, args, special_token_ids, unigram_probs)
    metrics.update({"target_layer": args.target_layer, "task_name": args.task_name})

    torch.save(probe.state_dict(), probe_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if metrics["val_token_ce"] >= min(metrics["random_ce"], metrics["unigram_ce"]):
        print(
            "WARNING: recall probe CE is not clearly below random/unigram baselines. "
            "Continuing training as requested."
        )

    restore_requires_grad(model, original_requires_grad)
    probe.eval()
    for param in probe.parameters():
        param.requires_grad = False
    return probe


def select_forgotten_candidates(r, valid, args):
    selected = torch.zeros_like(valid, dtype=torch.bool)
    batch_size = valid.shape[0]

    for batch_idx in range(batch_size):
        idx = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue

        vals = r[batch_idx, idx].float()
        if args.key_select_mode == "topk":
            k = max(1, int(math.ceil(args.forgotten_ratio * idx.numel())))
            pick_local = torch.topk(vals, k=min(k, vals.numel())).indices
            selected[batch_idx, idx[pick_local]] = True
        elif args.key_select_mode == "percentile_band":
            low = torch.quantile(vals, args.key_percentile_low)
            high = torch.quantile(vals, args.key_percentile_high)
            picked = idx[(vals >= low) & (vals <= high)]
            if picked.numel() == 0:
                k = max(1, int(math.ceil(args.forgotten_ratio * idx.numel())))
                picked = idx[torch.topk(vals, k=min(k, vals.numel())).indices]
            selected[batch_idx, picked] = True
        else:
            raise ValueError(f"Unknown key_select_mode: {args.key_select_mode}")

    return selected


def randomize_selected_mask(selected, valid):
    randomized = torch.zeros_like(selected)
    for batch_idx in range(selected.shape[0]):
        k = int(selected[batch_idx].sum().item())
        idx = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
        if k == 0 or idx.numel() == 0:
            continue
        perm = torch.randperm(idx.numel(), device=idx.device)[: min(k, idx.numel())]
        randomized[batch_idx, idx[perm]] = True
    return randomized


def masked_mean(values, mask, default=0.0):
    if mask is None or mask.sum().item() == 0:
        return default
    return float(values[mask].float().mean().detach().cpu())


def masked_quantile(values, mask, q, default=0.0):
    if mask is None or mask.sum().item() == 0:
        return default
    return float(torch.quantile(values[mask].float(), q).detach().cpu())


def compute_key_token_weights(
    task_loss,
    h,
    input_ids,
    attention_mask,
    probe: RecallProbe,
    tokenizer,
    args,
):
    delta = args.delta
    batch_size, seq_len, _ = h.shape
    alpha_full = torch.zeros(batch_size, seq_len, device=h.device, dtype=h.dtype)

    if seq_len <= delta:
        return alpha_full, {
            "key_recall_loss": h.sum() * 0.0,
            "raw_weighted_ce": 0.0,
            "mean_r": 0.0,
            "selected_r_mean": 0.0,
            "selected_r_p50": 0.0,
            "selected_r_p90": 0.0,
            "all_forgotten_mean_cos": 0.0,
            "all_forgotten_pos_cos_ratio": 0.0,
            "active_weight_ratio": 0.0,
            "active_mean_cos": 0.0,
            "active_weighted_predicted_gain": 0.0,
            "num_selected_tokens": 0,
            "num_active_tokens": 0,
        }

    h_local = h[:, :-delta, :]
    h_future = h[:, delta:, :]
    x_i = input_ids[:, :-delta]
    valid = attention_mask[:, :-delta].bool() & attention_mask[:, delta:].bool()
    special_token_ids = list(set(tokenizer.all_special_ids + [tokenizer.pad_token_id]))
    if special_token_ids:
        special = torch.tensor(special_token_ids, device=input_ids.device, dtype=input_ids.dtype)
        valid = valid & ~torch.isin(x_i, special)

    logits_local = probe(h_local)
    logits_future = probe(h_future)
    logp_local = F.log_softmax(logits_local, dim=-1).gather(-1, x_i.unsqueeze(-1)).squeeze(-1)
    logp_future = F.log_softmax(logits_future, dim=-1).gather(-1, x_i.unsqueeze(-1)).squeeze(-1)
    r = logp_local - logp_future
    unit_recall_loss = -logp_future

    selected = select_forgotten_candidates(r.detach(), valid, args)
    if args.key_weight_mode == "random":
        selected = randomize_selected_mask(selected, valid)

    num_selected = int(selected.sum().detach().cpu())
    if num_selected == 0:
        return alpha_full, {
            "key_recall_loss": h.sum() * 0.0,
            "raw_weighted_ce": 0.0,
            "mean_r": masked_mean(r, valid),
            "selected_r_mean": 0.0,
            "selected_r_p50": 0.0,
            "selected_r_p90": 0.0,
            "all_forgotten_mean_cos": 0.0,
            "all_forgotten_pos_cos_ratio": 0.0,
            "active_weight_ratio": 0.0,
            "active_mean_cos": 0.0,
            "active_weighted_predicted_gain": 0.0,
            "num_selected_tokens": 0,
            "num_active_tokens": 0,
        }

    cos = torch.zeros_like(r)
    if args.key_weight_mode == "alignment":
        g_task = torch.autograd.grad(
            task_loss,
            h,
            retain_graph=True,
            create_graph=False,
        )[0]
        selected_loss = (unit_recall_loss * selected.to(unit_recall_loss.dtype)).sum()
        g_rec = torch.autograd.grad(
            selected_loss,
            h,
            retain_graph=True,
            create_graph=False,
        )[0]
        cos = F.cosine_similarity(g_task[:, delta:, :], g_rec[:, delta:, :], dim=-1, eps=1e-8)
        alpha_short = (selected.to(h.dtype) * cos.clamp_min(0.0)).detach()
    elif args.key_weight_mode == "forgotten_only":
        alpha_short = selected.to(h.dtype).detach()
    elif args.key_weight_mode == "random":
        alpha_short = selected.to(h.dtype).detach()
    else:
        raise ValueError(f"Unknown key_weight_mode: {args.key_weight_mode}")

    active = alpha_short > 0
    active_count = int(active.sum().detach().cpu())
    active_ratio = active_count / max(num_selected, 1)
    if active_ratio < args.min_active_weight_ratio:
        alpha_short = torch.zeros_like(alpha_short)
        active = alpha_short > 0
        active_count = 0

    alpha_full[:, :-delta] = alpha_short
    raw_weighted_ce = (alpha_short * unit_recall_loss).sum()
    key_recall_loss = raw_weighted_ce / max(num_selected, 1)

    stats = {
        "key_recall_loss": key_recall_loss,
        "raw_weighted_ce": float(raw_weighted_ce.detach().cpu()),
        "mean_r": masked_mean(r, valid),
        "selected_r_mean": masked_mean(r, selected),
        "selected_r_p50": masked_quantile(r, selected, 0.5),
        "selected_r_p90": masked_quantile(r, selected, 0.9),
        "all_forgotten_mean_cos": masked_mean(cos, selected),
        "all_forgotten_pos_cos_ratio": masked_mean((cos > 0).float(), selected),
        "active_weight_ratio": active_count / max(num_selected, 1),
        "active_mean_cos": masked_mean(cos, active),
        "active_weighted_predicted_gain": float((alpha_short * r.detach()).sum().detach().cpu() / max(active_count, 1)),
        "num_selected_tokens": num_selected,
        "num_active_tokens": active_count,
    }
    return alpha_full.detach(), stats


def compute_lm_loss(logits, label_ids):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), label_ids.reshape(-1), ignore_index=-100)


def peft_config_for_method(method: str) -> Dict:
    bias_type = {
        "sot": "SILU_Z_C",
        "dsot": "DYNAMIC_SILU_Z_C",
        "keyaware_dsot": "KEY_AWARE_DYNAMIC_SILU_Z_C",
    }[method]
    return {
        "peft_type": "SUFFIX_TUNING",
        "bias_init": "ZERO",
        "bias_type": bias_type,
    }


def model_forward_for_method(model, batch, probe, tokenizer, args, train: bool, alpha_label_ids=None):
    method = args.method
    if method != "keyaware_dsot":
        clear_keyaware_context(model)
        outputs = model(batch["input_ids"])
        task_loss = compute_lm_loss(outputs.logits, batch["label_ids"])
        return task_loss, outputs.logits, {
            "task_loss": task_loss,
            "key_recall_loss": task_loss.detach() * 0.0,
            "lambda_times_key_recall_loss": 0.0,
            "raw_weighted_ce": 0.0,
        }

    clear_keyaware_context(model)
    first_outputs, h = forward_with_hidden(model, batch["input_ids"], args.target_layer)
    label_ids_for_alpha = alpha_label_ids if alpha_label_ids is not None else batch["label_ids"]
    task_loss_for_alpha = compute_lm_loss(first_outputs.logits, label_ids_for_alpha)

    alpha, key_stats = compute_key_token_weights(
        task_loss_for_alpha,
        h,
        batch["input_ids"],
        batch["attention_mask"],
        probe,
        tokenizer,
        args,
    )

    set_keyaware_context(model, alpha, beta_key_context=args.beta_key_context, eps=args.key_context_eps)
    second_outputs = model(batch["input_ids"])
    context_stats = collect_keyaware_context_stats(model)
    task_loss = compute_lm_loss(second_outputs.logits, batch["label_ids"])

    key_recall_loss = key_stats.get("key_recall_loss", task_loss.detach() * 0.0)
    if args.lambda_key_recall > 0:
        loss = task_loss + args.lambda_key_recall * key_recall_loss
    else:
        loss = task_loss

    metrics = {
        "task_loss": task_loss,
        "key_recall_loss": key_recall_loss,
        "lambda_times_key_recall_loss": float((args.lambda_key_recall * key_recall_loss).detach().cpu()),
        **{k: v for k, v in key_stats.items() if k != "key_recall_loss"},
        **context_stats,
    }
    return loss, second_outputs.logits, metrics


def predictions_from_logits(logits, label_ids, choice_ids, task_name):
    predictions = []
    positions = (label_ids != -100).float().argmax(dim=1)
    choice = torch.tensor(choice_ids, device=logits.device, dtype=torch.long)

    for batch_idx, pos in enumerate(positions.tolist()):
        scores = logits[batch_idx, pos, choice].float()
        if task_name == "stsb":
            probs = F.softmax(scores, dim=-1)
            values = torch.arange(len(choice_ids), device=logits.device, dtype=probs.dtype)
            predictions.append(float((probs * values).sum().detach().cpu()))
        else:
            predictions.append(int(scores.argmax().detach().cpu()))
    return predictions


def pseudo_label_ids_from_logits(logits, label_ids, choice_ids):
    pseudo_label_ids = torch.full_like(label_ids, -100)
    positions = (label_ids != -100).float().argmax(dim=1)
    choice = torch.tensor(choice_ids, device=logits.device, dtype=torch.long)

    for batch_idx, pos in enumerate(positions.tolist()):
        scores = logits[batch_idx, pos, choice].float()
        predicted_choice = choice[scores.argmax()]
        pseudo_label_ids[batch_idx, pos] = predicted_choice
    return pseudo_label_ids


def compute_glue_metrics(task_name, predictions, references, split_names):
    if task_name == "mnli":
        metrics = {}
        refs = np.array(references, dtype=np.int64)
        preds = np.array(predictions, dtype=np.int64)
        splits = np.array(split_names)
        matched = splits == "validation_matched"
        mismatched = splits == "validation_mismatched"
        metrics["validation_matched_accuracy"] = float((preds[matched] == refs[matched]).mean()) if matched.any() else 0.0
        metrics["validation_mismatched_accuracy"] = float((preds[mismatched] == refs[mismatched]).mean()) if mismatched.any() else 0.0
        metrics["metric_main"] = 0.5 * (metrics["validation_matched_accuracy"] + metrics["validation_mismatched_accuracy"])
        return metrics

    metric = evaluate.load("glue", task_name)
    if task_name == "stsb":
        metrics = metric.compute(predictions=predictions, references=references)
        if "spearmanr" in metrics:
            metrics["spearman"] = metrics["spearmanr"]
        metrics["metric_main"] = float(metrics.get("pearson", 0.0))
        return metrics

    refs = [int(x) for x in references]
    metrics = metric.compute(predictions=predictions, references=refs)
    accuracy = float(np.mean(np.array(predictions) == np.array(refs)))
    metrics.setdefault("accuracy", accuracy)
    if task_name == "cola":
        metrics["metric_main"] = float(metrics.get("matthews_correlation", 0.0))
    elif task_name in {"mrpc", "qqp"}:
        metrics["metric_main"] = float(metrics.get("f1", 0.0))
    else:
        metrics["metric_main"] = float(metrics.get("accuracy", accuracy))
    return metrics


def evaluate_model(model, dataloader, probe, tokenizer, args, choice_ids):
    model.eval()
    predictions = []
    references = []
    split_names = []
    losses = []

    for batch in tqdm(dataloader, desc="Eval"):
        batch = move_batch_to_device(batch, args.torch_device)
        if args.method == "keyaware_dsot":
            with torch.no_grad():
                clear_keyaware_context(model)
                first_outputs = model(batch["input_ids"])
                alpha_label_ids = pseudo_label_ids_from_logits(first_outputs.logits, batch["label_ids"], choice_ids)
            with torch.enable_grad():
                loss, logits, _ = model_forward_for_method(
                    model,
                    batch,
                    probe,
                    tokenizer,
                    args,
                    train=False,
                    alpha_label_ids=alpha_label_ids,
                )
        else:
            with torch.no_grad():
                loss, logits, _ = model_forward_for_method(model, batch, probe, tokenizer, args, train=False)
        predictions.extend(predictions_from_logits(logits, batch["label_ids"], choice_ids, args.task_name))
        references.extend(batch["gold_labels"].detach().cpu().tolist())
        split_names.extend(batch["split_names"])
        losses.append(float(loss.detach().cpu()))

    clear_keyaware_context(model)
    metrics = compute_glue_metrics(args.task_name, predictions, references, split_names)
    metrics["eval_loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def append_jsonl(path: Path, payload: Dict):
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def save_json(path: Path, payload: Dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_model_and_tokenizer(args):
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.prec]
    model_kwargs = dict(
        dtype=dtype,
        device=str(args.torch_device),
        use_fast_path=False,
        mamba_cls=MambaPeft,
        backend=args.backend,
    )
    model_tokenizer = load_mamba(
        args.model_name_or_path,
        cls=MambaLMHeadModelPeft,
        **model_kwargs,
    )
    model, tokenizer = model_tokenizer["model"], model_tokenizer["tokenizer"]
    peft_config = peft_config_for_method(args.method)
    model = get_mamba_peft_model(model, peft_config, no_print=True)
    model.to(args.torch_device)
    return model, tokenizer


def train(args):
    set_seed(args.seed)
    args.torch_device = resolve_device(args.device)
    args.aligned_sot_config_path = None
    if args.align_with_sot_config:
        apply_sot_training_config(args)
    elif args.learning_rate is None:
        args.learning_rate = DEFAULT_LR[args.task_name]

    if args.output_dir is None:
        args.output_dir = str(ROOT / "outputs" / "keyaware_dsot_glue" / args.task_name / f"seed_{args.seed}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    print_trainable_parameter_names(model)
    print(f"Trainable parameter ratio: {get_trainable_parameters_ratio(model):.6f}")

    collator = PromptCollator(tokenizer)
    train_dataset = PromptGlueDataset(
        tokenizer,
        args.task_name,
        "train",
        max_length=args.max_length,
        max_samples=args.max_train_samples,
    )
    val_dataset = PromptGlueDataset(
        tokenizer,
        args.task_name,
        "validation",
        max_length=args.max_length,
        max_samples=args.max_eval_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_data_workers,
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_data_workers,
        collate_fn=collator,
        drop_last=False,
    )
    probe_loader = DataLoader(
        train_dataset,
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=args.num_data_workers,
        collate_fn=collator,
        drop_last=False,
    )
    probe_val_loader = DataLoader(
        val_dataset,
        batch_size=args.probe_batch_size,
        shuffle=False,
        num_workers=args.num_data_workers,
        collate_fn=collator,
        drop_last=False,
    )

    probe = None
    if args.method == "keyaware_dsot":
        probe = train_recall_probe_for_task(model, probe_loader, probe_val_loader, args, output_dir, tokenizer)
        probe.to(args.torch_device)
        probe.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_training_steps = args.num_train_epochs * len(train_loader)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )
    choice_ids = get_label_token_ids(tokenizer, args.task_name)
    train_log_path = output_dir / "train_log.jsonl"
    if train_log_path.exists() and args.overwrite:
        train_log_path.unlink()

    save_json(output_dir / "args.json", {k: str(v) if isinstance(v, torch.device) else v for k, v in vars(args).items()})

    global_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Train epoch {epoch + 1}/{args.num_train_epochs}")
        for batch in progress:
            batch = move_batch_to_device(batch, args.torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, metrics = model_forward_for_method(model, batch, probe, tokenizer, args, train=True)
            loss.backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            clear_keyaware_context(model)

            global_step += 1
            if global_step % args.logging_steps == 0:
                payload = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach().cpu()),
                    "task_loss": float(metrics["task_loss"].detach().cpu()),
                    "method": args.method,
                    "beta_key_context": args.beta_key_context,
                    "key_select_mode": args.key_select_mode,
                    "key_weight_mode": args.key_weight_mode,
                }
                for key, value in metrics.items():
                    if key in {"task_loss", "key_recall_loss"}:
                        payload[key] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                    elif torch.is_tensor(value):
                        payload[key] = float(value.detach().cpu())
                    else:
                        payload[key] = value
                if args.lambda_key_recall <= 0:
                    payload.pop("key_recall_loss", None)
                    payload.pop("lambda_times_key_recall_loss", None)
                    payload.pop("raw_weighted_ce", None)
                append_jsonl(train_log_path, payload)
                progress.set_postfix(loss=payload["loss"], task_loss=payload["task_loss"])

    eval_metrics = evaluate_model(model, val_loader, probe, tokenizer, args, choice_ids)
    metrics_payload = {
        "task_name": args.task_name,
        "method": args.method,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_train_epochs": args.num_train_epochs,
        "aligned_sot_config_path": args.aligned_sot_config_path,
        "beta_key_context": args.beta_key_context,
        "key_select_mode": args.key_select_mode,
        "key_weight_mode": args.key_weight_mode,
        **eval_metrics,
    }
    save_json(output_dir / "metrics.json", metrics_payload)
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")
    print(json.dumps(metrics_payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sot", "dsot", "keyaware_dsot"], default="keyaware_dsot")
    parser.add_argument("--model_name_or_path", default="state-spaces/mamba-130m")
    parser.add_argument("--task_name", choices=sorted(TASK_TO_KEYS.keys()), required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--align_with_sot_config", type=str2bool, default=True)
    parser.add_argument("--sot_config_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--prec", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--num_data_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)

    parser.add_argument("--target_layer", type=int, default=12)
    parser.add_argument("--probe_steps", type=int, default=400)
    parser.add_argument("--probe_lr", type=float, default=5e-4)
    parser.add_argument("--probe_batch_size", type=int, default=16)
    parser.add_argument("--probe_eval_batches", type=int, default=0)
    parser.add_argument("--always_train_probe", type=str2bool, default=True)
    parser.add_argument("--skip_probe_training", type=str2bool, default=False)

    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--key_select_mode", choices=["topk", "percentile_band"], default="percentile_band")
    parser.add_argument("--key_percentile_low", type=float, default=0.7)
    parser.add_argument("--key_percentile_high", type=float, default=0.9)
    parser.add_argument("--forgotten_ratio", type=float, default=0.1)
    parser.add_argument("--key_weight_mode", choices=["alignment", "random", "forgotten_only"], default="alignment")
    parser.add_argument("--min_active_weight_ratio", type=float, default=0.0)
    parser.add_argument("--beta_key_context", type=float, default=0.1)
    parser.add_argument("--key_context_eps", type=float, default=1e-6)
    parser.add_argument("--lambda_key_recall", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
