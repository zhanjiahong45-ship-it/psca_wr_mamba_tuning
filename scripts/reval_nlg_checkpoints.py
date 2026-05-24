#!/usr/bin/env python
"""Run generation evaluation for saved NLG checkpoints.

This is intended for NLG tasks where default Trainer evaluation would keep
large logits tensors in memory. The training run can use ``--skip_eval`` and
save checkpoints, then this script evaluates each checkpoint with generation
metrics independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

from dataset import load_dataset  # noqa: E402
from modules import load_mamba  # noqa: E402
from modules.generation import create_generator  # noqa: E402
from trainer.mamba_trainer import MambaTrainer, MambaTrainingArguments  # noqa: E402
from utils.utils import get_tokenizer_cache_prefix  # noqa: E402


DEFAULT_DART_EVAL_GEN = {
    "max_length": 1024,
    "min_length": 5,
    "num_beams": 5,
}


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _find_checkpoints(output_dir: Path, checkpoint_names: Optional[Iterable[str]] = None) -> List[Path]:
    if checkpoint_names:
        checkpoints = []
        for name in checkpoint_names:
            path = Path(name)
            if not path.is_absolute():
                path = output_dir / path
            checkpoints.append(path)
    else:
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=_checkpoint_step)

    checkpoints = [path for path in checkpoints if (path / "peft.pt").is_file()]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints with peft.pt found under {output_dir}")
    return checkpoints


def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _split_to_dart_raw(split: str) -> Path:
    split = {"validation": "val", "dev": "val"}.get(split, split)
    filenames = {
        "train": "dart-v1.1.1-full-train.json",
        "val": "dart-v1.1.1-full-dev.json",
        "test": "dart-v1.1.1-full-test.json",
    }
    if split not in filenames:
        raise ValueError(f"Unsupported DART split for raw cache preparation: {split}")
    return Path("data") / "dart" / "raw" / "v1.1.1" / filenames[split]


def _linearize_triples(triples: Iterable[Iterable[Any]]) -> str:
    return " | ".join(" : ".join(str(part) for part in triple) for triple in triples)


def _dart_subset_size(data_name: str, split: str) -> Optional[int]:
    parts = data_name.split("_")
    if len(parts) <= 1:
        return None
    subset_size = int(parts[1])
    if split == "val":
        subset_size = int(0.1 * subset_size)
    return subset_size


def _prepare_dart_gen_cache(
    tokenizer: Any,
    *,
    data_name: str,
    split: str,
    max_seqlen: Optional[int] = None,
) -> Optional[Path]:
    """Create DART generation cache from the bundled raw JSON if needed.

    The project caches pre-tokenized examples before adding the prompt prefix.
    This mirrors ``DartDataset(mode="gen")`` closely enough to avoid calling
    ``datasets.load_dataset("dart")`` when the remote HF/datasets version
    cannot load the deprecated dataset script.
    """

    if not data_name.startswith("dart"):
        return None

    split = {"validation": "val", "dev": "val"}.get(split, split)
    subset_size = _dart_subset_size(data_name, split)
    cache_stem = get_tokenizer_cache_prefix(tokenizer) + f"cache_dart_{split}_gen"
    if subset_size is not None:
        cache_stem += f"_{subset_size}"
    cache_file = Path("data") / "dart" / f"{cache_stem}.pkl"
    if cache_file.is_file():
        print(f"[NLG reval] found DART gen cache: {cache_file}")
        return cache_file

    raw_file = _split_to_dart_raw(split)
    if not raw_file.is_file():
        print(f"[NLG reval] DART raw file not found, will fall back to HF loader: {raw_file}")
        return None

    print(f"[NLG reval] building DART gen cache from {raw_file}")
    with open(raw_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    indices = list(range(len(records)))
    if subset_size is not None:
        random.Random(0).shuffle(indices)
        indices = indices[:subset_size]

    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else "<tool_calls>"
    eos_token = tokenizer.eos_token
    data = []
    skipped = 0
    for idx in indices:
        record = records[idx]
        label_refs = [ann["text"] for ann in record["annotations"]]
        if any(sep_token in text for text in label_refs):
            raise ValueError("DART reference contains the tokenizer separator token.")
        input_text = _linearize_triples(record["tripleset"]) + sep_token
        label_text = sep_token.join(label_refs) + eos_token
        input_ids = torch.LongTensor(tokenizer.encode(input_text))
        label_ids = torch.LongTensor(tokenizer.encode(label_text))
        if max_seqlen is not None and input_ids.shape[0] + label_ids.shape[0] > max_seqlen:
            skipped += 1
            continue
        data.append((input_ids, label_ids))

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"[NLG reval] wrote {cache_file} with {len(data)} samples, skipped={skipped}")
    return cache_file


def _generator_kwargs(cfg: Dict[str, Any], data_name: str) -> Dict[str, Any]:
    eval_gen = cfg.get("eval_gen")
    if isinstance(eval_gen, dict):
        return dict(eval_gen)
    if data_name.startswith("dart"):
        print(f"[NLG reval] cfg has no eval_gen; using DART defaults: {DEFAULT_DART_EVAL_GEN}")
        return dict(DEFAULT_DART_EVAL_GEN)
    raise ValueError("Config has no eval_gen; pass a cfg with generation settings or add a task default.")


def _peft_info(model: Any) -> Dict[str, Any]:
    peft_args = getattr(model, "peft_args", None)
    peft_cfg = peft_args.get("peft") if isinstance(peft_args, dict) else None
    if not isinstance(peft_cfg, dict):
        peft_cfg = {}
    return {
        "method": peft_cfg.get("method"),
        "use_psca_wr": peft_cfg.get("use_psca_wr"),
        "psca_rank": peft_cfg.get("psca_rank"),
        "psca_alpha": peft_cfg.get("psca_alpha"),
        "psca_projector_scale": peft_cfg.get("psca_projector_scale"),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    cfg: Dict[str, Any],
    *,
    eval_output_root: Path,
    eval_batch_size: int,
    num_data_workers: int,
) -> Dict[str, Any]:
    print(f"[NLG reval] loading {checkpoint}")
    loaded = load_mamba(str(checkpoint))
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]

    data_name = str(cfg.get("val_data") or cfg.get("data", "dart"))
    split = str(cfg.get("val_data_split", "val"))
    gen_kwargs = _generator_kwargs(cfg, data_name)
    if gen_kwargs.get("num_beams") is not None and eval_batch_size != 1:
        print("[NLG reval] beam search generator supports batch size 1 here; forcing eval_batch_size=1")
        eval_batch_size = 1

    _prepare_dart_gen_cache(
        tokenizer,
        data_name=data_name,
        split=split,
        max_seqlen=cfg.get("eval_max_seqlen"),
    )
    val_data_module = load_dataset(data_name, tokenizer, split, mode="gen", return_module=True)
    generator = create_generator(tokenizer, **gen_kwargs)

    output_dir = eval_output_root / checkpoint.name
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = MambaTrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=eval_batch_size,
        dataloader_num_workers=num_data_workers,
        dataloader_drop_last=False,
        eval_accumulation_steps=1,
        seed=int(cfg.get("seed", 88)),
        report_to=[],
        do_train=False,
        do_eval=True,
        logging_strategy="no",
        save_strategy="no",
        info={
            "task": data_name,
            "method": cfg.get("method", "nlg"),
            "seed": cfg.get("seed", 88),
            "reval_checkpoint": str(checkpoint),
            "eval_gen": gen_kwargs,
        },
    )

    trainer = MambaTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=val_data_module.dataset.compute_metrics,
        data_collator=val_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=generator,
        skip_metrics=False,
    )
    trainer.state.global_step = _checkpoint_step(checkpoint)
    metrics = trainer.evaluate(metric_key_prefix="eval")
    result = {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "step": _checkpoint_step(checkpoint),
        **_peft_info(model),
        "metrics": _jsonable(metrics),
    }

    meteor = metrics.get("eval_meteor")
    bleu = metrics.get("eval_bleu")
    print(
        f"[NLG reval] {checkpoint.name} "
        f"step={result['step']} meteor={meteor if meteor is not None else 'n/a'} "
        f"bleu={bleu if bleu is not None else 'n/a'}"
    )

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_results(results: List[Dict[str, Any]], save_file: Path) -> None:
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")

    csv_file = save_file.with_suffix(".csv")
    metric_keys = sorted({key for row in results for key in row.get("metrics", {}).keys()})
    fieldnames = [
        "checkpoint",
        "step",
        "error",
        "method",
        "use_psca_wr",
        "psca_rank",
        "psca_alpha",
        "psca_projector_scale",
        *metric_keys,
        "checkpoint_path",
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            flat = {key: row.get(key) for key in fieldnames}
            flat.update(row.get("metrics", {}))
            writer.writerow(flat)

    print(f"[NLG reval] wrote {save_file}")
    print(f"[NLG reval] wrote {csv_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory containing checkpoint-* folders.")
    parser.add_argument("--cfg", type=Path, required=True, help="Training cfg used to recover task and eval settings.")
    parser.add_argument("--save_file", type=Path, default=None, help="Default: <output_dir>/gen_reval_results.json")
    parser.add_argument("--checkpoint", action="append", default=None, help="Optional checkpoint name/path; repeatable.")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_data_workers", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="Stop on the first checkpoint evaluation failure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.cfg)
    if args.save_file is None:
        args.save_file = args.output_dir / "gen_reval_results.json"

    checkpoints = _find_checkpoints(args.output_dir, args.checkpoint)
    print(f"[NLG reval] checkpoints: {[path.name for path in checkpoints]}")

    results: List[Dict[str, Any]] = []
    eval_output_root = args.output_dir / "gen_reval"
    for checkpoint in checkpoints:
        try:
            result = evaluate_checkpoint(
                checkpoint,
                cfg,
                eval_output_root=eval_output_root,
                eval_batch_size=args.eval_batch_size,
                num_data_workers=args.num_data_workers,
            )
        except Exception as exc:
            if args.strict:
                raise
            print(f"[NLG reval] ERROR {checkpoint.name}: {exc}")
            result = {
                "checkpoint": checkpoint.name,
                "checkpoint_path": str(checkpoint),
                "step": _checkpoint_step(checkpoint),
                "error": repr(exc),
                "metrics": {},
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(result)
        write_results(results, args.save_file)

    valid = [row for row in results if not row.get("error") and "eval_meteor" in row.get("metrics", {})]
    if valid:
        best = max(valid, key=lambda row: row["metrics"]["eval_meteor"])
        print(
            "[NLG reval] best by eval_meteor: "
            f"{best['checkpoint']} meteor={best['metrics']['eval_meteor']} "
            f"bleu={best['metrics'].get('eval_bleu')}"
        )
    else:
        print("[NLG reval] no successful generation metric rows yet")


if __name__ == "__main__":
    main()
