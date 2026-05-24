#!/usr/bin/env python
"""Re-evaluate saved SDFT GLUE checkpoints.

This script deliberately avoids Trainer resume state, because old
trainer_state.json files may be truncated by very large SDFT telemetry logs.
It only loads each checkpoint's peft.pt and runs validation again.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep this script quiet and local; the caller can still override these.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

from dataset import load_dataset  # noqa: E402
from modules import load_mamba  # noqa: E402
from trainer.mamba_trainer import MambaTrainer, MambaTrainingArguments  # noqa: E402


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
        raise ValueError(f"Expected a mapping in {path}")
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


def _task_name(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("data", "glue_mnli")).removeprefix("glue_")


def _tag(cfg: Dict[str, Any]) -> str:
    return f"[{_task_name(cfg).upper()} reval]"


def _set_sdft_eval_logging_quiet(model: Any) -> None:
    """Avoid recreating huge per-layer validation logs while re-evaluating."""
    sdft_config = getattr(model, "sdft_config", None)
    if isinstance(sdft_config, dict):
        sdft_config["sdft_log_per_layer"] = False
        sdft_config["sdft_log_grad"] = False

    peft_args = getattr(model, "peft_args", None)
    peft_cfg = peft_args.get("peft") if isinstance(peft_args, dict) else None
    if isinstance(peft_cfg, dict):
        peft_cfg["sdft_log_per_layer"] = False
        peft_cfg["sdft_log_grad"] = False


def _peft_info(model: Any) -> Dict[str, Any]:
    peft_args = getattr(model, "peft_args", None)
    peft_cfg = peft_args.get("peft") if isinstance(peft_args, dict) else None
    if not isinstance(peft_cfg, dict):
        peft_cfg = {}
    return {
        "use_sdft": peft_cfg.get("use_sdft"),
        "sdft_rank": peft_cfg.get("sdft_rank"),
        "sdft_gate_mode": peft_cfg.get("sdft_gate_mode"),
        "sdft_target_layers": peft_cfg.get("sdft_target_layers"),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    cfg: Dict[str, Any],
    eval_output_root: Path,
    val_data_module: Optional[Any],
) -> tuple[Dict[str, Any], Any]:
    tag = _tag(cfg)
    print(f"{tag} loading {checkpoint}")
    loaded = load_mamba(str(checkpoint))
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    _set_sdft_eval_logging_quiet(model)

    if val_data_module is None:
        data_name = cfg.get("val_data") or cfg.get("data", "glue_mnli")
        split = cfg.get("val_data_split", "val")
        val_data_module = load_dataset(data_name, tokenizer, split, mode="lm", return_module=True)

    output_dir = eval_output_root / checkpoint.name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = MambaTrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=int(cfg.get("eval_batch_size", 1)),
        dataloader_num_workers=int(cfg.get("num_data_workers", 0)),
        dataloader_drop_last=val_data_module.dataset.eval_type != "log_likelihood",
        eval_accumulation_steps=int(cfg.get("eval_accumulation_steps", 128)),
        seed=int(cfg.get("seed", 88)),
        report_to=[],
        do_train=False,
        do_eval=True,
        logging_strategy="no",
        save_strategy="no",
        info={
            "task": cfg.get("data", "glue_mnli"),
            "method": cfg.get("method", "sdft"),
            "seed": cfg.get("seed", 88),
            "reval_checkpoint": str(checkpoint),
        },
    )

    trainer = MambaTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=val_data_module.dataset.compute_metrics,
        data_collator=val_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=None,
        skip_metrics=False,
    )
    trainer.preprocess_logits_for_metrics = val_data_module.dataset.preprocess_logits_for_metrics
    trainer.state.global_step = _checkpoint_step(checkpoint)

    metrics = trainer.evaluate(metric_key_prefix="eval")
    result = {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "step": _checkpoint_step(checkpoint),
        **_peft_info(model),
        "metrics": _jsonable(metrics),
    }

    accuracy = metrics.get("eval_accuracy", metrics.get("accuracy"))
    loss = metrics.get("eval_loss")
    print(
        f"{tag} "
        f"{checkpoint.name} step={result['step']} "
        f"accuracy={accuracy if accuracy is not None else 'n/a'} "
        f"loss={loss if loss is not None else 'n/a'}"
    )

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, val_data_module


def write_results(results: List[Dict[str, Any]], save_file: Path, tag: str) -> None:
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
        "use_sdft",
        "sdft_rank",
        "sdft_gate_mode",
        "sdft_target_layers",
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

    print(f"{tag} wrote {save_file}")
    print(f"{tag} wrote {csv_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/sdft/mamba-130m/glue_mnli/seed88"),
        help="Directory containing checkpoint-* folders.",
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("cfg/final/exps/mamba-130m/glue_mnli/sdft.yaml"),
        help="Training config used to recover data split, seed, and eval batch size.",
    )
    parser.add_argument(
        "--save_file",
        type=Path,
        default=None,
        help="JSON result file. Default: <output_dir>/<task>_reval_results.json",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Optional checkpoint name/path. Repeat to select a subset.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_data_workers", type=int, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately if a checkpoint cannot be loaded or evaluated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    cfg = _load_yaml(args.cfg)
    cfg.setdefault("data", "glue_mnli")
    cfg.setdefault("val_data_split", "val")
    cfg.setdefault("seed", 88)
    cfg.setdefault("eval_batch_size", 1)
    cfg.setdefault("num_data_workers", 0)
    if args.eval_batch_size is not None:
        cfg["eval_batch_size"] = args.eval_batch_size
    if args.num_data_workers is not None:
        cfg["num_data_workers"] = args.num_data_workers

    output_dir = args.output_dir
    tag = _tag(cfg)
    save_file = args.save_file or (output_dir / f"{_task_name(cfg)}_reval_results.json")
    checkpoints = _find_checkpoints(output_dir, args.checkpoint)
    eval_output_root = output_dir / "reval_predictions"

    print(f"{tag} checkpoints:")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint}")

    results: List[Dict[str, Any]] = []
    val_data_module = None
    for checkpoint in checkpoints:
        try:
            result, val_data_module = evaluate_checkpoint(
                checkpoint=checkpoint,
                cfg=cfg,
                eval_output_root=eval_output_root,
                val_data_module=val_data_module,
            )
        except Exception as exc:
            if args.strict:
                raise
            result = {
                "checkpoint": checkpoint.name,
                "checkpoint_path": str(checkpoint),
                "step": _checkpoint_step(checkpoint),
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": {},
            }
            print(f"{tag} {checkpoint.name} failed: {result['error']}")
        results.append(result)
        write_results(results, save_file, tag)


if __name__ == "__main__":
    main()
