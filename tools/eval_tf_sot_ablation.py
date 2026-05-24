#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Same-checkpoint TF-SOT ablation eval.

Runs evaluation twice with the same checkpoint:
1. normal: trained TF-SOT scale is active.
2. off: TF-SOT temporal scale is disabled, so s_{l,t}=1, while all other
   checkpoint parameters stay unchanged.

Example:
python tools/eval_tf_sot_ablation.py \
  --checkpoint outputs/tf_sot/mamba-130m/glue_rte/checkpoint-6230 \
  --device 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import load_dataset
from modules import load_mamba
from trainer.mamba_trainer import MambaTrainer, MambaTrainingArguments


DEFAULT_CFG = {
    "eval_batch_size": 1,
    "val_data": None,
    "val_data_split": "val",
    "num_data_workers": 8,
    "seed": 88,
    "skip_metrics": False,
    "log_speed": False,
}


TASK_PRIMARY_METRIC = {
    "glue_cola": "matthews_correlation",
    "glue_mnli": "accuracy",
    "glue_mrpc": "f1",
    "glue_qnli": "accuracy",
    "glue_qqp": "f1",
    "glue_rte": "accuracy",
    "glue_sst2": "accuracy",
}


def infer_cfg_path(checkpoint: str) -> str:
    checkpoint_path = Path(checkpoint)
    parts = checkpoint_path.parts
    for part in parts:
        if part.startswith("glue_"):
            candidate = REPO_ROOT / "cfg" / "final" / "exps" / "mamba-130m" / part / "tf_sot.yaml"
            if candidate.is_file():
                return str(candidate)
    raise ValueError(
        "Could not infer --cfg from checkpoint path. Pass --cfg explicitly, e.g. "
        "cfg/final/exps/mamba-130m/glue_rte/tf_sot.yaml"
    )


def load_eval_args(cfg_path: str, checkpoint: str, output_dir: str | None, eval_batch_size: int | None, seed: int | None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    args = {**DEFAULT_CFG, **cfg}
    args["cfg_path"] = cfg_path
    args["checkpoint"] = checkpoint

    if eval_batch_size is not None:
        args["eval_batch_size"] = eval_batch_size
    if seed is not None:
        args["seed"] = seed

    if output_dir is None:
        output_dir = str(Path(checkpoint) / "tf_sot_ablation_eval")
    args["output_dir"] = output_dir

    if "data" not in args:
        raise ValueError(f"Config does not define data: {cfg_path}")

    return SimpleNamespace(**args)


def set_tf_sot_enabled(model, enabled: bool) -> List[Tuple[str, str, bool]]:
    changed = []
    for module_name, module in model.named_modules():
        tf_sot_enabled = getattr(module, "tf_sot_enabled", None)
        if not isinstance(tf_sot_enabled, dict):
            continue

        for adapter_name, old_value in list(tf_sot_enabled.items()):
            old_bool = bool(old_value)
            if old_bool != enabled:
                tf_sot_enabled[adapter_name] = enabled
                changed.append((module_name, adapter_name, old_bool))
    return changed


def restore_tf_sot_enabled(model, states: List[Tuple[str, str, bool]]) -> None:
    modules = dict(model.named_modules())
    for module_name, adapter_name, old_value in states:
        modules[module_name].tf_sot_enabled[adapter_name] = old_value


def count_tf_sot_coeffs(model) -> int:
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if "suffixtuning_tf_coeff" in name
    )


def build_trainer(model, tokenizer, args):
    val_data_module = load_dataset(
        args.val_data if args.val_data is not None else args.data,
        tokenizer,
        args.val_data_split,
        mode="lm",
        return_module=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = MambaTrainer(
        model=model,
        train_dataset=None,
        tokenizer=tokenizer,
        args=MambaTrainingArguments(
            output_dir=str(output_dir),
            per_device_eval_batch_size=args.eval_batch_size,
            dataloader_num_workers=args.num_data_workers,
            dataloader_prefetch_factor=2 if args.num_data_workers > 0 else None,
            eval_accumulation_steps=128,
            dataloader_drop_last=val_data_module.dataset.eval_type != "log_likelihood",
            seed=args.seed,
            report_to=[],
            run_name=str(output_dir),
            info={
                "task": args.data,
                "checkpoint": args.checkpoint,
                "cfg_path": args.cfg_path,
                "ablation": "tf_sot_same_checkpoint",
            },
        ),
        compute_metrics=val_data_module.dataset.compute_metrics,
        data_collator=val_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=None,
        skip_metrics=args.skip_metrics,
        log_speed=args.log_speed,
    )
    trainer.preprocess_logits_for_metrics = val_data_module.dataset.preprocess_logits_for_metrics
    return trainer


def primary_score(metrics: Dict[str, float], task: str, prefix: str):
    metric = TASK_PRIMARY_METRIC.get(task)
    if metric is None:
        return None, None
    key = f"{prefix}_{metric}"
    return key, metrics.get(key)


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_data_workers", type=int, default=None)
    args = parser.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    cfg_path = args.cfg or infer_cfg_path(args.checkpoint)
    eval_args = load_eval_args(
        cfg_path=cfg_path,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    if args.num_data_workers is not None:
        eval_args.num_data_workers = args.num_data_workers

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {cfg_path}")
    print(f"Task:       {eval_args.data}")
    print(f"Output dir: {eval_args.output_dir}")

    model_tokenizer = load_mamba(args.checkpoint)
    model, tokenizer = model_tokenizer["model"], model_tokenizer["tokenizer"]
    model.eval()

    tf_coeff_params = count_tf_sot_coeffs(model)
    if tf_coeff_params == 0:
        raise RuntimeError("No suffixtuning_tf_coeff parameters found in this checkpoint.")
    print(f"TF-SOT coeff params: {tf_coeff_params}")

    trainer = build_trainer(model, tokenizer, eval_args)

    print("\n========== Normal Eval: TF-SOT scale active ==========")
    normal_metrics = trainer.evaluate(metric_key_prefix="normal")

    print("\n========== Off Eval: force s_{l,t}=1 ==========")
    changed_states = set_tf_sot_enabled(model, enabled=False)
    if not changed_states:
        raise RuntimeError("No enabled TF-SOT adapters found to disable.")
    print(f"Disabled TF-SOT scale in {len(changed_states)} modules.")
    off_metrics = trainer.evaluate(metric_key_prefix="off")
    restore_tf_sot_enabled(model, changed_states)

    normal_key, normal_score = primary_score(normal_metrics, eval_args.data, "normal")
    off_key, off_score = primary_score(off_metrics, eval_args.data, "off")
    delta = None if normal_score is None or off_score is None else normal_score - off_score

    payload = {
        "checkpoint": args.checkpoint,
        "cfg": cfg_path,
        "task": eval_args.data,
        "tf_sot_coeff_params": tf_coeff_params,
        "num_disabled_modules": len(changed_states),
        "normal_metrics": normal_metrics,
        "off_metrics": off_metrics,
        "primary_metric": TASK_PRIMARY_METRIC.get(eval_args.data),
        "normal_score_key": normal_key,
        "off_score_key": off_key,
        "normal_score": normal_score,
        "off_score": off_score,
        "delta_normal_minus_off": delta,
    }

    out_file = Path(eval_args.output_dir) / "tf_sot_same_checkpoint_ablation.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)

    print("\n========== TF-SOT Same-Checkpoint Ablation ==========")
    print(f"Task:                  {eval_args.data}")
    print(f"Primary metric:        {TASK_PRIMARY_METRIC.get(eval_args.data)}")
    print(f"Score normal:          {normal_score}")
    print(f"Score off, s_l_t=1:    {off_score}")
    print(f"Delta normal - off:    {delta}")
    print(f"Disabled modules:      {len(changed_states)}")
    print(f"Saved JSON:            {out_file}")


if __name__ == "__main__":
    main()
