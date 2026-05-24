from dataclasses import dataclass, field
from functools import partial
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, Any

_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_MAMBA_SRC = (_REPO_ROOT / "src" / "mamba").resolve()
if os.environ.get("S6_BRIDGE_USE_LOCAL_MAMBA", "0").lower() not in {"1", "true", "yes"}:
    sys.path = [
        path
        for path in sys.path
        if not path or Path(path).resolve() != _LOCAL_MAMBA_SRC
    ]

from mamba_ssm.modules.mamba_simple import Mamba
import torch
import argparse
import numpy as np
from torch import nn

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments, GPTNeoXTokenizerFast
import yaml
from peft import PeftConfig, PeftModelForSeq2SeqLM, LoraModel
from modules import MambaPeft, MambaLMHeadModelPeft, get_mamba_peft_model, get_trainable_parameters_ratio, load_mamba, print_trainable_parameter_names
from modules.deep_supervision import configure_deep_supervision
from modules.s6_attention_bridge import configure_s6_attention_bridge
from modules.generation import create_generator
from modules.lm_head_full_tuning import enable_lm_head_full_tuning
from modules.mamba_peft_utils import set_peft_params_trainable
from modules.sdft import (
    SDFTConfig,
    freeze_for_sdft,
    freeze_lm_head_weight_for_sdft,
    inject_sdft_adapters,
    is_sdft_config_dict,
    merge_sdft_config,
    print_sdft_summary,
)
from modules.pd_dft import (
    PDDFTConfig,
    freeze_lm_head_weight_for_pd_dft,
    inject_pd_dft_adapters,
    is_pd_dft_config_dict,
    mark_only_pd_dft_as_trainable,
    merge_pd_dft_config,
    print_pd_dft_summary,
)
from modules.snoft import (
    SNOFTConfig,
    freeze_lm_head_weight_for_snoft,
    inject_snoft_adapters,
    is_snoft_config_dict,
    is_snoft_requested,
    mark_only_snoft_as_trainable,
    merge_snoft_config,
    print_snoft_summary,
)
from modules.psca_wr import (
    PSCAWRConfig,
    freeze_lm_head_weight_for_psca_wr,
    inject_psca_wr_adapters,
    is_psca_wr_config_dict,
    mark_only_psca_wr_as_trainable,
    merge_psca_wr_config,
    print_psca_wr_summary,
)
from dataset import load_dataset
from trainer.mamba_trainer import MambaTrainer, MambaTrainingArguments

try:
    import tensorboard
except ImportError:
    tensorboard = None
try:
    import wandb
except ImportError:
    wandb = None

from utils.utils import create_non_existent_file


GLUE_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "wnli": 2,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_glue_task_name(task_name):
    return str(task_name).removeprefix("glue_")


def normalize_data_name(task_name):
    task_name = normalize_glue_task_name(task_name)
    return f"glue_{task_name}"


def get_label_token_ids(tokenizer, task_name):
    task_name = normalize_glue_task_name(task_name)
    label_token_ids = []
    for label in range(GLUE_NUM_LABELS[task_name]):
        token_ids = tokenizer.encode(str(label))
        if len(token_ids) != 1:
            raise ValueError(f"Expected GLUE label {label!r} to map to one token, got {token_ids}.")
        label_token_ids.append(int(token_ids[0]))
    return label_token_ids


def load_peft_config_dict(peft_path):
    if peft_path is None:
        return None
    try:
        with open(peft_path, "r") as f:
            return json.load(f)
    except (OSError, TypeError):
        return None


def get_method_name(peft_path):
    if peft_path is None:
        return "full"

    peft_cfg = load_peft_config_dict(peft_path)
    if peft_cfg is None:
        return str(peft_path)

    if is_sdft_config_dict(peft_cfg):
        return "sdft"

    if is_pd_dft_config_dict(peft_cfg):
        return "pd_dft"

    if is_snoft_config_dict(peft_cfg):
        return "snoft_e"

    if is_psca_wr_config_dict(peft_cfg):
        return "psca_lite" if bool(peft_cfg.get("psca_fallback_lite", False)) else "psca_wr"

    if peft_cfg.get("method") in ("adamix_sot", "sot_sft", "sot_ds", "probe_then_adapt_ds_sot"):
        return peft_cfg.get("method")

    if peft_cfg.get("use_sft"):
        return "sot_sft"

    if peft_cfg.get("use_cf_sot"):
        return "cf_sot"

    if peft_cfg.get("use_tf_sot"):
        return "tf_sot"

    bias_type_to_method = {
        "SILU_Z_C": "sot",
        "KERNEL_SILU_Z_C": "ksot",
        "DYNAMIC_SILU_Z_C": "dsot",
        "KEY_AWARE_DYNAMIC_SILU_Z_C": "keyaware_dsot",
        "SILU_Z": "output_tuning",
    }
    return bias_type_to_method.get(peft_cfg.get("bias_type"), peft_cfg.get("peft_type", str(peft_path)).lower())


def get_default_best_metric(data):
    task = data.removeprefix("glue_")
    if task == "cola":
        return "matthews_correlation"
    if task in ("mnli", "qnli", "rte", "sst2"):
        return "accuracy"
    if task in ("mrpc", "qqp"):
        return "f1"
    return None


def get_peft_attr(peft_cfg, name, default=None):
    return getattr(peft_cfg, name, default) if peft_cfg is not None else default


def is_lm_head_full_tuning(args):
    method = getattr(args, "method", None)
    return str(method).lower() == "lm_head_full" or bool(getattr(args, "tune_lm_head_only", False))


def get_parameter_counts(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    return total_params, trainable_params, trainable_percentage


def _env_flag_enabled(name):
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "y", "on")


def get_report_to():
    if _env_flag_enabled("WANDB_DISABLED") or str(os.environ.get("WANDB_MODE", "")).strip().lower() == "disabled":
        return []
    if wandb is None:
        return []
    return "wandb"


def get_adamix_effective_offset_params(model, num_experts):
    if num_experts is None or int(num_experts) <= 0:
        return None

    num_experts = int(num_experts)
    trainable_offset_params = 0
    for name, param in model.named_parameters():
        if (
            param.requires_grad
            and "suffixtuning_bias" in name
            and param.ndim >= 3
            and param.shape[0] == num_experts
        ):
            trainable_offset_params += param.numel()

    if trainable_offset_params == 0:
        return None
    return trainable_offset_params // num_experts


def print_adamix_sot_summary(model, peft_cfg):
    if get_peft_attr(peft_cfg, "method") != "adamix_sot":
        return

    total_params, trainable_params, trainable_percentage = get_parameter_counts(model)
    num_experts = get_peft_attr(peft_cfg, "num_experts")
    effective_offset_params = get_adamix_effective_offset_params(model, num_experts)

    print("AdaMix-SOT summary:")
    print(f"  total parameters: {total_params:,}")
    print(f"  training_trainable_params: {trainable_params:,}")
    print(f"  trainable percentage: {trainable_percentage:.6f}%")
    print(f"  num_experts: {num_experts}")
    print(f"  consistency_lambda: {get_peft_attr(peft_cfg, 'consistency_lambda')}")
    print(f"  use_consistency: {get_peft_attr(peft_cfg, 'use_consistency')}")
    print(f"  inference_merge: {get_peft_attr(peft_cfg, 'inference_merge')}")
    if effective_offset_params is not None:
        print(f"  inference_effective_offset_params: {effective_offset_params:,}")


def print_sft_parameter_summary(model):
    sft_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and "sft_delta_bias" in name
    ]

    if not sft_params:
        return

    total_params, trainable_params, trainable_percentage = get_parameter_counts(model)
    print("Trainable SFT parameters:")
    for name, param in sft_params:
        print(f"  - {name}: shape={tuple(param.shape)}, params={param.numel():,}")
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_percentage:.6f}%")


def save_results_json(args, method_name, peft_cfg, trainer):
    total_params, trainable_params, trainable_percentage = get_parameter_counts(trainer.model)
    trainable_ratio = trainable_percentage / 100.0

    def _extract_eval_metrics():
        metric_name = getattr(args, "metric_for_best_model", None)
        metric_key = f"eval_{metric_name}" if metric_name else None
        eval_logs = [
            log for log in trainer.state.log_history
            if isinstance(log, dict) and any(str(key).startswith("eval_") for key in log.keys())
        ]
        if not eval_logs:
            return {}
        if metric_key is not None and trainer.state.best_metric is not None:
            best_value = float(trainer.state.best_metric)
            candidates = [log for log in eval_logs if metric_key in log]
            if candidates:
                return min(candidates, key=lambda log: abs(float(log[metric_key]) - best_value))
        return eval_logs[-1]

    eval_log = _extract_eval_metrics()
    eval_metrics = {
        key.removeprefix("eval_"): value
        for key, value in eval_log.items()
        if str(key).startswith("eval_")
    }
    summary = {
        "task_name": args.data,
        "method": method_name,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_train_epochs": args.num_epochs,
        "num_groups": get_peft_attr(peft_cfg, "num_groups"),
        "chunk_size": get_peft_attr(peft_cfg, "chunk_size"),
        "router_rank": get_peft_attr(peft_cfg, "router_rank"),
        "tau_logit_init": get_peft_attr(peft_cfg, "tau_logit_init"),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric_name": getattr(args, "metric_for_best_model", None),
        "best_metric": trainer.state.best_metric,
        "eval_accuracy": eval_metrics.get("accuracy"),
        "eval_f1": eval_metrics.get("f1"),
        "eval_matthews_correlation": eval_metrics.get("matthews_correlation"),
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "bridge_enabled": getattr(args, "bridge_enabled", False),
        "lambda_lin": getattr(args, "lambda_lin", 0.01),
        "lambda_handoff": getattr(args, "lambda_handoff", 0.03),
        "use_psca_wr": get_peft_attr(peft_cfg, "use_psca_wr"),
        "psca_rank": get_peft_attr(peft_cfg, "psca_rank"),
        "psca_alpha": get_peft_attr(peft_cfg, "psca_alpha"),
        "psca_adapt_b": get_peft_attr(peft_cfg, "psca_adapt_b"),
        "psca_adapt_c": get_peft_attr(peft_cfg, "psca_adapt_c"),
        "psca_use_projector_shift": get_peft_attr(peft_cfg, "psca_use_projector_shift"),
        "psca_projector_residual": get_peft_attr(peft_cfg, "psca_projector_residual"),
        "psca_projector_scale": get_peft_attr(peft_cfg, "psca_projector_scale"),
        "psca_fallback_lite": get_peft_attr(peft_cfg, "psca_fallback_lite"),
    }

    payload = {
        "task": args.data,
        "task_name": args.data,
        "method": method_name,
        "seed": args.seed,
        "cfg_path": args.cfg_path,
        "output_dir": args.output_dir,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric_name": getattr(args, "metric_for_best_model", None),
        "best_metric": trainer.state.best_metric,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_train_epochs": args.num_epochs,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "bridge_enabled": getattr(args, "bridge_enabled", False),
        "bridge_all_layers": getattr(args, "bridge_all_layers", True),
        "lambda_lin": getattr(args, "lambda_lin", 0.01),
        "lambda_handoff": getattr(args, "lambda_handoff", 0.03),
        "bridge_use_soft_mixing": getattr(args, "bridge_use_soft_mixing", True),
        "bridge_mix_ratio_init": getattr(args, "bridge_mix_ratio_init", 1.0),
        "bridge_mix_decay_portion": getattr(args, "bridge_mix_decay_portion", 0.3),
        "final_no_attention_portion": getattr(args, "final_no_attention_portion", 0.1),
        "bridge_log_interval": getattr(args, "bridge_log_interval", 10),
        "num_experts": get_peft_attr(peft_cfg, "num_experts"),
        "consistency_lambda": get_peft_attr(peft_cfg, "consistency_lambda"),
        "use_consistency": get_peft_attr(peft_cfg, "use_consistency"),
        "inference_merge": get_peft_attr(peft_cfg, "inference_merge"),
        "use_sft": get_peft_attr(peft_cfg, "use_sft"),
        "sft_type": get_peft_attr(peft_cfg, "sft_type"),
        "sft_delta_scale": get_peft_attr(peft_cfg, "sft_delta_scale"),
        "sft_clamp": get_peft_attr(peft_cfg, "sft_clamp"),
        "use_tf_sot": get_peft_attr(peft_cfg, "use_tf_sot"),
        "tf_sot_num_freqs": get_peft_attr(peft_cfg, "tf_sot_num_freqs"),
        "tf_sot_num_basis": get_peft_attr(peft_cfg, "tf_sot_num_basis"),
        "tf_sot_eps": get_peft_attr(peft_cfg, "tf_sot_eps"),
        "tf_sot_max_seq_len": get_peft_attr(peft_cfg, "tf_sot_max_seq_len"),
        "tf_sot_freq_grid": get_peft_attr(peft_cfg, "tf_sot_freq_grid"),
        "tf_sot_normalize_basis": get_peft_attr(peft_cfg, "tf_sot_normalize_basis"),
        "use_cf_sot": get_peft_attr(peft_cfg, "use_cf_sot"),
        "cf_sot_num_freqs": get_peft_attr(peft_cfg, "cf_sot_num_freqs"),
        "cf_sot_num_basis": get_peft_attr(peft_cfg, "cf_sot_num_basis"),
        "cf_sot_eps": get_peft_attr(peft_cfg, "cf_sot_eps"),
        "cf_sot_max_seq_len": get_peft_attr(peft_cfg, "cf_sot_max_seq_len"),
        "cf_sot_freq_grid": get_peft_attr(peft_cfg, "cf_sot_freq_grid"),
        "cf_sot_normalize_basis": get_peft_attr(peft_cfg, "cf_sot_normalize_basis"),
        "cf_sot_context_center": get_peft_attr(peft_cfg, "cf_sot_context_center"),
        "cf_sot_context_reduce": get_peft_attr(peft_cfg, "cf_sot_context_reduce"),
        "use_sdft": get_peft_attr(peft_cfg, "use_sdft"),
        "sdft_rank": get_peft_attr(peft_cfg, "sdft_rank"),
        "sdft_rho_init": get_peft_attr(peft_cfg, "sdft_rho_init"),
        "sdft_gate_mode": get_peft_attr(peft_cfg, "sdft_gate_mode"),
        "sdft_dropout": get_peft_attr(peft_cfg, "sdft_dropout"),
        "sdft_target_layers": get_peft_attr(peft_cfg, "sdft_target_layers"),
        "sdft_freeze_base_model": get_peft_attr(peft_cfg, "sdft_freeze_base_model"),
        "sdft_train_classifier": get_peft_attr(peft_cfg, "sdft_train_classifier"),
        "sdft_log_stats": get_peft_attr(peft_cfg, "sdft_log_stats"),
        "sdft_log_interval": get_peft_attr(peft_cfg, "sdft_log_interval"),
        "sdft_log_per_layer": get_peft_attr(peft_cfg, "sdft_log_per_layer"),
        "sdft_log_grad": get_peft_attr(peft_cfg, "sdft_log_grad"),
        "use_pd_dft": get_peft_attr(peft_cfg, "use_pd_dft"),
        "pd_dft_rank": get_peft_attr(peft_cfg, "pd_dft_rank"),
        "pd_dft_dropout": get_peft_attr(peft_cfg, "pd_dft_dropout"),
        "pd_dft_rho_param_init": get_peft_attr(peft_cfg, "pd_dft_rho_param_init"),
        "pd_dft_rho_scan_init": get_peft_attr(peft_cfg, "pd_dft_rho_scan_init"),
        "pd_dft_learnable_rho": get_peft_attr(peft_cfg, "pd_dft_learnable_rho"),
        "pd_dft_mode": get_peft_attr(peft_cfg, "pd_dft_mode"),
        "pd_dft_target_layers": get_peft_attr(peft_cfg, "pd_dft_target_layers"),
        "pd_dft_share_down": get_peft_attr(peft_cfg, "pd_dft_share_down"),
        "pd_dft_max_delta_ratio_param": get_peft_attr(peft_cfg, "pd_dft_max_delta_ratio_param"),
        "pd_dft_max_delta_ratio_scan": get_peft_attr(peft_cfg, "pd_dft_max_delta_ratio_scan"),
        "pd_dft_log_stats": get_peft_attr(peft_cfg, "pd_dft_log_stats"),
        "pd_dft_log_per_layer": get_peft_attr(peft_cfg, "pd_dft_log_per_layer"),
        "pd_dft_log_grad": get_peft_attr(peft_cfg, "pd_dft_log_grad"),
        "pd_dft_log_interval": get_peft_attr(peft_cfg, "pd_dft_log_interval"),
        "use_snoft": get_peft_attr(peft_cfg, "use_snoft"),
        "snoft_enabled": get_peft_attr(peft_cfg, "enabled"),
        "snoft_num_groups": get_peft_attr(peft_cfg, "num_groups"),
        "snoft_chunk_size": get_peft_attr(peft_cfg, "chunk_size"),
        "snoft_router_rank": get_peft_attr(peft_cfg, "router_rank"),
        "snoft_tau_logit_init": get_peft_attr(peft_cfg, "tau_logit_init"),
        "snoft_freeze_backbone": get_peft_attr(peft_cfg, "freeze_backbone"),
        "snoft_train_task_head": get_peft_attr(peft_cfg, "train_task_head"),
        "snoft_target_layers": get_peft_attr(peft_cfg, "target_layers"),
        "use_psca_wr": get_peft_attr(peft_cfg, "use_psca_wr"),
        "psca_rank": get_peft_attr(peft_cfg, "psca_rank"),
        "psca_alpha": get_peft_attr(peft_cfg, "psca_alpha"),
        "psca_dropout": get_peft_attr(peft_cfg, "psca_dropout"),
        "psca_target_modules": get_peft_attr(peft_cfg, "psca_target_modules"),
        "psca_init_zero": get_peft_attr(peft_cfg, "psca_init_zero"),
        "psca_adapt_b": get_peft_attr(peft_cfg, "psca_adapt_b"),
        "psca_adapt_c": get_peft_attr(peft_cfg, "psca_adapt_c"),
        "psca_use_projector_shift": get_peft_attr(peft_cfg, "psca_use_projector_shift"),
        "psca_projector_residual": get_peft_attr(peft_cfg, "psca_projector_residual"),
        "psca_projector_scale": get_peft_attr(peft_cfg, "psca_projector_scale"),
        "psca_fallback_lite": get_peft_attr(peft_cfg, "psca_fallback_lite"),
        "psca_random_gate": get_peft_attr(peft_cfg, "psca_random_gate"),
        "psca_independent_gate": get_peft_attr(peft_cfg, "psca_independent_gate"),
        "eval_accuracy": eval_metrics.get("accuracy"),
        "eval_f1": eval_metrics.get("f1"),
        "eval_matthews_correlation": eval_metrics.get("matthews_correlation"),
        "summary": summary,
        "use_deep_supervision": getattr(args, "use_deep_supervision", False),
        "aux_layers": getattr(args, "aux_layers", []),
        "aux_loss_weight": getattr(args, "aux_loss_weight", 0.1),
        "aux_weight_scheme": getattr(args, "aux_weight_scheme", "linear_increase"),
        "aux_pooling": getattr(args, "aux_pooling", "last_token"),
        "ds_adaptive": getattr(args, "ds_adaptive", False),
        "ds_adaptive_strategy": getattr(args, "ds_adaptive_strategy", "loss_drop"),
        "candidate_aux_layers": getattr(args, "candidate_aux_layers", []),
        "probe_ratio": getattr(args, "probe_ratio", 0.15),
        "probe_aux_weight": getattr(args, "probe_aux_weight", 0.05),
        "probe_start_window_ratio": getattr(args, "probe_start_window_ratio", 0.3),
        "probe_end_window_ratio": getattr(args, "probe_end_window_ratio", 0.3),
        "probe_loss_stat": getattr(args, "probe_loss_stat", "window_mean"),
        "adaptive_top_k": getattr(args, "adaptive_top_k", 3),
        "adaptive_late_bias_gamma": getattr(args, "adaptive_late_bias_gamma", 1.0),
        "adaptive_score_mode": getattr(args, "adaptive_score_mode", "drop_plus_confidence"),
        "adaptive_confidence_weight": getattr(args, "adaptive_confidence_weight", 0.1),
        "adaptive_score_threshold": getattr(args, "adaptive_score_threshold", 0.02),
        "adaptive_min_layer": getattr(args, "adaptive_min_layer", 0),
        "adaptive_disable_if_low_score": getattr(args, "adaptive_disable_if_low_score", False),
        "fallback_aux_layers": getattr(args, "fallback_aux_layers", [16, 20, 24]),
        "fallback_aux_weight_scheme": getattr(args, "fallback_aux_weight_scheme", "linear_increase"),
        "fallback_aux_loss_weight_scale": getattr(args, "fallback_aux_loss_weight_scale", 0.5),
        "fallback_confidence_threshold": getattr(args, "fallback_confidence_threshold", 0.0),
        "ds_schedule": getattr(args, "ds_schedule", "constant"),
        "ds_start_ratio": getattr(args, "ds_start_ratio", None),
        "ds_warmup_ratio": getattr(args, "ds_warmup_ratio", 0.3),
        "log_history": trainer.state.log_history,
    }

    def _json_default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if torch.is_tensor(obj):
            return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
        return str(obj)

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    if method_name in ("snoft", "snoft_e"):
        with open(Path(args.output_dir) / "snoft_e_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        print(json.dumps(summary, indent=2, default=_json_default))


def init_embedding(model, tokenizer):
    model.backbone.embedding = nn.Embedding(
        tokenizer.vocab_size,
        model.backbone.embedding.embedding_dim,
        device=model.backbone.embedding.weight.device,
        dtype=model.backbone.embedding.weight.dtype,
    )
    model.lm_head = nn.Linear(
        model.backbone.embedding.embedding_dim,
        tokenizer.vocab_size,
        device=model.backbone.embedding.weight.device,
        dtype=model.backbone.embedding.weight.dtype,
        bias=False
    )
    model.tie_weights()


def run_train(args):
    if args.overwrite and args.sdt:
        assert Path(args.output_dir).exists()

    if not args.overwrite:
        if (Path(args.output_dir) / "cfg.yaml").exists():
            if args.resume:
                resume_from_checkpoint = True
            else:
                assert False, str(Path(args.output_dir) / "cfg.yaml") + " exists!"
                resume_from_checkpoint = False
        else:
            resume_from_checkpoint = False
    else:
        resume_from_checkpoint = False

    assert args.data.startswith("glue_") or args.data in ("glue_rte", "glue_mrpc", "glue_cola", "spider_1000") or not (args.no_save and args.num_epochs > 1), "don't train for more than one epoch without saving ckpts!"
    lm_head_full_requested = is_lm_head_full_tuning(args)
    peft_cfg_dict = load_peft_config_dict(args.peft)
    explicit_keys = set(getattr(args, "_explicit_keys", []))
    sdft_overrides = {
        key: getattr(args, key)
        for key in SDFTConfig.__dataclass_fields__.keys()
        if key in explicit_keys and hasattr(args, key)
    }
    if str(getattr(args, "method", "")).lower() == "sdft":
        sdft_overrides["method"] = "sdft"
    sdft_config = merge_sdft_config(peft_cfg_dict, SimpleNamespace(**sdft_overrides))
    sdft_requested = bool(sdft_config.use_sdft)
    pd_dft_overrides = {
        key: getattr(args, key)
        for key in PDDFTConfig.__dataclass_fields__.keys()
        if key in explicit_keys and hasattr(args, key)
    }
    if str(getattr(args, "method", "")).lower() in ("pd_dft", "path_decoupled_dft"):
        pd_dft_overrides["method"] = "pd_dft"
    pd_dft_config = merge_pd_dft_config(peft_cfg_dict, SimpleNamespace(**pd_dft_overrides))
    pd_dft_requested = bool(pd_dft_config.use_pd_dft)
    snoft_overrides = {}
    if str(getattr(args, "method", "")).lower() in ("snoft", "snoft_e"):
        snoft_overrides["method"] = getattr(args, "method")
    if "use_snoft" in explicit_keys and hasattr(args, "use_snoft"):
        snoft_overrides["use_snoft"] = getattr(args, "use_snoft")
    if "snoft" in explicit_keys and hasattr(args, "snoft"):
        snoft_overrides["snoft"] = getattr(args, "snoft")
    for key in SNOFTConfig.__dataclass_fields__.keys():
        prefixed = f"snoft_{key}"
        if prefixed in explicit_keys and hasattr(args, prefixed):
            snoft_overrides[prefixed] = getattr(args, prefixed)
    snoft_args = SimpleNamespace(**snoft_overrides)
    snoft_requested = is_snoft_requested(peft_cfg_dict, snoft_args)
    snoft_config = merge_snoft_config(peft_cfg_dict, snoft_args) if snoft_requested else SNOFTConfig()
    if snoft_requested:
        snoft_config.enabled = True
    psca_overrides = {
        key: getattr(args, key)
        for key in PSCAWRConfig.__dataclass_fields__.keys()
        if key in explicit_keys and hasattr(args, key)
    }
    if str(getattr(args, "method", "")).lower() in ("psca_wr", "psca-wr", "psca_lite", "psca-lite"):
        psca_overrides["method"] = getattr(args, "method")
    if bool(getattr(args, "debug", False)):
        psca_overrides["psca_debug"] = True
    psca_config = merge_psca_wr_config(peft_cfg_dict, SimpleNamespace(**psca_overrides))
    psca_requested = bool(psca_config.use_psca_wr)
    if psca_requested and args.peft is not None and not is_psca_wr_config_dict(peft_cfg_dict):
        raise RuntimeError("[PSCA-WR] Do not combine PSCA-WR CLI flags with a non-PSCA PEFT config.")
    if sum([sdft_requested, pd_dft_requested, snoft_requested, psca_requested]) > 1:
        raise RuntimeError("SDFT, PD-DFT, SNOFT-E, and PSCA-WR cannot be enabled together.")
    if sdft_requested:
        sdft_config.use_sdft = True
        for key, value in sdft_config.to_dict().items():
            setattr(args, key, value)
    if pd_dft_requested:
        pd_dft_config.use_pd_dft = True
        for key, value in pd_dft_config.to_dict().items():
            setattr(args, key, value)
    if snoft_requested:
        for key, value in snoft_config.to_dict().items():
            setattr(args, f"snoft_{key}", value)
        args.use_deep_supervision = False
        args.ds_adaptive = False
    if psca_requested:
        psca_config.use_psca_wr = True
        for key, value in psca_config.to_dict().items():
            setattr(args, key, value)

    is_custom_tokenizer = False
    # is_custom_tokenizer = args.tokenizer != "EleutherAI/gpt-neox-20b"
    # tokenizer = get_tokenizer(args.tokenizer)

    model_kwargs = dict(
        dtype={"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}[args.prec],
        device="cuda",
        use_fast_path=False,
        mamba_cls=MambaPeft if sdft_requested or pd_dft_requested or snoft_requested or psca_requested or (args.peft is not None and not lm_head_full_requested) else Mamba,
        backend=args.backend,
    )

    # model = load_mamba(args.model, **model_kwargs)
    model_tokenizer = load_mamba(
        args.model,
        cls=MambaLMHeadModelPeft,
        **model_kwargs
    )
    model, tokenizer = model_tokenizer["model"], model_tokenizer["tokenizer"]

    if args.from_scratch:
        model = MambaLMHeadModelPeft(model.config, **model_kwargs)

    if is_custom_tokenizer:
        print(f"Resizing, randomly initializing and unfreezing embedding layer for custom tokenizer")
        init_embedding(model, tokenizer)

    sdft_target_layers = []
    pd_dft_target_layers = []
    snoft_target_layers = []
    psca_target_layers = []
    if sdft_requested and not lm_head_full_requested:
        sdft_target_layers = inject_sdft_adapters(model, sdft_config)
        peft_cfg = SimpleNamespace(method="sdft", **sdft_config.to_dict())
        model.peft_args = {
            "peft": {
                "method": "sdft",
                **sdft_config.to_dict(),
            }
        }
    elif pd_dft_requested and not lm_head_full_requested:
        pd_dft_target_layers = inject_pd_dft_adapters(model, pd_dft_config)
        peft_cfg = SimpleNamespace(method="pd_dft", **pd_dft_config.to_dict())
        model.peft_args = {
            "peft": {
                "method": "pd_dft",
                **pd_dft_config.to_dict(),
            }
        }
    elif snoft_requested and not lm_head_full_requested:
        snoft_target_layers = inject_snoft_adapters(model, snoft_config)
        snoft_values = snoft_config.to_dict()
        snoft_method = snoft_values.pop("method")
        peft_cfg = SimpleNamespace(method=snoft_method, use_snoft=True, **snoft_values)
        model.peft_args = {
            "peft": {
                "method": "snoft_e",
                "snoft": snoft_config.to_dict(),
            }
        }
    elif psca_requested and not lm_head_full_requested:
        psca_target_layers = inject_psca_wr_adapters(model, psca_config)
        peft_method = "psca_lite" if psca_config.psca_fallback_lite else "psca_wr"
        peft_cfg = SimpleNamespace(method=peft_method, **psca_config.to_dict())
        model.peft_args = {
            "peft": {
                "method": peft_method,
                **psca_config.to_dict(),
            }
        }
    elif args.peft is not None and not lm_head_full_requested:
        model, peft_cfg = get_mamba_peft_model(model, args.peft, return_peft_cfg=True, train_embedding=is_custom_tokenizer, no_print=True)
    else:
        if lm_head_full_requested and args.peft is not None:
            print(f"[LM_HEAD_FULL] Skipping PEFT config for lm_head-only tuning: {args.peft}")
        peft_cfg = None

    if lm_head_full_requested and args.train_all_peft:
        raise RuntimeError("[LM_HEAD_FULL] train_all_peft=True conflicts with lm_head-only tuning.")
    if sdft_requested and args.train_all_peft:
        raise RuntimeError("[SDFT] train_all_peft=True conflicts with independent SDFT tuning.")
    if pd_dft_requested and args.train_all_peft:
        raise RuntimeError("[PD-DFT] train_all_peft=True conflicts with independent PD-DFT tuning.")
    if snoft_requested and args.train_all_peft:
        raise RuntimeError("[SNOFT-E] train_all_peft=True conflicts with independent SNOFT-E tuning.")
    if psca_requested and args.train_all_peft:
        raise RuntimeError("[PSCA-WR] train_all_peft=True conflicts with independent PSCA-WR tuning.")

    if args.train_all_peft:
        lora = model.base_model.model.base_model
        assert isinstance(lora, LoraModel)
        set_peft_params_trainable(model, lora.prefix, enable_train=True, disable_train=False)

    deep_supervision_requested = bool(getattr(args, "use_deep_supervision", False) or getattr(args, "ds_adaptive", False))
    if lm_head_full_requested and deep_supervision_requested:
        print("[LM_HEAD_FULL] Skipping deep supervision and auxiliary heads for lm_head-only tuning.")
        deep_supervision_requested = False
    if sdft_requested and deep_supervision_requested:
        print("[SDFT] Skipping deep supervision; SDFT runs without SOT+DS auxiliary heads.")
        deep_supervision_requested = False
        args.use_deep_supervision = False
        args.ds_adaptive = False
    if pd_dft_requested and deep_supervision_requested:
        print("[PD-DFT] Skipping deep supervision; PD-DFT runs without SOT+DS auxiliary heads.")
        deep_supervision_requested = False
        args.use_deep_supervision = False
        args.ds_adaptive = False
    if snoft_requested and deep_supervision_requested:
        print("[SNOFT-E] Skipping deep supervision; SNOFT-E runs without SOT+DS auxiliary heads.")
        deep_supervision_requested = False
        args.use_deep_supervision = False
        args.ds_adaptive = False

    if deep_supervision_requested:
        if not args.data.startswith("glue_"):
            raise ValueError("SOT+DS first version supports GLUE classification tasks only.")
        task_name = normalize_glue_task_name(args.data)
        if task_name not in GLUE_NUM_LABELS:
            raise ValueError(f"Unsupported GLUE task for SOT+DS: {task_name}")
        model = configure_deep_supervision(
            model,
            use_deep_supervision=True,
            aux_layers=getattr(args, "aux_layers", []),
            aux_loss_weight=getattr(args, "aux_loss_weight", 0.1),
            aux_weight_scheme=getattr(args, "aux_weight_scheme", "linear_increase"),
            aux_pooling=getattr(args, "aux_pooling", "last_token"),
            ds_adaptive=getattr(args, "ds_adaptive", False),
            ds_adaptive_strategy=getattr(args, "ds_adaptive_strategy", "loss_drop"),
            candidate_aux_layers=getattr(args, "candidate_aux_layers", None),
            probe_ratio=getattr(args, "probe_ratio", 0.15),
            probe_aux_weight=getattr(args, "probe_aux_weight", 0.05),
            probe_start_window_ratio=getattr(args, "probe_start_window_ratio", 0.3),
            probe_end_window_ratio=getattr(args, "probe_end_window_ratio", 0.3),
            probe_loss_stat=getattr(args, "probe_loss_stat", "window_mean"),
            adaptive_top_k=getattr(args, "adaptive_top_k", 3),
            adaptive_late_bias_gamma=getattr(args, "adaptive_late_bias_gamma", 1.0),
            adaptive_score_mode=getattr(args, "adaptive_score_mode", "drop_plus_confidence"),
            adaptive_confidence_weight=getattr(args, "adaptive_confidence_weight", 0.1),
            adaptive_score_threshold=getattr(args, "adaptive_score_threshold", 0.02),
            adaptive_min_layer=getattr(args, "adaptive_min_layer", 0),
            adaptive_disable_if_low_score=getattr(args, "adaptive_disable_if_low_score", False),
            fallback_aux_layers=getattr(args, "fallback_aux_layers", [16, 20, 24]),
            fallback_aux_weight_scheme=getattr(args, "fallback_aux_weight_scheme", "linear_increase"),
            fallback_aux_loss_weight_scale=getattr(args, "fallback_aux_loss_weight_scale", 0.5),
            fallback_confidence_threshold=getattr(args, "fallback_confidence_threshold", 0.0),
            ds_schedule=getattr(args, "ds_schedule", "constant"),
            ds_start_ratio=getattr(args, "ds_start_ratio", None),
            ds_warmup_ratio=getattr(args, "ds_warmup_ratio", 0.3),
            num_labels=GLUE_NUM_LABELS[task_name],
            label_token_ids=get_label_token_ids(tokenizer, task_name),
            task_name=task_name,
        )
    else:
        model = configure_deep_supervision(model, use_deep_supervision=False)

    method_name = (
        "sdft" if sdft_requested and not lm_head_full_requested
        else "pd_dft" if pd_dft_requested and not lm_head_full_requested
        else "snoft_e" if snoft_requested and not lm_head_full_requested
        else ("psca_lite" if psca_requested and psca_config.psca_fallback_lite and not lm_head_full_requested else "psca_wr") if psca_requested and not lm_head_full_requested
        else ("lm_head_full" if lm_head_full_requested else (getattr(args, "method", None) or get_method_name(args.peft)))
    )
    if deep_supervision_requested and getattr(args, "ds_adaptive", False) and method_name in ("sot", "sot_ds"):
        method_name = "probe_then_adapt_ds_sot"
    if deep_supervision_requested and method_name == "sot":
        method_name = "sot_ds"
    if lm_head_full_requested:
        model = enable_lm_head_full_tuning(model)
    if sdft_requested and not lm_head_full_requested:
        if sdft_config.sdft_freeze_base_model:
            freeze_for_sdft(model, train_classifier=sdft_config.sdft_train_classifier)
        freeze_lm_head_weight_for_sdft(model)
    if pd_dft_requested and not lm_head_full_requested:
        mark_only_pd_dft_as_trainable(model, train_classifier=True)
        freeze_lm_head_weight_for_pd_dft(model)
    if snoft_requested and not lm_head_full_requested:
        if snoft_config.freeze_backbone:
            mark_only_snoft_as_trainable(model, train_task_head=snoft_config.train_task_head)
        freeze_lm_head_weight_for_snoft(model)
    model = configure_s6_attention_bridge(
        model,
        enabled=bool(getattr(args, "bridge_enabled", True)),
        bridge_all_layers=bool(getattr(args, "bridge_all_layers", True)),
        lambda_lin=getattr(args, "lambda_lin", 0.01),
        lambda_handoff=getattr(args, "lambda_handoff", 0.03),
        bridge_use_soft_mixing=bool(getattr(args, "bridge_use_soft_mixing", True)),
        bridge_mix_ratio_init=getattr(args, "bridge_mix_ratio_init", 1.0),
        bridge_mix_decay_portion=getattr(args, "bridge_mix_decay_portion", 0.3),
        final_no_attention_portion=getattr(args, "final_no_attention_portion", 0.1),
        bridge_log_interval=getattr(args, "bridge_log_interval", 10),
    )
    if psca_requested and not lm_head_full_requested:
        mark_only_psca_wr_as_trainable(model, train_classifier=True)
        freeze_lm_head_weight_for_psca_wr(model)
    print(f"Task: {args.data}")
    print(f"Method: {method_name}")
    print(f"Seed: {args.seed}")
    if not (pd_dft_requested and not lm_head_full_requested):
        print_trainable_parameter_names(model)
    print_sft_parameter_summary(model)
    print_adamix_sot_summary(model, peft_cfg)
    if sdft_requested and not lm_head_full_requested:
        print_sdft_summary(model, sdft_config, sdft_target_layers)
    if pd_dft_requested and not lm_head_full_requested:
        print_pd_dft_summary(model, pd_dft_config, pd_dft_target_layers)
    if snoft_requested and not lm_head_full_requested:
        print_snoft_summary(model, snoft_config, snoft_target_layers)
    if psca_requested and not lm_head_full_requested:
        print_psca_wr_summary(model, psca_config, psca_target_layers)

    print("Loaded model")

    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # tokenizer.eos_token = "<|endoftext|>"
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

    train_data_module = load_dataset(args.data, tokenizer, "train", return_module=True)

    # save_steps = its_per_epoch
    dataloader_num_workers = args.num_data_workers

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(args.output_dir) / "cfg.yaml", "w") as f:
        yaml.dump({key: value for key, value in vars(args).items() if key != "_explicit_keys"}, f)

    if args.eval_gen is not None:
        eval_generator = create_generator(tokenizer, **args.eval_gen)
    else:
        eval_generator = None

    val_data_module = load_dataset(
        args.val_data if args.val_data is not None else args.data,
        tokenizer,
        args.val_data_split,
        mode="lm" if args.eval_gen is None else "gen",
        return_module=True)

    compute_metrics = val_data_module.dataset.compute_metrics

    if args.debug:
        eval_type = val_data_module.dataset.eval_type
        eval_do_concat_batches = val_data_module.dataset.eval_do_concat_batches
        preprocess_logits_for_metrics = val_data_module.dataset.preprocess_logits_for_metrics

        train_data_module.dataset = torch.utils.data.Subset(train_data_module.dataset, range(8))
        val_data_module.dataset = torch.utils.data.Subset(val_data_module.dataset, range(2))
        val_data_module.dataset.eval_type = eval_type
        val_data_module.dataset.eval_do_concat_batches = eval_do_concat_batches
        val_data_module.dataset.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        args.num_epochs = 1

    its_per_epoch = int(np.ceil(len(train_data_module.dataset) / args.batch_size))
    logging_steps = min(50, its_per_epoch)
    if sdft_requested and sdft_config.sdft_log_interval is not None:
        logging_steps = sdft_config.sdft_log_interval
    if pd_dft_requested and pd_dft_config.pd_dft_log_interval is not None:
        logging_steps = pd_dft_config.pd_dft_log_interval
    if sdft_requested:
        sdft_config.sdft_log_interval = logging_steps
        if hasattr(model, "sdft_config"):
            model.sdft_config["sdft_log_interval"] = logging_steps
        if hasattr(model, "peft_args") and isinstance(model.peft_args.get("peft"), dict):
            model.peft_args["peft"]["sdft_log_interval"] = logging_steps
    if pd_dft_requested:
        pd_dft_config.pd_dft_log_interval = logging_steps
        if hasattr(model, "pd_dft_config"):
            model.pd_dft_config["pd_dft_log_interval"] = logging_steps
        if hasattr(model, "peft_args") and isinstance(model.peft_args.get("peft"), dict):
            model.peft_args["peft"]["pd_dft_log_interval"] = logging_steps

    run_name = str(args.output_dir).replace("weights/", "")
    # wandb.init()  # project='my_research'
    # wandb.run.name = run_name

    print("Dropping last batch")
    trainer = MambaTrainer(
        model=model,
        train_dataset=train_data_module.dataset,
        tokenizer=tokenizer,
        args=MambaTrainingArguments(
            learning_rate=args.learning_rate,
            # num_train_epochs=args.num_epochs,
            max_steps=int(args.num_epochs * its_per_epoch),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output_dir,
            logging_steps=logging_steps,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=2,
            eval_accumulation_steps=128,
            info={
                "trainable_params": get_trainable_parameters_ratio(model),
                "task": args.data,
                "method": method_name,
                "seed": args.seed,
                "bridge_enabled": getattr(args, "bridge_enabled", False),
                "bridge_all_layers": getattr(args, "bridge_all_layers", True),
                "lambda_lin": getattr(args, "lambda_lin", 0.01),
                "lambda_handoff": getattr(args, "lambda_handoff", 0.03),
                "bridge_use_soft_mixing": getattr(args, "bridge_use_soft_mixing", True),
                "bridge_mix_ratio_init": getattr(args, "bridge_mix_ratio_init", 1.0),
                "bridge_mix_decay_portion": getattr(args, "bridge_mix_decay_portion", 0.3),
                "final_no_attention_portion": getattr(args, "final_no_attention_portion", 0.1),
                "bridge_log_interval": getattr(args, "bridge_log_interval", 10),
                "num_experts": get_peft_attr(peft_cfg, "num_experts"),
                "consistency_lambda": get_peft_attr(peft_cfg, "consistency_lambda"),
                "use_consistency": get_peft_attr(peft_cfg, "use_consistency"),
                "inference_merge": get_peft_attr(peft_cfg, "inference_merge"),
                "use_sft": get_peft_attr(peft_cfg, "use_sft"),
                "sft_type": get_peft_attr(peft_cfg, "sft_type"),
                "sft_delta_scale": get_peft_attr(peft_cfg, "sft_delta_scale"),
                "sft_clamp": get_peft_attr(peft_cfg, "sft_clamp"),
                "use_tf_sot": get_peft_attr(peft_cfg, "use_tf_sot"),
                "tf_sot_num_freqs": get_peft_attr(peft_cfg, "tf_sot_num_freqs"),
                "tf_sot_num_basis": get_peft_attr(peft_cfg, "tf_sot_num_basis"),
                "tf_sot_eps": get_peft_attr(peft_cfg, "tf_sot_eps"),
                "tf_sot_max_seq_len": get_peft_attr(peft_cfg, "tf_sot_max_seq_len"),
                "tf_sot_freq_grid": get_peft_attr(peft_cfg, "tf_sot_freq_grid"),
                "tf_sot_normalize_basis": get_peft_attr(peft_cfg, "tf_sot_normalize_basis"),
                "use_cf_sot": get_peft_attr(peft_cfg, "use_cf_sot"),
                "cf_sot_num_freqs": get_peft_attr(peft_cfg, "cf_sot_num_freqs"),
                "cf_sot_num_basis": get_peft_attr(peft_cfg, "cf_sot_num_basis"),
                "cf_sot_eps": get_peft_attr(peft_cfg, "cf_sot_eps"),
                "cf_sot_max_seq_len": get_peft_attr(peft_cfg, "cf_sot_max_seq_len"),
                "cf_sot_freq_grid": get_peft_attr(peft_cfg, "cf_sot_freq_grid"),
                "cf_sot_normalize_basis": get_peft_attr(peft_cfg, "cf_sot_normalize_basis"),
                "cf_sot_context_center": get_peft_attr(peft_cfg, "cf_sot_context_center"),
                "cf_sot_context_reduce": get_peft_attr(peft_cfg, "cf_sot_context_reduce"),
                "use_sdft": get_peft_attr(peft_cfg, "use_sdft"),
                "sdft_rank": get_peft_attr(peft_cfg, "sdft_rank"),
                "sdft_rho_init": get_peft_attr(peft_cfg, "sdft_rho_init"),
                "sdft_gate_mode": get_peft_attr(peft_cfg, "sdft_gate_mode"),
                "sdft_dropout": get_peft_attr(peft_cfg, "sdft_dropout"),
                "sdft_target_layers": get_peft_attr(peft_cfg, "sdft_target_layers"),
                "sdft_freeze_base_model": get_peft_attr(peft_cfg, "sdft_freeze_base_model"),
                "sdft_train_classifier": get_peft_attr(peft_cfg, "sdft_train_classifier"),
                "sdft_log_stats": get_peft_attr(peft_cfg, "sdft_log_stats"),
                "sdft_log_interval": get_peft_attr(peft_cfg, "sdft_log_interval"),
                "sdft_log_per_layer": get_peft_attr(peft_cfg, "sdft_log_per_layer"),
                "sdft_log_grad": get_peft_attr(peft_cfg, "sdft_log_grad"),
                "use_pd_dft": get_peft_attr(peft_cfg, "use_pd_dft"),
                "pd_dft_rank": get_peft_attr(peft_cfg, "pd_dft_rank"),
                "pd_dft_dropout": get_peft_attr(peft_cfg, "pd_dft_dropout"),
                "pd_dft_rho_param_init": get_peft_attr(peft_cfg, "pd_dft_rho_param_init"),
                "pd_dft_rho_scan_init": get_peft_attr(peft_cfg, "pd_dft_rho_scan_init"),
                "pd_dft_learnable_rho": get_peft_attr(peft_cfg, "pd_dft_learnable_rho"),
                "pd_dft_mode": get_peft_attr(peft_cfg, "pd_dft_mode"),
                "pd_dft_target_layers": get_peft_attr(peft_cfg, "pd_dft_target_layers"),
                "pd_dft_share_down": get_peft_attr(peft_cfg, "pd_dft_share_down"),
                "pd_dft_max_delta_ratio_param": get_peft_attr(peft_cfg, "pd_dft_max_delta_ratio_param"),
                "pd_dft_max_delta_ratio_scan": get_peft_attr(peft_cfg, "pd_dft_max_delta_ratio_scan"),
                "pd_dft_log_stats": get_peft_attr(peft_cfg, "pd_dft_log_stats"),
                "pd_dft_log_per_layer": get_peft_attr(peft_cfg, "pd_dft_log_per_layer"),
                "pd_dft_log_grad": get_peft_attr(peft_cfg, "pd_dft_log_grad"),
                "pd_dft_log_interval": get_peft_attr(peft_cfg, "pd_dft_log_interval"),
                "use_snoft": get_peft_attr(peft_cfg, "use_snoft"),
                "snoft_enabled": get_peft_attr(peft_cfg, "enabled"),
                "snoft_num_groups": get_peft_attr(peft_cfg, "num_groups"),
                "snoft_chunk_size": get_peft_attr(peft_cfg, "chunk_size"),
                "snoft_router_rank": get_peft_attr(peft_cfg, "router_rank"),
                "snoft_tau_logit_init": get_peft_attr(peft_cfg, "tau_logit_init"),
                "snoft_freeze_backbone": get_peft_attr(peft_cfg, "freeze_backbone"),
                "snoft_train_task_head": get_peft_attr(peft_cfg, "train_task_head"),
                "snoft_target_layers": get_peft_attr(peft_cfg, "target_layers"),
                "use_psca_wr": get_peft_attr(peft_cfg, "use_psca_wr"),
                "psca_rank": get_peft_attr(peft_cfg, "psca_rank"),
                "psca_alpha": get_peft_attr(peft_cfg, "psca_alpha"),
                "psca_dropout": get_peft_attr(peft_cfg, "psca_dropout"),
                "psca_target_modules": get_peft_attr(peft_cfg, "psca_target_modules"),
                "psca_init_zero": get_peft_attr(peft_cfg, "psca_init_zero"),
                "psca_adapt_b": get_peft_attr(peft_cfg, "psca_adapt_b"),
                "psca_adapt_c": get_peft_attr(peft_cfg, "psca_adapt_c"),
                "psca_use_projector_shift": get_peft_attr(peft_cfg, "psca_use_projector_shift"),
                "psca_projector_residual": get_peft_attr(peft_cfg, "psca_projector_residual"),
                "psca_projector_scale": get_peft_attr(peft_cfg, "psca_projector_scale"),
                "psca_fallback_lite": get_peft_attr(peft_cfg, "psca_fallback_lite"),
                "psca_random_gate": get_peft_attr(peft_cfg, "psca_random_gate"),
                "psca_independent_gate": get_peft_attr(peft_cfg, "psca_independent_gate"),
                "use_deep_supervision": getattr(args, "use_deep_supervision", False),
                "aux_layers": getattr(args, "aux_layers", []),
                "aux_loss_weight": getattr(args, "aux_loss_weight", 0.1),
                "aux_weight_scheme": getattr(args, "aux_weight_scheme", "linear_increase"),
                "aux_pooling": getattr(args, "aux_pooling", "last_token"),
                "ds_adaptive": getattr(args, "ds_adaptive", False),
                "ds_adaptive_strategy": getattr(args, "ds_adaptive_strategy", "loss_drop"),
                "candidate_aux_layers": getattr(args, "candidate_aux_layers", []),
                "probe_ratio": getattr(args, "probe_ratio", 0.15),
                "probe_aux_weight": getattr(args, "probe_aux_weight", 0.05),
                "probe_start_window_ratio": getattr(args, "probe_start_window_ratio", 0.3),
                "probe_end_window_ratio": getattr(args, "probe_end_window_ratio", 0.3),
                "probe_loss_stat": getattr(args, "probe_loss_stat", "window_mean"),
                "adaptive_top_k": getattr(args, "adaptive_top_k", 3),
                "adaptive_late_bias_gamma": getattr(args, "adaptive_late_bias_gamma", 1.0),
                "adaptive_score_mode": getattr(args, "adaptive_score_mode", "drop_plus_confidence"),
                "adaptive_confidence_weight": getattr(args, "adaptive_confidence_weight", 0.1),
                "adaptive_score_threshold": getattr(args, "adaptive_score_threshold", 0.02),
                "adaptive_min_layer": getattr(args, "adaptive_min_layer", 0),
                "adaptive_disable_if_low_score": getattr(args, "adaptive_disable_if_low_score", False),
                "fallback_aux_layers": getattr(args, "fallback_aux_layers", [16, 20, 24]),
                "fallback_aux_weight_scheme": getattr(args, "fallback_aux_weight_scheme", "linear_increase"),
                "fallback_aux_loss_weight_scale": getattr(args, "fallback_aux_loss_weight_scale", 0.5),
                "fallback_confidence_threshold": getattr(args, "fallback_confidence_threshold", 0.0),
                "ds_schedule": getattr(args, "ds_schedule", "constant"),
                "ds_start_ratio": getattr(args, "ds_start_ratio", None),
                "ds_warmup_ratio": getattr(args, "ds_warmup_ratio", 0.3),
                # "peft_cfg": peft_cfg.to_dict() if peft_cfg is not None else None,
                "cfg_path": args.cfg_path,
                "device": os.environ.get("CUDA_VISIBLE_DEVICES", None),
            },
            save_strategy="steps" if not args.no_save else "no",
            evaluation_strategy="steps" if not args.skip_eval else "no",
            save_steps=int(args.eval_epochs * its_per_epoch),
            eval_steps=int(args.eval_epochs * its_per_epoch),
            load_best_model_at_end=(not args.no_save and not args.skip_eval and args.metric_for_best_model is not None),
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=True,
            dataloader_drop_last=val_data_module.dataset.eval_type != "log_likelihood", # only ll works for batch
            seed=args.seed,
            run_name=run_name,
            report_to=get_report_to(),
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=eval_generator,
        min_eval_metric_after_epoch=args.min_eval_metric_after_epoch,
        skip_metrics=args.skip_metrics,
        log_speed=args.log_speed
    )

    trainer.preprocess_logits_for_metrics = val_data_module.dataset.preprocess_logits_for_metrics

    # resume_from_checkpoint is bugged
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best validation score: {trainer.state.best_metric}")
    save_results_json(args, method_name, peft_cfg, trainer)
    # trainer.evaluate()

    if args.log_speed:
        with open(create_non_existent_file(Path(args.output_dir) / "train_timestamps.yaml"), "w") as f:
            yaml.safe_dump(trainer.train_timestamps, f)


def get_output_path_for_cfg(cfg_path):
    output_dir = str(Path(cfg_path).parent / Path(cfg_path).stem)
    output_dir = output_dir.replace("cfg/exps/", "")
    output_dir = Path("weights", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sdt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--task_name")
    parser.add_argument("--method")
    parser.add_argument("--peft")
    parser.add_argument("--prec")
    parser.add_argument("--device")
    parser.add_argument("--log_speed", action="store_true")
    parser.add_argument("--use_sdft", type=str2bool)
    parser.add_argument("--sdft_rank", type=int)
    parser.add_argument("--sdft_rho_init", type=float)
    parser.add_argument("--sdft_gate_mode", choices=["none", "z"])
    parser.add_argument("--sdft_dropout", type=float)
    parser.add_argument("--sdft_target_layers", nargs="*")
    parser.add_argument("--sdft_freeze_base_model", type=str2bool)
    parser.add_argument("--sdft_train_classifier", type=str2bool)
    parser.add_argument("--sdft_log_stats", type=str2bool)
    parser.add_argument("--sdft_log_interval", type=int)
    parser.add_argument("--sdft_log_per_layer", type=str2bool)
    parser.add_argument("--sdft_log_grad", type=str2bool)
    parser.add_argument("--use_pd_dft", type=str2bool)
    parser.add_argument("--pd_dft_rank", type=int)
    parser.add_argument("--pd_dft_dropout", type=float)
    parser.add_argument("--pd_dft_rho_param_init", type=float)
    parser.add_argument("--pd_dft_rho_scan_init", type=float)
    parser.add_argument("--pd_dft_learnable_rho", type=str2bool)
    parser.add_argument("--pd_dft_mode", choices=["param_only", "scan_only", "both"])
    parser.add_argument("--pd_dft_target_layers", nargs="*")
    parser.add_argument("--pd_dft_share_down", type=str2bool)
    parser.add_argument("--pd_dft_max_delta_ratio_param", type=float)
    parser.add_argument("--pd_dft_max_delta_ratio_scan", type=float)
    parser.add_argument("--pd_dft_log_stats", type=str2bool)
    parser.add_argument("--pd_dft_log_per_layer", type=str2bool)
    parser.add_argument("--pd_dft_log_grad", type=str2bool)
    parser.add_argument("--pd_dft_log_interval", type=int)
    parser.add_argument("--use_snoft", type=str2bool)
    parser.add_argument("--snoft_num_groups", type=int)
    parser.add_argument("--snoft_chunk_size", type=int)
    parser.add_argument("--snoft_router_rank", type=int)
    parser.add_argument("--snoft_tau_logit_init", type=float)
    parser.add_argument("--snoft_eps", type=float)
    parser.add_argument("--snoft_freeze_backbone", type=str2bool)
    parser.add_argument("--snoft_train_task_head", type=str2bool)
    parser.add_argument("--snoft_target_layers", nargs="*")
    parser.add_argument("--snoft_sanity_check", type=str2bool)
    parser.add_argument("--use_psca_wr", type=str2bool)
    parser.add_argument("--psca_rank", type=int)
    parser.add_argument("--psca_alpha", type=float)
    parser.add_argument("--psca_dropout", type=float)
    parser.add_argument("--psca_target_modules", nargs="*")
    parser.add_argument("--psca_init_zero", type=str2bool)
    parser.add_argument("--psca_adapt_b", type=str2bool)
    parser.add_argument("--psca_adapt_c", type=str2bool)
    parser.add_argument("--psca_use_projector_shift", type=str2bool)
    parser.add_argument("--psca_projector_residual", type=str2bool)
    parser.add_argument("--psca_projector_scale", type=float)
    parser.add_argument("--psca_fallback_lite", type=str2bool)
    parser.add_argument("--psca_random_gate", type=str2bool)
    parser.add_argument("--psca_independent_gate", type=str2bool)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--metric_for_best_model")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--tune_lm_head_only", type=str2bool)
    parser.add_argument("--use_deep_supervision", type=str2bool)
    parser.add_argument("--aux_layers", nargs="*", type=int)
    parser.add_argument("--aux_loss_weight", type=float)
    parser.add_argument("--aux_weight_scheme", choices=["uniform", "linear_increase", "adaptive"])
    parser.add_argument("--aux_pooling", choices=["last_token"])
    parser.add_argument("--ds_adaptive", type=str2bool)
    parser.add_argument("--ds_adaptive_strategy", choices=["loss_drop"])
    parser.add_argument("--candidate_aux_layers", nargs="*", type=int)
    parser.add_argument("--probe_ratio", type=float)
    parser.add_argument("--probe_aux_weight", type=float)
    parser.add_argument("--probe_start_window_ratio", type=float)
    parser.add_argument("--probe_end_window_ratio", type=float)
    parser.add_argument("--probe_loss_stat", choices=["window_mean", "ema"])
    parser.add_argument("--adaptive_top_k", type=int)
    parser.add_argument("--adaptive_late_bias_gamma", type=float)
    parser.add_argument("--adaptive_score_mode", choices=["loss_drop", "drop_plus_confidence"])
    parser.add_argument("--adaptive_confidence_weight", type=float)
    parser.add_argument("--adaptive_score_threshold", type=float)
    parser.add_argument("--adaptive_min_layer", type=int)
    parser.add_argument("--adaptive_disable_if_low_score", type=str2bool)
    parser.add_argument("--fallback_aux_layers", nargs="*", type=int)
    parser.add_argument("--fallback_aux_weight_scheme", choices=["uniform", "linear_increase"])
    parser.add_argument("--fallback_aux_loss_weight_scale", type=float)
    parser.add_argument("--fallback_confidence_threshold", type=float)
    parser.add_argument("--ds_schedule", choices=["constant", "linear_warmup"])
    parser.add_argument("--ds_start_ratio", type=float)
    parser.add_argument("--ds_warmup_ratio", type=float)
    parser.add_argument("--bridge_enabled", type=str2bool)
    parser.add_argument("--bridge_all_layers", type=str2bool)
    parser.add_argument("--lambda_lin", type=float)
    parser.add_argument("--lambda_handoff", type=float)
    parser.add_argument("--bridge_use_soft_mixing", type=str2bool)
    parser.add_argument("--bridge_mix_ratio_init", type=float)
    parser.add_argument("--bridge_mix_decay_portion", type=float)
    parser.add_argument("--final_no_attention_portion", type=float)
    parser.add_argument("--bridge_log_interval", type=int)
    args = parser.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    dft_cfg = {
        "tokenizer": "EleutherAI/gpt-neox-20b",
        "learning_rate": 5e-5,
        "batch_size": 4,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optim": "adamw_torch",
        "peft": None,
        "from_scratch": False,
        "skip_eval": False,
        "eval_epochs": 1,
        "val_data": None,
        "val_data_split": "val",
        "no_save": False,
        "backend": "cuda",
        "num_data_workers": 8,
        "model_transform": None,
        "repeat": None,
        "eval_gen": None,
        "min_eval_metric_after_epoch": None,
        "train_all_peft": False,
        "skip_metrics": False,
        "seed": 42,
        "log_speed": False,
        "metric_for_best_model": None,
        "method": None,
        "use_sdft": False,
        "sdft_rank": 4,
        "sdft_rho_init": 0.05,
        "sdft_gate_mode": "none",
        "sdft_dropout": 0.05,
        "sdft_target_layers": "all",
        "sdft_freeze_base_model": True,
        "sdft_train_classifier": True,
        "sdft_log_stats": True,
        "sdft_log_interval": None,
        "sdft_log_per_layer": False,
        "sdft_log_grad": True,
        "use_pd_dft": False,
        "pd_dft_rank": 4,
        "pd_dft_dropout": 0.05,
        "pd_dft_rho_param_init": 0.05,
        "pd_dft_rho_scan_init": 0.05,
        "pd_dft_learnable_rho": True,
        "pd_dft_mode": "both",
        "pd_dft_target_layers": "all",
        "pd_dft_share_down": True,
        "pd_dft_max_delta_ratio_param": None,
        "pd_dft_max_delta_ratio_scan": None,
        "pd_dft_log_stats": True,
        "pd_dft_log_per_layer": False,
        "pd_dft_log_grad": True,
        "pd_dft_log_interval": None,
        "use_snoft": False,
        "snoft": None,
        "snoft_num_groups": 16,
        "snoft_chunk_size": 32,
        "snoft_router_rank": 8,
        "snoft_tau_logit_init": 3.0,
        "snoft_eps": 1e-6,
        "snoft_freeze_backbone": True,
        "snoft_train_task_head": True,
        "snoft_target_layers": "all",
        "snoft_sanity_check": True,
        "use_psca_wr": False,
        "psca_rank": 8,
        "psca_alpha": 1.0,
        "psca_dropout": 0.0,
        "psca_target_modules": "all",
        "psca_init_zero": True,
        "psca_adapt_b": True,
        "psca_adapt_c": True,
        "psca_use_projector_shift": True,
        "psca_projector_residual": True,
        "psca_projector_scale": 0.01,
        "psca_fallback_lite": False,
        "psca_random_gate": False,
        "psca_independent_gate": False,
        "psca_debug": False,
        "tune_lm_head_only": False,
        "use_deep_supervision": False,
        "aux_layers": [],
        "aux_loss_weight": 0.1,
        "aux_weight_scheme": "linear_increase",
        "aux_pooling": "last_token",
        "ds_adaptive": False,
        "ds_adaptive_strategy": "loss_drop",
        "candidate_aux_layers": [4, 8, 12, 16, 20, 24],
        "probe_ratio": 0.15,
        "probe_aux_weight": 0.05,
        "probe_start_window_ratio": 0.3,
        "probe_end_window_ratio": 0.3,
        "probe_loss_stat": "window_mean",
        "adaptive_top_k": 3,
        "adaptive_late_bias_gamma": 1.0,
        "adaptive_score_mode": "drop_plus_confidence",
        "adaptive_confidence_weight": 0.1,
        "adaptive_score_threshold": 0.02,
        "adaptive_min_layer": 0,
        "adaptive_disable_if_low_score": False,
        "fallback_aux_layers": [16, 20, 24],
        "fallback_aux_weight_scheme": "linear_increase",
        "fallback_aux_loss_weight_scale": 0.5,
        "fallback_confidence_threshold": 0.0,
        "ds_schedule": "constant",
        "ds_start_ratio": None,
        "ds_warmup_ratio": 0.3,
        "bridge_enabled": True,
        "bridge_all_layers": True,
        "lambda_lin": 0.01,
        "lambda_handoff": 0.03,
        "bridge_use_soft_mixing": True,
        "bridge_mix_ratio_init": 1.0,
        "bridge_mix_decay_portion": 0.3,
        "final_no_attention_portion": 0.1,
        "bridge_log_interval": 10,
    }

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f) or {}

    args_dict = vars(args)
    cli_output_dir = args_dict.pop("output_dir", None)

    for key in list(args_dict.keys()):
        if args_dict[key] is None:
            del args_dict[key]

    explicit_keys = set(cfg.keys()) | set(args_dict.keys())
    output_dir = cli_output_dir or cfg.get("output_dir") or get_output_path_for_cfg(args.cfg)
    args = {**dft_cfg, **cfg, **args_dict, "output_dir": str(output_dir), "cfg_path": args.cfg, "_explicit_keys": explicit_keys}

    if args.get("model_name_or_path") is not None:
        args["model"] = args["model_name_or_path"]
    if args.get("task_name") is not None:
        args["data"] = normalize_data_name(args["task_name"])
    if args.get("num_train_epochs") is not None:
        args["num_epochs"] = args["num_train_epochs"]
    if args.get("ds_adaptive"):
        if "aux_weight_scheme" not in explicit_keys:
            args["aux_weight_scheme"] = "adaptive"
        if "ds_schedule" not in explicit_keys:
            args["ds_schedule"] = "linear_warmup"
        if "ds_start_ratio" not in explicit_keys:
            args["ds_start_ratio"] = args["probe_ratio"]

    if args["metric_for_best_model"] is None:
        args["metric_for_best_model"] = get_default_best_metric(args["data"])

    if args["repeat"] is None:
        run_train(SimpleNamespace(**args))
    else:
        for i in range(args["repeat"]):
            print(f"Starting run {i}")
            args["output_dir"] = str(Path(output_dir) / f"{i:03d}")
            run_train(SimpleNamespace(**args))


if __name__ == "__main__":
    main()
