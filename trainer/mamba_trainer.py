from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from transformers.trainer import logger
from transformers.trainer_utils import denumpify_detensorize
import torch
import numpy as np
from torch import nn
import os
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
import yaml
from yaml import CSafeLoader
from peft import PeftModel

from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from modules import load_mamba, save_mamba
from modules.deep_supervision import (
    compute_deep_supervision_loss,
    get_deep_supervision_required_layers,
    get_deep_supervision_loss_weight,
    get_mamba_backbone_model,
    is_deep_supervision_enabled,
    set_latest_loss_dict,
)
from modules.sdft import collect_sdft_grad_stats, collect_sdft_stats, has_sdft_adapters
from modules.pd_dft import collect_pd_dft_grad_stats, collect_pd_dft_stats, has_pd_dft_adapters
from modules.s6_attention_bridge import (
    compute_s6_attention_bridge_loss,
    get_s6_attention_bridge_log_interval,
    reset_s6_attention_bridge_losses,
    s6_attention_bridge_is_enabled,
    set_s6_attention_bridge_runtime,
)
from trainer.loss import CrossEntropy, Accuracy

import torch.nn.functional as F



class MambaEvalPrediction:
    def __init__(self, tokenizer=None, input_ids=None, pred_ids=None, label_ids=None, save_file=None, remove_eos=False):
        self.tokenizer = tokenizer

        self.inputs = tokenizer.batch_decode(self.remove_pad_token_id(input_ids) if remove_eos else input_ids) if input_ids is not None else None
        self.preds = tokenizer.batch_decode(self.remove_eos_token_id(pred_ids) if remove_eos else pred_ids) if pred_ids is not None else None
        self.labels = tokenizer.batch_decode(self.remove_eos_token_id(label_ids) if remove_eos else label_ids) if label_ids is not None else None

        self.input_ids = [t.cpu().numpy() for t in input_ids] if input_ids is not None else None
        self.pred_ids = [t.cpu().numpy() for t in pred_ids] if pred_ids is not None else None
        self.label_ids = [t.cpu().numpy() for t in label_ids] if label_ids is not None else None

        self.save_file = save_file

    def remove_pad_token_id(self, ids):
        ids_no_eos = [(id if id[-1] != self.tokenizer.pad_token_id else id[:-1])  for id in ids]
        return ids_no_eos

    def remove_eos_token_id(self, ids):
        eos_token_id = self.tokenizer.eos_token_id

        ids_no_eos = [(id if id[-1] != eos_token_id else id[:-1])  for id in ids]
        return ids_no_eos

    @staticmethod
    def from_file(path):
        p = MambaEvalPrediction()
        p.load(path)
        return p

    def load(self, path):
        with open(path, "r") as f:
            state = yaml.load(f, Loader=CSafeLoader)

        self.inputs = state["inputs"]
        self.preds = state["preds"]
        self.labels = state["labels"]
        self.input_ids = [np.array(x) for x in state["input_ids"]]
        self.pred_ids = [np.array(x) for x in state["pred_ids"]]
        self.label_ids = [np.array(x) for x in state["label_ids"]]
        self.save_file = path

    def save(self, path=None):
        if path is None:
            path = self.save_file

        out_dict = dict(
            inputs=self.inputs,
            preds=self.preds,
            labels=self.labels,
            input_ids=[t.astype(int).tolist() for t in self.input_ids],
            pred_ids=[t.astype(int).tolist() for t in self.pred_ids],
            label_ids=[t.astype(int).tolist() for t in self.label_ids],
        )

        Path(path).parent.mkdir(exist_ok=True, parents=True)

        with open(path, "w") as f:
            yaml.safe_dump(out_dict, f, sort_keys=False)


class MambaLogLikelihoodPrediction:
    def __init__(self, input_ids, pred_lls, label_ids) -> None:
        self.input_ids = input_ids
        self.pred_lls = pred_lls
        self.label_ids = label_ids


class TrainLossEarlyStop:
    def __init__(self) -> None:
        self.nan_limit = 10
        self.consec_nans = 0
        self.should_stop = False

    def __call__(self, control, train_loss) -> Any:
        train_loss = train_loss.item()

        if np.isnan(train_loss) or train_loss <= 1.e-6:
            self.consec_nans += 1

            if self.consec_nans >= self.nan_limit:
                print(f"Stopping after {self.consec_nans} 0/nan losses")
                self.should_stop = True
                control.should_training_stop = True
        else:
            self.consec_nans = 0


class BadEvalEarlyStop:
    def __init__(self, eval_after_epochs, metric=None):
        self.eval_after_epochs = eval_after_epochs
        self.metric = metric

    def __call__(self, control, metrics) -> Any:
        epoch = int(metrics["epoch"])

        if epoch in self.eval_after_epochs:
            metric = self.metric if self.metric is not None else next(iter(metrics.keys()))
            min_val = self.eval_after_epochs[epoch]
            val = metrics[metric]

            if val < min_val:
                control.should_training_stop = True


@dataclass
class MambaTrainingArguments(TrainingArguments):
    info: Dict[str, Any] = field(default=None)


class MambaTrainer(Trainer):
    def __init__(self, 
                 model: PreTrainedModel | Module = None, 
                 args: TrainingArguments = None, 
                 data_collator: Any | None = None, 
                 train_dataset: Dataset | None = None, 
                 eval_dataset: Dataset | Dict[str, Dataset] | None = None, 
                 tokenizer: PreTrainedTokenizerBase | None = None, 
                 model_init: Callable[[], PreTrainedModel] | None = None, 
                 compute_metrics: Callable[[EvalPrediction], Dict] | None = None, 
                 callbacks: List[TrainerCallback] | None = None, 
                 optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                 eval_generator=None,
                 min_eval_metric_after_epoch=None,
                 log_speed=False,
                 skip_metrics=False):
        # args.include_inputs_for_metrics = True
        if callbacks is None:
            callbacks = []

        args.eval_do_concat_batches = eval_dataset.eval_do_concat_batches if eval_dataset is not None else None

        """
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        """

        super_extra_kwargs = {}

        if "processing_class" in Trainer.__init__.__code__.co_names:  
            # for new transformers versions  
            super_extra_kwargs["processing_class"] = tokenizer
        else:
            super_extra_kwargs["tokenizer"] = tokenizer

        super().__init__(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset, 
                         eval_dataset=eval_dataset, 
                         model_init=model_init, compute_metrics=compute_metrics, callbacks=callbacks, 
                         optimizers=optimizers, preprocess_logits_for_metrics=preprocess_logits_for_metrics, **super_extra_kwargs)
        
        self.train_crit = CrossEntropy()
        self.val_crits = [Accuracy()]
        self.train_loss_early_stop = TrainLossEarlyStop()
        self.eval_generator = eval_generator
        self.min_eval_metric_after_epoch_early_stop = BadEvalEarlyStop(min_eval_metric_after_epoch) if min_eval_metric_after_epoch is not None else None
        self.skip_metrics = skip_metrics
        self.run_name = args.run_name
        self.wandb_init = False
        self.train_timestamps = [] if log_speed else None

        if hasattr(model, "load_config"):
            model.load_config(self.args.output_dir)

    def _get_sdft_config(self, model=None):
        if model is None:
            model = self.model
        candidates = [model]
        for attr in ("module", "model", "base_model"):
            candidate = getattr(model, attr, None)
            if candidate is not None:
                candidates.append(candidate)
        for candidate in candidates:
            config = getattr(candidate, "sdft_config", None)
            if isinstance(config, dict):
                return config
        return {}

    def _get_pd_dft_config(self, model=None):
        if model is None:
            model = self.model
        candidates = [model]
        for attr in ("module", "model", "base_model"):
            candidate = getattr(model, attr, None)
            if candidate is not None:
                candidates.append(candidate)
        for candidate in candidates:
            config = getattr(candidate, "pd_dft_config", None)
            if isinstance(config, dict):
                return config
        return {}

    @staticmethod
    def _format_param_count(value):
        if value is None:
            return "n/a"
        value = float(value)
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if value >= 1_000:
            return f"{value / 1_000:.2f}K"
        return f"{value:.0f}"

    @staticmethod
    def _format_float(value):
        if value is None:
            return "n/a"
        return f"{float(value):.3g}"

    def _print_sdft_compact_summary(self, logs):
        ratio_mean = logs.get("sdft/global/mean_delta_to_v_rms_ratio")
        ratio_max = logs.get("sdft/global/max_delta_to_v_rms_ratio")
        gate_mean = logs.get("sdft/global/mean_gate")
        rho_mean = logs.get("sdft/global/mean_rho")
        grad_up_mean = logs.get("sdft/global/mean_up_grad_norm")
        grad_down_mean = logs.get("sdft/global/mean_down_grad_norm")
        trainable = logs.get("sdft/global/trainable_param_count")
        if ratio_mean is None and rho_mean is None:
            return
        print(
            "[SDFT] "
            f"step={self.state.global_step} "
            f"ratio_mean={self._format_float(ratio_mean)} "
            f"ratio_max={self._format_float(ratio_max)} "
            f"gate_mean={self._format_float(gate_mean)} "
            f"rho_mean={self._format_float(rho_mean)} "
            f"grad_up_mean={self._format_float(grad_up_mean)} "
            f"grad_down_mean={self._format_float(grad_down_mean)} "
            f"trainable={self._format_param_count(trainable)}"
        )

    def _format_active_float(self, value, active=True):
        if not active:
            return "inactive"
        return self._format_float(value)

    def _print_pd_dft_compact_summary(self, logs):
        config = self._get_pd_dft_config()
        mode = str(config.get("pd_dft_mode", "both"))
        param_active = mode in ("param_only", "both")
        scan_active = mode in ("scan_only", "both")
        p_ratio = logs.get("pd_dft/global/mean_delta_param_to_v_ratio")
        s_ratio = logs.get("pd_dft/global/mean_delta_scan_to_v_ratio")
        p_max = logs.get("pd_dft/global/max_delta_param_to_v_ratio")
        s_max = logs.get("pd_dft/global/max_delta_scan_to_v_ratio")
        rho_p = logs.get("pd_dft/global/mean_rho_param")
        rho_s = logs.get("pd_dft/global/mean_rho_scan")
        grad_p = logs.get("pd_dft/global/mean_up_param_grad_norm")
        grad_s = logs.get("pd_dft/global/mean_up_scan_grad_norm")
        trainable = logs.get("pd_dft/global/trainable_param_count")
        if p_ratio is None and s_ratio is None and rho_p is None and rho_s is None:
            return
        print(
            "[PD-DFT] "
            f"step={self.state.global_step} "
            f"mode={mode} "
            f"p_ratio={self._format_active_float(p_ratio, param_active)} "
            f"s_ratio={self._format_active_float(s_ratio, scan_active)} "
            f"p_max={self._format_active_float(p_max, param_active)} "
            f"s_max={self._format_active_float(s_max, scan_active)} "
            f"rho_p={self._format_active_float(rho_p, param_active)} "
            f"rho_s={self._format_active_float(rho_s, scan_active)} "
            f"grad_p={self._format_active_float(grad_p, param_active)} "
            f"grad_s={self._format_active_float(grad_s, scan_active)} "
            f"trainable={self._format_param_count(trainable)}"
        )

    @staticmethod
    def _without_sdft_layer_logs(logs):
        return {
            key: value
            for key, value in logs.items()
            if not str(key).startswith("sdft/layer_")
        }

    @staticmethod
    def _without_pd_dft_layer_logs(logs):
        return {
            key: value
            for key, value in logs.items()
            if not str(key).startswith("pd_dft/layer_")
        }

    def _write_sdft_jsonl(self, logs):
        if not logs:
            return
        payload = {
            "step": int(self.state.global_step),
            "epoch": None if self.state.epoch is None else float(self.state.epoch),
            **logs,
        }
        payload = denumpify_detensorize(payload)
        path = Path(self.args.output_dir) / "sdft_stats.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def _write_pd_dft_jsonl(self, logs):
        if not logs:
            return
        payload = {
            "step": int(self.state.global_step),
            "epoch": None if self.state.epoch is None else float(self.state.epoch),
            **logs,
        }
        payload = denumpify_detensorize(payload)
        path = Path(self.args.output_dir) / "pd_dft_stats.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def _log_sdft_external(self, logs):
        if not logs:
            return

        per_layer_logs = {
            key: value
            for key, value in logs.items()
            if str(key).startswith("sdft/layer_")
        }
        if not per_layer_logs:
            return

        # Keep full per-layer SDFT telemetry out of TrainerState.log_history.
        # It is written only when sdft_log_per_layer is explicitly enabled.
        self._write_sdft_jsonl(per_layer_logs)

        if wandb is not None and wandb.run is not None:
            try:
                wandb.log(per_layer_logs, step=self.state.global_step)
            except Exception as exc:
                logger.warning(f"Failed to log SDFT per-layer metrics to wandb: {exc}")

        callback_handler = getattr(self, "callback_handler", None)
        callbacks = getattr(callback_handler, "callbacks", []) if callback_handler is not None else []
        for callback in callbacks:
            writer = getattr(callback, "tb_writer", None)
            if writer is None:
                continue
            try:
                for key, value in per_layer_logs.items():
                    writer.add_scalar(key, value, self.state.global_step)
                writer.flush()
            except Exception as exc:
                logger.warning(f"Failed to log SDFT per-layer metrics to tensorboard: {exc}")

    def _log_pd_dft_external(self, logs):
        if not logs:
            return

        per_layer_logs = {
            key: value
            for key, value in logs.items()
            if str(key).startswith("pd_dft/layer_")
        }
        if not per_layer_logs:
            return

        self._write_pd_dft_jsonl(per_layer_logs)

        if wandb is not None and wandb.run is not None:
            try:
                wandb.log(per_layer_logs, step=self.state.global_step)
            except Exception as exc:
                logger.warning(f"Failed to log PD-DFT per-layer metrics to wandb: {exc}")

        callback_handler = getattr(self, "callback_handler", None)
        callbacks = getattr(callback_handler, "callbacks", []) if callback_handler is not None else []
        for callback in callbacks:
            writer = getattr(callback, "tb_writer", None)
            if writer is None:
                continue
            try:
                for key, value in per_layer_logs.items():
                    writer.add_scalar(key, value, self.state.global_step)
                writer.flush()
            except Exception as exc:
                logger.warning(f"Failed to log PD-DFT per-layer metrics to tensorboard: {exc}")

    def log(self, *args, **kwargs):
        logs = None
        if len(args) > 0 and isinstance(args[0], dict):
            logs = dict(args[0])
            args = (logs, *args[1:])
        elif isinstance(kwargs.get("logs"), dict):
            logs = dict(kwargs["logs"])
            kwargs["logs"] = logs

        if logs is not None and has_sdft_adapters(self.model):
            sdft_config = self._get_sdft_config()
            log_stats = bool(sdft_config.get("sdft_log_stats", True))
            log_per_layer = bool(sdft_config.get("sdft_log_per_layer", False))
            log_grad = bool(sdft_config.get("sdft_log_grad", True))
            sdft_full_logs = {}
            if log_stats:
                sdft_stats = collect_sdft_stats(self.model, clear=True, log_per_layer=log_per_layer)
                if sdft_stats:
                    sdft_full_logs.update(sdft_stats)
                    logs.update(self._without_sdft_layer_logs(sdft_stats))
            if log_grad:
                grad_stats = getattr(self, "_latest_sdft_grad_stats", None)
                if grad_stats:
                    sdft_full_logs.update(grad_stats)
                    logs.update(self._without_sdft_layer_logs(grad_stats))
                self._latest_sdft_grad_stats = None
            if any(key.startswith("sdft/") for key in logs):
                self._print_sdft_compact_summary(logs)
            self._pending_sdft_full_logs = sdft_full_logs

        if logs is not None and has_pd_dft_adapters(self.model):
            pd_dft_config = self._get_pd_dft_config()
            log_stats = bool(pd_dft_config.get("pd_dft_log_stats", True))
            log_per_layer = bool(pd_dft_config.get("pd_dft_log_per_layer", False))
            log_grad = bool(pd_dft_config.get("pd_dft_log_grad", True))
            pd_dft_full_logs = {}
            if log_stats:
                pd_dft_stats = collect_pd_dft_stats(self.model, clear=True, log_per_layer=log_per_layer)
                if pd_dft_stats:
                    pd_dft_full_logs.update(pd_dft_stats)
                    logs.update(self._without_pd_dft_layer_logs(pd_dft_stats))
            if log_grad:
                grad_stats = getattr(self, "_latest_pd_dft_grad_stats", None)
                if grad_stats:
                    pd_dft_full_logs.update(grad_stats)
                    logs.update(self._without_pd_dft_layer_logs(grad_stats))
                self._latest_pd_dft_grad_stats = None
            if any(key.startswith("pd_dft/") for key in logs):
                self._print_pd_dft_compact_summary(logs)
            self._pending_pd_dft_full_logs = pd_dft_full_logs

        super().log(*args, **kwargs)

        sdft_full_logs = getattr(self, "_pending_sdft_full_logs", None)
        if sdft_full_logs:
            self._log_sdft_external(sdft_full_logs)
            self._pending_sdft_full_logs = None

        pd_dft_full_logs = getattr(self, "_pending_pd_dft_full_logs", None)
        if pd_dft_full_logs:
            self._log_pd_dft_external(pd_dft_full_logs)
            self._pending_pd_dft_full_logs = None

        if wandb is not None and wandb.run is not None and not self.wandb_init and not self.args.report_to == []:
            wandb.run.name = self.run_name
            self.wandb_init = True

    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)
        if has_sdft_adapters(model) and bool(self._get_sdft_config(model).get("sdft_log_grad", True)):
            self._latest_sdft_grad_stats = collect_sdft_grad_stats(
                model,
                log_per_layer=bool(self._get_sdft_config(model).get("sdft_log_per_layer", False)),
            )
        if has_pd_dft_adapters(model) and bool(self._get_pd_dft_config(model).get("pd_dft_log_grad", True)):
            self._latest_pd_dft_grad_stats = collect_pd_dft_grad_stats(
                model,
                log_per_layer=bool(self._get_pd_dft_config(model).get("pd_dft_log_per_layer", False)),
            )
        return loss

    def log_train_seq(self, input_ids, label_ids, lm_logits, idx=0):
        input_ids, label_ids, lm_logits = input_ids[idx], label_ids[idx], lm_logits[idx]

        output_ids = lm_logits.argmax(-1)

        valid_ids = label_ids != -100

        input_txt = self.tokenizer.decode(input_ids)
        input_txt_valid = self.tokenizer.decode(input_ids[valid_ids])
        label_txt_valid = self.tokenizer.decode(label_ids[valid_ids])
        output_txt_valid = self.tokenizer.decode(output_ids[valid_ids])

        print(input_txt)
        print(input_txt_valid, "->", label_txt_valid)
        print(output_txt_valid, "==", label_txt_valid)

    def _extract_hidden_states_from_outputs(self, outputs):
        if hasattr(outputs, "hidden_states"):
            return outputs.hidden_states
        if isinstance(outputs, dict):
            return outputs.get("hidden_states")
        return None

    def _register_mamba_hidden_state_hooks(self, model):
        backbone_model = get_mamba_backbone_model(model)
        layers = backbone_model.backbone.layers
        captured_hidden_states = [None] * (len(layers) + 1)
        handles = []

        def _make_hook(layer_number):
            def _hook(_module, _module_inputs, module_output):
                # Block.forward returns (hidden_states, residual). Keep the
                # post-block hidden states at the same 1-based index used by
                # deep supervision configs, leaving slot 0 as the embedding.
                hidden = module_output[0] if isinstance(module_output, (tuple, list)) else module_output
                captured_hidden_states[layer_number] = hidden

            return _hook

        for layer_number, layer in enumerate(layers, start=1):
            handles.append(layer.register_forward_hook(_make_hook(layer_number)))
        return captured_hidden_states, handles

    def _hidden_states_cover_required_layers(self, hidden_states, required_layers):
        if hidden_states is None:
            return False
        if not required_layers:
            return True
        if len(hidden_states) <= max(required_layers):
            return False
        return all(hidden_states[layer] is not None for layer in required_layers)

    def _resolve_deep_supervision_hidden_states(self, model, outputs, captured_hidden_states):
        returned_hidden_states = self._extract_hidden_states_from_outputs(outputs)
        required_layers = get_deep_supervision_required_layers(model)

        if self._hidden_states_cover_required_layers(returned_hidden_states, required_layers):
            return returned_hidden_states
        if self._hidden_states_cover_required_layers(captured_hidden_states, required_layers):
            return tuple(captured_hidden_states)

        returned_len = 0 if returned_hidden_states is None else len(returned_hidden_states)
        captured_len = 0 if captured_hidden_states is None else len(captured_hidden_states)
        raise RuntimeError(
            "Deep supervision is enabled, but hidden_states for the required Mamba layers were not collected. "
            f"required_layers={required_layers}, returned_hidden_states_len={returned_len}, "
            f"captured_hidden_states_len={captured_len}. "
            "The Trainer requested output_hidden_states=True and also installed Mamba block hooks; "
            "please check that the model forward actually executes backbone.layers."
        )

    def _forward(self, model, inputs, output_hidden_states=False):
        if wandb is not None and not self.wandb_init and wandb.run is not None:
            wandb.run.name = self.run_name
            self.wandb_init = True

        input_ids = inputs["input_ids"]
        label_ids = inputs["label_ids"]

        add_inputs = {}
        if output_hidden_states:
            add_inputs["output_hidden_states"] = True
        if "attention_mask" in inputs:
            add_inputs["attention_mask"] = inputs["attention_mask"]

        if isinstance(model, PeftModel):
            base = model.base_model

            # if "label_ids" in base.forward.__code__.co_varnames:
            #     add_inputs["label_ids"] = label_ids

        captured_hidden_states = None
        hook_handles = []
        if output_hidden_states:
            captured_hidden_states, hook_handles = self._register_mamba_hidden_state_hooks(model)

        try:
            outputs = model(input_ids, **add_inputs)
        finally:
            for handle in hook_handles:
                handle.remove()

        lm_logits = outputs.logits
        hidden_states = None
        if output_hidden_states:
            hidden_states = self._resolve_deep_supervision_hidden_states(
                model,
                outputs,
                captured_hidden_states,
            )

        return input_ids, label_ids, lm_logits, outputs, hidden_states

    def _get_active_peft_config(self, model):
        while hasattr(model, "module"):
            model = model.module

        peft_config = getattr(model, "peft_config", None)
        if isinstance(peft_config, dict):
            active_adapter = getattr(model, "active_adapter", None)
            if active_adapter in peft_config:
                return peft_config[active_adapter]
            if len(peft_config) > 0:
                return next(iter(peft_config.values()))
        return peft_config

    def _get_adamix_sot_consistency_config(self, model):
        peft_config = self._get_active_peft_config(model)
        method = getattr(peft_config, "method", None)

        if method != "adamix_sot":
            return None
        if not bool(getattr(peft_config, "use_consistency", False)):
            return None
        if not model.training:
            return None

        return {
            "lambda": float(getattr(peft_config, "consistency_lambda", 1.0)),
        }

    def _compute_symmetric_kl_consistency(self, logits_a, logits_b, label_ids):
        if logits_a.shape != logits_b.shape:
            raise ValueError(
                f"AdaMix-SOT consistency logits shape mismatch: {tuple(logits_a.shape)} vs {tuple(logits_b.shape)}"
            )

        valid_pos = label_ids != self.train_crit.ignore_index
        if not bool(valid_pos.any()):
            return logits_a.new_zeros(())

        logits_a = logits_a[valid_pos].float()
        logits_b = logits_b[valid_pos].float()

        log_p_a = F.log_softmax(logits_a, dim=-1)
        log_p_b = F.log_softmax(logits_b, dim=-1)

        # Each direction gives gradients to its input logits. Detaching the
        # target side avoids unstable gradients through near-zero probabilities.
        kl_ab = F.kl_div(log_p_a, log_p_b.detach(), reduction="batchmean", log_target=True)
        kl_ba = F.kl_div(log_p_b, log_p_a.detach(), reduction="batchmean", log_target=True)
        return 0.5 * (kl_ab + kl_ba)

    def log_iter(self, metrics, interval):
        if (self.state.global_step + 1) % interval == 0:
            self.log(metrics)

    def _get_train_progress(self):
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps <= 0:
            max_steps = int(getattr(self.args, "max_steps", 0) or 0)
        max_steps = max(1, max_steps)
        return min(1.0, max(0.0, float(self.state.global_step) / float(max_steps)))

    def _prepare_s6_bridge_forward(self, model, progress):
        set_s6_attention_bridge_runtime(model, progress=progress)
        reset_s6_attention_bridge_losses(model)

    def _disable_s6_bridge_for_eval(self, model):
        if s6_attention_bridge_is_enabled(model) and not getattr(self, "_s6_bridge_eval_notice_printed", False):
            print("[S6 bridge] eval/predict: bridge disabled; no softmax attention is constructed.")
            self._s6_bridge_eval_notice_printed = True
        self._prepare_s6_bridge_forward(model, progress=1.0)

    @staticmethod
    def _average_bridge_logs(*logs):
        logs = [log for log in logs if log]
        if not logs:
            return {}
        merged = {}
        for key in sorted(set().union(*(log.keys() for log in logs))):
            values = [log[key] for log in logs if key in log]
            first = values[0]
            if torch.is_tensor(first):
                if first.numel() == 1:
                    merged[key] = torch.stack([value.reshape(()) for value in values]).mean()
                else:
                    merged[key] = torch.stack([value.detach().float() for value in values]).mean()
            elif isinstance(first, (int, float)):
                merged[key] = sum(float(value) for value in values) / len(values)
            else:
                merged[key] = first
        return merged

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        adamix_consistency_config = self._get_adamix_sot_consistency_config(model)

        use_aux_hidden_states = is_deep_supervision_enabled(model)
        progress = self._get_train_progress()
        self._prepare_s6_bridge_forward(model, progress)
        input_ids, label_ids, lm_logits, outputs, hidden_states = self._forward(
            model,
            inputs,
            output_hidden_states=use_aux_hidden_states,
        )
        task_loss_a = self.train_crit(lm_logits, label_ids)
        bridge_loss_a, bridge_log_a = compute_s6_attention_bridge_loss(model)

        if adamix_consistency_config is not None:
            self._prepare_s6_bridge_forward(model, progress)
            _, _, lm_logits_b, _, _ = self._forward(model, inputs)
            task_loss_b = self.train_crit(lm_logits_b, label_ids)
            bridge_loss_b, bridge_log_b = compute_s6_attention_bridge_loss(model)
            task_loss = 0.5 * (task_loss_a + task_loss_b)
            consistency_loss = self._compute_symmetric_kl_consistency(lm_logits, lm_logits_b, label_ids)
            consistency_lambda = adamix_consistency_config["lambda"]
            lm_loss = task_loss + consistency_lambda * consistency_loss
            if bridge_loss_a is not None and bridge_loss_b is not None:
                bridge_aux_loss = 0.5 * (bridge_loss_a + bridge_loss_b)
            else:
                bridge_aux_loss = bridge_loss_a if bridge_loss_a is not None else bridge_loss_b
            bridge_log = self._average_bridge_logs(bridge_log_a, bridge_log_b)
            self.log_iter(
                {
                    "task_loss": task_loss.item(),
                    "task_loss_a": task_loss_a.item(),
                    "task_loss_b": task_loss_b.item(),
                    "task_loss_mode": "two_sided",
                    "consistency_loss": consistency_loss.item(),
                    "consistency_lambda": consistency_lambda,
                },
                10,
            )
        else:
            lm_loss = task_loss_a
            task_loss = task_loss_a
            bridge_aux_loss = bridge_loss_a
            bridge_log = bridge_log_a

        if bridge_aux_loss is not None:
            lm_loss = lm_loss + bridge_aux_loss
            bridge_metrics = {
                "task_loss": task_loss.detach(),
                "progress": progress,
                "bridge_eval_disabled": 0,
                **bridge_log,
            }
            self.log_iter(
                {
                    key: float(value.detach().cpu()) if torch.is_tensor(value) else value
                    for key, value in bridge_metrics.items()
                },
                get_s6_attention_bridge_log_interval(model),
            )

        main_loss = lm_loss
        if use_aux_hidden_states:
            aux_loss_total, aux_loss_dict = compute_deep_supervision_loss(
                model,
                hidden_states,
                label_ids,
                progress=progress,
            )
            aux_loss_weight = get_deep_supervision_loss_weight(model, progress=progress)
            if aux_loss_total is not None:
                lm_loss = lm_loss + aux_loss_weight * aux_loss_total

            if aux_loss_total is not None or len(aux_loss_dict) > 0:
                loss_log = {
                    "main_loss": main_loss.detach(),
                    "scheduled_aux_weight": aux_loss_weight,
                    "progress": progress,
                    "total_loss": lm_loss.detach(),
                    **aux_loss_dict,
                }
                if aux_loss_total is not None:
                    loss_log["aux_loss_total"] = aux_loss_total.detach()
                    loss_log["aux_loss_weight"] = aux_loss_weight
                set_latest_loss_dict(model, loss_log)
                self.log_iter(
                    {
                        key: float(value.detach().cpu()) if torch.is_tensor(value) else value
                        for key, value in loss_log.items()
                    },
                    10,
                )

        if hasattr(model, "compute_reg_loss"):
            reg_loss, reg_loss_dict = model.compute_reg_loss()
            if len(reg_loss_dict) > 0:
                self.log_iter({"loss": lm_loss.item(), "reg_loss": reg_loss.item(), **reg_loss_dict}, 10)
        else:
            reg_loss, reg_loss_dict = 0, {}

        lm_loss = lm_loss + reg_loss

        # if getattr(model, "should_reset_optimizer", False):
        #     self.reset_optimizer()

        if getattr(model, "should_training_stop", False):
            if hasattr(model, "save_config"):
                model.save_config(self.args.output_dir)
                self.control.should_training_stop = True

        # from modules.sdt import SDTReg
        # lm_loss = lm_loss + SDTReg()(model)

        if False:
            self.log_train_seq(input_ids, label_ids, lm_logits)

        self.train_loss_early_stop(self.control, lm_loss)

        if self.train_timestamps is not None:
            self.train_timestamps.append({
                "time": time.time(),
                "mem": torch.cuda.memory_allocated(),
            })
            # if len(self.train_timestamps) > 1:
            #     print(f"delta: {self.train_timestamps[-1] - self.train_timestamps[-2]}")

        return lm_loss
    
    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        self._disable_s6_bridge_for_eval(model)
        input_ids, label_ids, lm_logits, _, _ = self._forward(model, inputs)
        lm_loss = self.train_crit(lm_logits, label_ids)

        logits_valid = []
        label_ids_valid = []
        for i, (logits_sample, label_ids_sample) in enumerate(zip(lm_logits, label_ids)):
            valid_pos = label_ids_sample != self.train_crit.ignore_index

            logits_sample_valid = logits_sample[valid_pos]  # .argmax(-1)
            label_ids_sample_valid = label_ids_sample[valid_pos]

            logits_valid.append(logits_sample_valid)
            label_ids_valid.append(label_ids_sample_valid)

        return (lm_loss, logits_valid, label_ids_valid)
        
    def generation_step(self, generator, model, inputs):
        input_ids, label_ids = inputs["input_ids"], inputs["label_ids"]
        out_seq = generator(model, input_ids)
        output_ids = out_seq
        return (output_ids, label_ids)

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # torch.save(self.model, f"{output_dir}/model.pt")


        # try:
        #     torch.save(self.model, f"{output_dir}/model.pt")
        # except Exception as e:
        #     print(f"Failed saving model", e)
        save_mamba(self.model, output_dir)

        # try:
        #     save_mamba(self.model, output_dir)
        # except Exception as e:
        #     print(f"Failed saving peft", e)
        #     torch.save(self.model, f"{output_dir}/model.pt")

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.train_loss_early_stop.should_stop:
            self.control.should_evaluate = False

        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
    
    def load_model(self, path):
        self.model = load_mamba(path)["model"]

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        assert model is None

        self.model.load_state_dict(load_mamba(resume_from_checkpoint)["model"].state_dict())
        logger.info(f"Loading model from {resume_from_checkpoint}.")

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ):
        return data_collator

    def reset_optimizer(self):
        print("Resetting optimzer")
        self.optimizer = None
        self.lr_scheduler = None
        self.create_optimizer_and_scheduler(self.args.max_steps - self.state.global_step)

    def create_optimizer(self):
        if hasattr(self.model, "create_optimizer"):
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = self.model.create_optimizer(self.model, optimizer_cls, optimizer_kwargs)
            if self.optimizer is not None:
                return self.optimizer
            else:
                return super().create_optimizer()
        else:
            return super().create_optimizer()

    def _evaluate_default(self, eval_dataset, ignore_keys, metric_key_prefix):
        data = self.eval_dataset if eval_dataset is None else eval_dataset
        if data is not None:
            data.save_pred_file = str(Path(self.args.output_dir) / f"predictions-{self.state.global_step}.dat")
        
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if data is not None:
            data.save_pred_file = None
        
        return metrics

    def evaluate(self, eval_dataset: Dataset | Dict[str, Dataset] | None = None, ignore_keys: List[str] | None = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        if self.eval_generator is not None:
            metrics = self.evaluate_generation(self.eval_generator, metric_key_prefix=metric_key_prefix)
        elif self.get_eval_dataloader().dataset.eval_type == "log_likelihood":
            metrics = self.evaluate_log_likelihood()
        else:
            metrics = self._evaluate_default(eval_dataset, ignore_keys, metric_key_prefix)

        if self.min_eval_metric_after_epoch_early_stop is not None:
            self.min_eval_metric_after_epoch_early_stop(self.control, metrics)

        return metrics
    
    @torch.no_grad()
    def evaluate_log_likelihood(self, metric_key_prefix="eval"):
        dataloader = self.get_eval_dataloader()

        model = self.model
        model.eval()
        self._disable_s6_bridge_for_eval(model)

        input_ids_all = []
        pred_lls_all = []
        label_ids_all = []

        for step, inputs in enumerate(tqdm(dataloader, desc="Evaluate")):
            batch = self._forward(model, inputs)[:3]
            for input_ids, label_ids, lm_logits in zip(*batch):
                mask = label_ids != dataloader.dataset.ignore_index

                # assert lm_logits.ndim == 2
                label_ids = label_ids[mask]
                ll_all = F.log_softmax(lm_logits[mask], 1)

                # Obtain log-probs at the corresponding continuation token indices
                # pred_lls = ll_all[range(label_ids.shape[0]), label_ids]  # select gt tokens
                pred_lls = torch.gather(ll_all, 1, label_ids.unsqueeze(-1)).squeeze(-1)  # select gt tokens

                input_ids_all.append(input_ids.cpu())
                pred_lls_all.append(pred_lls.cpu())
                label_ids_all.append(label_ids.cpu())

        eval_pred = MambaLogLikelihoodPrediction(input_ids_all, pred_lls_all, label_ids_all)
        metrics = self.compute_metrics(eval_pred)

        if metric_key_prefix != "":
            metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics

    @torch.no_grad()
    def evaluate_generation(self, generator, use_cache=True, skip_metrics=None, metric_key_prefix="eval", pred_out_file=None):
        if skip_metrics is None:
            skip_metrics = self.skip_metrics

        if pred_out_file is not None:
            eval_pred_file = pred_out_file
        else:
            eval_pred_file = Path(self.args.output_dir) / f"predictions-{self.state.global_step}.yaml"

        if not use_cache or not eval_pred_file.is_file():
            model = self.model
            model.eval()
            self._disable_s6_bridge_for_eval(model)

            dataloader = self.get_eval_dataloader()

            input_ids_all = []
            pred_ids_all = []
            label_ids_all = []

            for step, inputs in enumerate(tqdm(dataloader, desc="Evaluate")):
                pred_ids, label_ids = self.generation_step(generator, model, inputs)
                input_ids_all += [*inputs["input_ids"]]
                pred_ids_all += [*pred_ids]
                label_ids_all += [*label_ids]

            eval_pred = MambaEvalPrediction(generator.tokenizer, input_ids_all, pred_ids_all, label_ids_all, 
                                            save_file=eval_pred_file, remove_eos=True)
            eval_pred.save(pred_out_file)
        else:
            if not skip_metrics:
                print(f"Loading prediction {eval_pred_file}")

        if not skip_metrics:
            eval_pred = MambaEvalPrediction.from_file(eval_pred_file)
            metrics = self.compute_metrics(eval_pred)

            if metric_key_prefix != "":
                metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

            return metrics
        else:
            return None
