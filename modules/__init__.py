
import os
from pathlib import Path

from mamba_ssm.modules.mamba_simple import Mamba

from .mamba_peft import MambaPeft
from .mixer_seq_simple import MambaLMHeadModelPeft

import torch
import json
from types import SimpleNamespace

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForSeq2SeqLM

from peft import get_peft_model, PeftConfig, PeftType


def get_checkpoints(path, return_dict=False, local_only=False):
    def _get_it(file):
        try:
            return int(Path(file).stem.split("-")[1])
        except ValueError:
            return 0

    if not Path(path).exists():
        checkpoints = [path]
    else:
        path = Path(path)
        checkpoints = list(path.glob("checkpoint-*"))

        if len(checkpoints) > 0:
            checkpoints = sorted(checkpoints, key=_get_it)
        else:
            checkpoints = [path]

    if local_only:
        assert all(((c / "model.pt").is_file() or (c / "peft.pt").is_file()) for c in checkpoints)

    if return_dict:
        checkpoints = {_get_it(c): str(c) for c in checkpoints}

    return checkpoints


def apply_mamba_fixes(model):
    from torch import nn
    from modules.mamba_peft_utils import ParameterProcessor
    from modules.mamba_peft import MultiLinearLayer
    
    dtype = torch.bfloat16

    for name, module in model.named_modules():
        if isinstance(module, MambaPeft):
            if not hasattr(module, "parameter_processors"):
                module.parameter_processors = nn.ModuleDict({
                    "A_log": ParameterProcessor(None, None, None, None),
                    "B": ParameterProcessor(None, None, None, None),
                    "C": ParameterProcessor(None, None, None, None),
                    "D": ParameterProcessor(None, None, None, None),
                    "dt": ParameterProcessor(None, None, None, None),
                })
            
            if "A" not in module.parameter_processors:
                module.parameter_processors["A"] = ParameterProcessor(None, None, None, None)

            if "x_after_conv" not in module.parameter_processors:
                module.parameter_processors["x_after_conv"] = ParameterProcessor(None, None, None, None)
        elif isinstance(module, MultiLinearLayer):
            if not hasattr(module, "cat_output"):
                module.cat_output = True

    return model


def load_mamba_full(pretrained, fuse_peft=False, apply_fixes=True, cls=MambaLMHeadModel, force_4bit=False, **kwargs):
    pretrained = get_checkpoints(pretrained)[-1]

    model_kwargs = kwargs

    trainable_params = 1

    if (Path(pretrained) / "model.pt").exists():
        model = torch.load(Path(pretrained) / "model.pt")

        if isinstance(model, dict):
            model = model["model"]

        dtype = next(iter(model.parameters())).dtype

        if dtype != model_kwargs.get("dtype", dtype):
            print(f'Moving model to {model_kwargs["dtype"]}')
            model = model.to(model_kwargs["dtype"])
            assert next(iter(model.parameters())).dtype == model_kwargs["dtype"]

        if hasattr(model, "get_nb_trainable_parameters"):
            trainable, all_params = model.get_nb_trainable_parameters()
            trainable_params = trainable / all_params
        else:
            trainable_params = 1

        if fuse_peft:
            if isinstance(model, PeftModelForSeq2SeqLM):
                model = model.merge_and_unload()

            if isinstance(model, MambaLMHeadModelPeft):
                try:
                    model.combine_layers()
                except AttributeError:
                    print("no method combine_layers")

        if apply_fixes:
            model = apply_mamba_fixes(model)

        tokenizer = _load_mamba_tokenizer()
    else:
        # if (pretrained / "pytorch_model.bin").exists():
        model = cls.from_pretrained(str(pretrained), **model_kwargs)
        tokenizer = _load_mamba_tokenizer()

        model.model_args = {
            "pretrained": pretrained,
            "cls": cls,
            **kwargs,
        }

    info = {
        "trainable_params": trainable_params
    }

    return {
        "model": model, 
        "tokenizer": tokenizer,
        "info": info
    }


def load_mamba_peft(path):
    path = Path(path)

    ckpt = torch.load(path / "peft.pt")
    train_state_dict = ckpt["state_dict"]
    model_args = ckpt["model_args"]
    peft_args = {**ckpt["peft_args"]}
    deep_supervision_args = peft_args.pop("deep_supervision", None)

    model_tokenizer = load_mamba_full(**model_args, apply_fixes=False)
    model = model_tokenizer["model"]
    tokenizer = model_tokenizer["tokenizer"]
    peft_cfg = peft_args.get("peft")
    from modules.sdft import freeze_for_sdft, freeze_lm_head_weight_for_sdft, inject_sdft_adapters, is_sdft_config_dict, merge_sdft_config
    from modules.pd_dft import (
        freeze_lm_head_weight_for_pd_dft,
        inject_pd_dft_adapters,
        is_pd_dft_config_dict,
        mark_only_pd_dft_as_trainable,
        merge_pd_dft_config,
    )
    from modules.snoft import (
        inject_snoft_adapters,
        is_snoft_config_dict,
        mark_only_snoft_as_trainable,
        merge_snoft_config,
        print_snoft_summary,
    )
    from modules.psca_wr import (
        inject_psca_wr_adapters,
        is_psca_wr_config_dict,
        mark_only_psca_wr_as_trainable,
        merge_psca_wr_config,
        print_psca_wr_summary,
    )

    sdft_loaded = is_sdft_config_dict(peft_cfg)
    pd_dft_loaded = is_pd_dft_config_dict(peft_cfg)
    snoft_loaded = is_snoft_config_dict(peft_cfg)
    psca_loaded = is_psca_wr_config_dict(peft_cfg)
    if sum([sdft_loaded, pd_dft_loaded, snoft_loaded, psca_loaded]) > 1:
        raise RuntimeError("SDFT, PD-DFT, SNOFT-E, and PSCA-WR cannot be loaded together.")
    if snoft_loaded:
        snoft_config = merge_snoft_config(peft_cfg)
        target_layers = inject_snoft_adapters(model, snoft_config)
        if snoft_config.freeze_backbone:
            mark_only_snoft_as_trainable(model, train_task_head=snoft_config.train_task_head)
        model.peft_args = {"peft": {"method": "snoft_e", "snoft": snoft_config.to_dict()}}
        print_snoft_summary(model, snoft_config, target_layers)
    elif sdft_loaded:
        sdft_config = merge_sdft_config(peft_cfg)
        inject_sdft_adapters(model, sdft_config)
        if sdft_config.sdft_freeze_base_model:
            freeze_for_sdft(model, train_classifier=sdft_config.sdft_train_classifier)
        freeze_lm_head_weight_for_sdft(model)
        model.peft_args = {"peft": peft_cfg}
    elif pd_dft_loaded:
        pd_dft_config = merge_pd_dft_config(peft_cfg)
        inject_pd_dft_adapters(model, pd_dft_config)
        mark_only_pd_dft_as_trainable(model, train_classifier=True)
        freeze_lm_head_weight_for_pd_dft(model)
        model.peft_args = {"peft": peft_cfg}
    elif psca_loaded:
        psca_config = merge_psca_wr_config(peft_cfg)
        target_layers = inject_psca_wr_adapters(model, psca_config)
        mark_only_psca_wr_as_trainable(model, train_classifier=True)
        method = "psca_lite" if psca_config.psca_fallback_lite else "psca_wr"
        model.peft_args = {"peft": {"method": method, **psca_config.to_dict()}}
        print_psca_wr_summary(model, psca_config, target_layers)
    else:
        model, _ = get_mamba_peft_model(model, return_peft_cfg=True, no_print=True, **peft_args)
    if deep_supervision_args is not None and not (sdft_loaded or pd_dft_loaded or snoft_loaded):
        from modules.deep_supervision import configure_deep_supervision

        model = configure_deep_supervision(model, no_print=True, **deep_supervision_args)
        if psca_loaded:
            mark_only_psca_wr_as_trainable(model, train_classifier=True)
    elif deep_supervision_args is not None and snoft_loaded:
        print("[SNOFT-E] Skipping deep supervision while loading SNOFT-E checkpoint.")
    elif deep_supervision_args is not None and sdft_loaded:
        print("[SDFT] Skipping deep supervision while loading SDFT checkpoint.")
    elif deep_supervision_args is not None and pd_dft_loaded:
        print("[PD-DFT] Skipping deep supervision while loading PD-DFT checkpoint.")
    missing_keys, unexpected_keys = model.load_state_dict(train_state_dict, strict=False)

    buffers = set(n for n, _ in model.named_buffers())

    assert len(unexpected_keys) == 0
    missing_keys = set(missing_keys)
    for n, p in model.named_parameters():
        if p.requires_grad or n in buffers:
            assert n not in missing_keys
        else:
            assert n in missing_keys or "s6_attention_bridge" in n

    return {
        "model": model, 
        "tokenizer": tokenizer,
    }


def save_mamba_peft(model, path):
    trainable_param_names = set(n for n, p in model.named_parameters() if p.requires_grad)
    buffers = set(n for n, _ in model.named_buffers())

    def _is_save_param(name, param):
        if name in trainable_param_names or name in buffers:
            return True
        
        return False

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    state_dict_peft = {n: p for n, p in state_dict.items() if _is_save_param(n, p)}
    torch.save({
        "model_args": model.model_args,
        "peft_args": model.peft_args,
        "state_dict": state_dict_peft,
    }, path / "peft.pt.temp")
    os.rename(path / "peft.pt.temp", path / "peft.pt")


def save_mamba(model, path):
    if hasattr(model, "peft_args"):
        save_mamba_peft(model, path)
    else:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model, path / "model.pt")

def load_mamba(path, **kwargs):
    if (Path(path) / "peft.pt").exists():
        return load_mamba_peft(path)
    else:
        return load_mamba_full(path, **kwargs)


def load_tokenizer(pretrained):
    tokenizer = _load_mamba_tokenizer()
    return tokenizer


def _load_mamba_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = "###"
    tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template
    return tokenizer


def print_trainable_parameter_names(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params > 0 else 0

    print("Trainable parameter summary:")
    print(f"  total params: {total_params:,}")
    print(f"  trainable params: {trainable_params:,}")
    print(f"  trainable ratio: {ratio:.6%}")
    print("Trainable parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def get_mamba_peft_model(model, peft, return_peft_cfg=False, train_embedding=False, no_print=False):
    model_args = getattr(model, "model_args", {})
    peft_args = peft

    if isinstance(peft, (str, Path)):
        with open(peft, "r") as f:
            peft = json.load(f)

    from modules.sdft import (
        freeze_for_sdft,
        freeze_lm_head_weight_for_sdft,
        inject_sdft_adapters,
        is_sdft_config_dict,
        merge_sdft_config,
        print_sdft_summary,
    )
    from modules.pd_dft import (
        inject_pd_dft_adapters,
        is_pd_dft_config_dict,
        mark_only_pd_dft_as_trainable,
        merge_pd_dft_config,
        print_pd_dft_summary,
    )
    from modules.snoft import (
        inject_snoft_adapters,
        is_snoft_config_dict,
        mark_only_snoft_as_trainable,
        merge_snoft_config,
        print_snoft_summary,
    )
    from modules.psca_wr import (
        inject_psca_wr_adapters,
        is_psca_wr_config_dict,
        mark_only_psca_wr_as_trainable,
        merge_psca_wr_config,
        print_psca_wr_summary,
    )

    if is_snoft_config_dict(peft):
        snoft_config = merge_snoft_config(peft)
        target_layers = inject_snoft_adapters(model, snoft_config)
        if snoft_config.freeze_backbone:
            mark_only_snoft_as_trainable(model, train_task_head=snoft_config.train_task_head)
        model.model_args = model_args
        model.peft_args = {"peft": {"method": "snoft_e", "snoft": snoft_config.to_dict()}}
        if not no_print:
            print_snoft_summary(model, snoft_config, target_layers)
        if return_peft_cfg:
            values = snoft_config.to_dict()
            method = values.pop("method")
            return model, SimpleNamespace(method=method, use_snoft=True, **values)
        return model

    if is_psca_wr_config_dict(peft):
        psca_config = merge_psca_wr_config(peft)
        target_layers = inject_psca_wr_adapters(model, psca_config)
        mark_only_psca_wr_as_trainable(model, train_classifier=True)
        method = "psca_lite" if psca_config.psca_fallback_lite else "psca_wr"
        model.model_args = model_args
        model.peft_args = {"peft": {"method": method, **psca_config.to_dict()}}
        if not no_print:
            print_psca_wr_summary(model, psca_config, target_layers)
        if return_peft_cfg:
            return model, SimpleNamespace(method=method, **psca_config.to_dict())
        return model

    if hasattr(model, "split_layers"):
        model.split_layers()
    else:
        print("no split_layers")

    if is_sdft_config_dict(peft):
        sdft_config = merge_sdft_config(peft)
        target_layers = inject_sdft_adapters(model, sdft_config)
        if sdft_config.sdft_freeze_base_model:
            freeze_for_sdft(model, train_classifier=sdft_config.sdft_train_classifier)
        freeze_lm_head_weight_for_sdft(model)
        model.model_args = model_args
        model.peft_args = {"peft": {"method": "sdft", **sdft_config.to_dict()}}
        if not no_print:
            print_sdft_summary(model, sdft_config, target_layers)
        if return_peft_cfg:
            return model, SimpleNamespace(method="sdft", **sdft_config.to_dict())
        return model

    if is_pd_dft_config_dict(peft):
        pd_dft_config = merge_pd_dft_config(peft)
        target_layers = inject_pd_dft_adapters(model, pd_dft_config)
        mark_only_pd_dft_as_trainable(model, train_classifier=True)
        model.model_args = model_args
        model.peft_args = {"peft": {"method": "pd_dft", **pd_dft_config.to_dict()}}
        if not no_print:
            print_pd_dft_summary(model, pd_dft_config, target_layers)
        if return_peft_cfg:
            return model, SimpleNamespace(method="pd_dft", **pd_dft_config.to_dict())
        return model

    if isinstance(peft, list):
        peft = {

            "peft_type": "MULTI_PEFT",
            "configs": peft
        }

    if isinstance(peft, dict):
        peft = PeftConfig.from_peft_type(**peft)

    model = get_peft_model(model, peft)

    if train_embedding:
        model.model.word_embeddings.weight.requires_grad = True

    if not no_print:
        print_trainable_parameter_names(model)

    model.model_args = model_args
    model.peft_args = {
        "peft": peft
    }

    if return_peft_cfg:
        return model, peft
    return model


def get_trainable_parameters_ratio(model):
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, all_params = model.get_nb_trainable_parameters()
        trainable_params = trainable / all_params
    else:
        all_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params = trainable / all_params if all_params > 0 else 0

    return trainable_params
