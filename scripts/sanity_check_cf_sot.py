import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn


def _install_suffix_tuning_import_stubs(repo_root):
    peft_mod = types.ModuleType("peft")
    peft_config_mod = types.ModuleType("peft.config")
    peft_tuners_mod = types.ModuleType("peft.tuners")
    peft_tuners_utils_mod = types.ModuleType("peft.tuners.tuners_utils")

    class PeftConfig:
        pass

    class BaseTuner(nn.Module):
        pass

    class BaseTunerLayer:
        def __init__(self, *args, **kwargs):
            self._active_adapter = None

        @property
        def active_adapter(self):
            return self._active_adapter

        @property
        def active_adapters(self):
            if self._active_adapter is None:
                return []
            if isinstance(self._active_adapter, (list, tuple)):
                return list(self._active_adapter)
            return [self._active_adapter]

        def set_adapter(self, adapter_names):
            if isinstance(adapter_names, (list, tuple)):
                if len(adapter_names) == 0:
                    self._active_adapter = None
                elif len(adapter_names) == 1:
                    self._active_adapter = adapter_names[0]
                else:
                    self._active_adapter = list(adapter_names)
            else:
                self._active_adapter = adapter_names

    peft_config_mod.PeftConfig = PeftConfig
    peft_tuners_utils_mod.BaseTuner = BaseTuner
    peft_tuners_utils_mod.BaseTunerLayer = BaseTunerLayer
    peft_tuners_utils_mod.check_target_module_exists = lambda *args, **kwargs: True

    sys.modules["peft"] = peft_mod
    sys.modules["peft.config"] = peft_config_mod
    sys.modules["peft.tuners"] = peft_tuners_mod
    sys.modules["peft.tuners.tuners_utils"] = peft_tuners_utils_mod

    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = [str(repo_root / "modules")]
    sys.modules["modules"] = modules_pkg

    mamba_peft_utils_mod = types.ModuleType("modules.mamba_peft_utils")

    class MambaPeftType(str):
        SUFFIX_TUNING = "SUFFIX_TUNING"

    def _identity_register(_name):
        def _wrap(cls):
            return cls

        return _wrap

    mamba_peft_utils_mod.MambaPeftType = MambaPeftType
    mamba_peft_utils_mod.register_peft_config = _identity_register
    mamba_peft_utils_mod.register_peft_tuner = _identity_register
    sys.modules["modules.mamba_peft_utils"] = mamba_peft_utils_mod

    mamba_tuner_utils_mod = types.ModuleType("modules.mamba_tuner_utils")
    mamba_tuner_utils_mod.MambaBaseTuner = BaseTuner
    sys.modules["modules.mamba_tuner_utils"] = mamba_tuner_utils_mod


def _load_suffix_tuning(repo_root):
    _install_suffix_tuning_import_stubs(repo_root)
    suffix_path = repo_root / "modules" / "suffix_tuning.py"
    spec = importlib.util.spec_from_file_location("modules.suffix_tuning", suffix_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modules.suffix_tuning"] = module
    spec.loader.exec_module(module)
    return module


class DummyBaseLayer(nn.Module):
    def __init__(self, d_inner, d_state):
        super().__init__()
        self.all_dims = {"b": -1, "d": d_inner, "l": -1, "n": d_state}
        self.dtype = torch.float32
        self.device = torch.device("cpu")


def _make_processor(module, use_cf_sot=False, use_tf_sot=False):
    return module.SuffixTuningBiasProcessor(
        DummyBaseLayer(d_inner=5, d_state=3),
        "default",
        bias_type=module.SuffixTuningBiasType.SILU_Z_C,
        bias_init=module.SuffixTuningBiasInit.ZERO,
        use_tf_sot=use_tf_sot,
        tf_sot_num_freqs=4,
        tf_sot_num_basis=8,
        tf_sot_eps=0.1,
        tf_sot_max_seq_len=256,
        tf_sot_freq_grid="geometric",
        tf_sot_normalize_basis=True,
        use_cf_sot=use_cf_sot,
        cf_sot_num_freqs=4,
        cf_sot_num_basis=8,
        cf_sot_eps=0.1,
        cf_sot_max_seq_len=256,
        cf_sot_freq_grid="geometric",
        cf_sot_normalize_basis=True,
        cf_sot_context_center=True,
        cf_sot_context_reduce="mean",
        cf_sot_init_tf_coeff="zero",
        cf_sot_init_context_alpha="zero",
    )


def run_check():
    torch.manual_seed(7)
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_suffix_tuning(repo_root)

    batch_size, seq_len, d_inner, d_state = 2, 9, 5, 3
    x = torch.randn(batch_size, d_inner, seq_len)
    z = torch.randn(batch_size, d_inner, seq_len)
    c = torch.randn(batch_size, d_state, seq_len)
    dummy = torch.empty(0)

    proc_sot = _make_processor(module, use_cf_sot=False)
    proc_disabled = _make_processor(module, use_cf_sot=False)
    proc_cf = _make_processor(module, use_cf_sot=True)

    h_prime = torch.randn(d_inner, d_state)
    with torch.no_grad():
        proc_sot.suffixtuning_bias["default"].copy_(h_prime)
        proc_disabled.suffixtuning_bias["default"].copy_(h_prime)
        proc_cf.suffixtuning_bias["default"].copy_(h_prime)

    y_sot = proc_sot(x, z, dummy, dummy, c, dummy, dummy)
    y_disabled = proc_disabled(x, z, dummy, dummy, c, dummy, dummy)
    y_cf_init = proc_cf(x, z, dummy, dummy, c, dummy, dummy)
    scale = proc_cf.compute_cf_sot_scale(torch.nn.functional.silu(z), "default")

    coeff_param_names = [
        name for name, param in proc_cf.named_parameters()
        if param.requires_grad and "cf_sot_tf_coeff" in name
    ]
    alpha_param_names = [
        name for name, param in proc_cf.named_parameters()
        if param.requires_grad and "cf_sot_context_alpha" in name
    ]
    optimizer = torch.optim.SGD(proc_cf.parameters(), lr=0.1)
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    coeff = proc_cf.cf_sot_tf_coeff["default"]
    alpha = proc_cf.cf_sot_context_alpha["default"]

    optimizer.zero_grad()
    loss = y_cf_init.square().mean()
    loss.backward()
    coeff_grad = coeff.grad
    alpha_grad = alpha.grad

    debug = proc_cf.last_cf_sot_debug["default"]

    mutual_exclusion_raised = False
    try:
        _make_processor(module, use_cf_sot=True, use_tf_sot=True)
    except ValueError:
        mutual_exclusion_raised = True

    finite_tensors = [y_sot, y_disabled, y_cf_init, scale, coeff_grad, alpha_grad]

    return {
        "use_cf_sot_false_matches_sot": torch.allclose(y_disabled, y_sot, atol=0.0, rtol=0.0),
        "cf_sot_zero_params_matches_sot": torch.allclose(y_cf_init, y_sot, atol=0.0, rtol=0.0),
        "max_abs_diff_use_cf_false": float((y_disabled - y_sot).abs().max().item()),
        "max_abs_diff_cf_zero_params": float((y_cf_init - y_sot).abs().max().item()),
        "tf_cf_mutual_exclusion_raised": mutual_exclusion_raised,
        "cf_sot_tf_coeff_trainable_names": coeff_param_names,
        "cf_sot_context_alpha_trainable_names": alpha_param_names,
        "cf_sot_tf_coeff_in_optimizer": id(coeff) in optimizer_param_ids,
        "cf_sot_context_alpha_in_optimizer": id(alpha) in optimizer_param_ids,
        "gate_shape": list(debug["gate_shape"]),
        "h_prime_shape": list(debug["h_prime_shape"]),
        "phi_shape": list(debug["phi_shape"]),
        "tf_coeff_shape": list(debug["tf_coeff_shape"]),
        "context_alpha_shape": list(debug["context_alpha_shape"]),
        "freq_raw_shape": list(debug["freq_raw_shape"]),
        "context_scalar_shape": list(debug["context_scalar_shape"]),
        "context_raw_shape": list(debug["context_raw_shape"]),
        "traj_raw_shape": list(debug["traj_raw_shape"]),
        "scale_shape": list(debug["scale_shape"]),
        "offset_shape": list(debug["offset_shape"]),
        "scale_all_ones_at_init": torch.allclose(scale, torch.ones_like(scale), atol=0.0, rtol=0.0),
        "initial_scale_min": float(scale.min().item()),
        "initial_scale_max": float(scale.max().item()),
        "initial_mean_abs_scale_minus_1": float((scale - 1.0).abs().mean().item()),
        "cf_sot_tf_coeff_grad_is_not_none": coeff_grad is not None,
        "cf_sot_tf_coeff_grad_norm": float(coeff_grad.norm().item()),
        "cf_sot_tf_coeff_grad_nonzero": bool(coeff_grad.norm().item() > 0.0),
        "cf_sot_context_alpha_grad_is_not_none": alpha_grad is not None,
        "cf_sot_context_alpha_grad_norm": float(alpha_grad.norm().item()),
        "cf_sot_context_alpha_grad_nonzero": bool(alpha_grad.norm().item() > 0.0),
        "all_checked_tensors_finite": all(torch.isfinite(t).all().item() for t in finite_tensors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = run_check()
    text = json.dumps(result, indent=2)
    print(text)

    if args.output is not None:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")

    failed = [
        key for key in (
            "use_cf_sot_false_matches_sot",
            "cf_sot_zero_params_matches_sot",
            "tf_cf_mutual_exclusion_raised",
            "cf_sot_tf_coeff_in_optimizer",
            "cf_sot_context_alpha_in_optimizer",
            "scale_all_ones_at_init",
            "cf_sot_tf_coeff_grad_is_not_none",
            "cf_sot_tf_coeff_grad_nonzero",
            "cf_sot_context_alpha_grad_is_not_none",
            "cf_sot_context_alpha_grad_nonzero",
            "all_checked_tensors_finite",
        )
        if not result[key]
    ]
    if failed:
        raise SystemExit(f"CF-SOT sanity check failed: {failed}")


if __name__ == "__main__":
    main()
