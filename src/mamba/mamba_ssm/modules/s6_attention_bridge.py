# Copyright (c) 2026

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _module_device_dtype(module: nn.Module):
    for param in module.parameters(recurse=False):
        return param.device, param.dtype
    for param in module.parameters():
        return param.device, param.dtype
    return torch.device("cpu"), torch.float32


def _iter_s6_bridge_hosts(model: nn.Module) -> Iterable[nn.Module]:
    for module in model.modules():
        if hasattr(module, "s6_attention_bridge"):
            yield module


class S6AttentionBridge(nn.Module):
    """All-layer state-compatible attention bridge for Mamba/S6 blocks."""

    def __init__(
        self,
        d_state: int,
        d_inner: int,
        eps: float = 1e-6,
        rms_eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_state = int(d_state)
        self.d_inner = int(d_inner)
        self.eps = float(eps)
        self.rms_eps = float(rms_eps)
        self.phi_proj = nn.Linear(
            self.d_state,
            self.d_state,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.tau_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), device=device, dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.phi_proj.weight.zero_()
            diag = min(self.phi_proj.weight.shape)
            self.phi_proj.weight[:diag, :diag].fill_(1.0)
            self.phi_proj.bias.zero_()
            self.tau_raw.fill_(_inverse_softplus(1.0))

    def tau(self) -> torch.Tensor:
        return F.softplus(self.tau_raw).clamp_min(self.eps)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.rms_eps)

    @staticmethod
    def _bdl_to_bld(tensor: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3D [B,D,L] or [B,L,D] tensor, got {tuple(tensor.shape)}.")
        if tensor.shape[0] != batch:
            raise ValueError(f"Expected batch={batch}, got {tuple(tensor.shape)}.")
        if tensor.shape[2] == seqlen:
            return tensor.transpose(1, 2)
        if tensor.shape[1] == seqlen:
            return tensor
        raise ValueError(f"Cannot infer sequence dimension for tensor shape {tuple(tensor.shape)}.")

    def _infer_content_shape(self, tensor: torch.Tensor):
        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3D [B,D,L] or [B,L,D] tensor, got {tuple(tensor.shape)}.")
        if tensor.shape[1] == self.d_inner:
            return tensor.shape[0], tensor.shape[2]
        if tensor.shape[2] == self.d_inner:
            return tensor.shape[0], tensor.shape[1]
        raise ValueError(f"Cannot infer content branch layout from shape {tuple(tensor.shape)}.")

    def _state_to_bln(self, tensor: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
        if tensor.ndim == 4:
            # Some PEFT adapters expand B/C to [B, groups/channels, N, L].
            tensor = tensor.float().mean(dim=1)
        if tensor.ndim == 3:
            if tensor.shape[0] != batch:
                raise ValueError(f"Expected B/C batch={batch}, got {tuple(tensor.shape)}.")
            if tensor.shape[1] == self.d_state:
                tensor = tensor.transpose(1, 2)
            elif tensor.shape[2] == self.d_state:
                pass
            else:
                raise ValueError(f"Cannot infer B/C state dimension from shape {tuple(tensor.shape)}.")
        elif tensor.ndim == 2:
            if tensor.shape[0] == self.d_state:
                tensor = tensor.transpose(0, 1)
            elif tensor.shape[-1] != self.d_state:
                raise ValueError(f"Cannot infer B/C state dimension from shape {tuple(tensor.shape)}.")
            tensor = tensor.unsqueeze(0).expand(batch, -1, -1)
        else:
            raise ValueError(f"Expected B/C with 2-4 dims, got {tuple(tensor.shape)}.")

        if tensor.shape[1] != seqlen:
            if tensor.shape[1] > seqlen:
                tensor = tensor[:, -seqlen:, :]
            else:
                pad_len = seqlen - tensor.shape[1]
                tensor = F.pad(tensor, (0, 0, pad_len, 0), value=0)
        return tensor.float()

    def _mask_to_bl(self, attention_mask: Optional[torch.Tensor], batch: int, seqlen: int) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        mask = attention_mask
        if mask.ndim > 2:
            mask = mask.reshape(mask.shape[0], -1)
        if mask.shape[0] != batch:
            raise ValueError(f"Expected attention_mask batch={batch}, got {tuple(mask.shape)}.")
        mask = mask.to(dtype=torch.bool)
        if mask.shape[1] > seqlen:
            mask = mask[:, -seqlen:]
        elif mask.shape[1] < seqlen:
            pad_len = seqlen - mask.shape[1]
            prefix_valid = torch.ones(batch, pad_len, device=mask.device, dtype=torch.bool)
            mask = torch.cat([prefix_valid, mask], dim=1)
        return mask

    def _phi(self, tensor: torch.Tensor) -> torch.Tensor:
        projected = self.phi_proj(tensor.to(dtype=self.phi_proj.weight.dtype))
        return F.softplus(projected.float()) + self.eps

    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return tensor.mean()
        mask_f = mask.to(device=tensor.device, dtype=tensor.dtype)
        return (tensor * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    def _cos_loss_and_mean(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        pred_norm = F.layer_norm(prediction.float(), (prediction.shape[-1],))
        target_norm = F.layer_norm(target.float(), (target.shape[-1],))
        cosine = F.cosine_similarity(pred_norm, target_norm, dim=-1)
        cosine_mean = self._masked_mean(cosine, mask)
        return 1.0 - cosine_mean, cosine_mean.detach()

    def forward(
        self,
        x,
        B,
        C,
        y_s6,
        attention_mask=None,
        gamma: float = 0.0,
        compute_soft: bool = True,
        compute_linear: bool = True,
        compute_losses: bool = True,
    ):
        batch, seqlen = self._infer_content_shape(x)
        value = self._bdl_to_bld(x, batch, seqlen).float()
        y_s6_bld = self._bdl_to_bld(y_s6, batch, seqlen).float()
        q = self._rms_norm(self._state_to_bln(C, batch, seqlen))
        k = self._rms_norm(self._state_to_bln(B, batch, seqlen))
        valid_mask = self._mask_to_bl(attention_mask, batch, seqlen)
        tau = self.tau().to(device=value.device)

        y_soft = None
        if compute_soft:
            scores = torch.einsum("bln,bsn->bls", q, k)
            scores = scores / (tau * math.sqrt(float(self.d_state)))
            causal_mask = torch.ones(seqlen, seqlen, device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            if valid_mask is not None:
                key_mask = ~valid_mask.to(device=scores.device)
                scores = scores.masked_fill(key_mask[:, None, :], torch.finfo(scores.dtype).min)
            y_soft = torch.softmax(scores, dim=-1).to(dtype=value.dtype) @ value

        y_lin = None
        if compute_linear:
            phi_q = self._phi(q)
            phi_k = self._phi(k)
            linear_value = value
            if valid_mask is not None:
                key_valid = valid_mask.to(device=value.device, dtype=value.dtype).unsqueeze(-1)
                phi_k = phi_k * key_valid
                linear_value = value * key_valid
            kv = torch.einsum("bln,bld->blnd", phi_k, linear_value)
            S = torch.cumsum(kv, dim=1)
            z = torch.cumsum(phi_k, dim=1)
            num = torch.einsum("bln,blnd->bld", phi_q, S)
            den = torch.einsum("bln,bln->bl", phi_q, z).unsqueeze(-1) + self.eps
            y_lin = num / den

        gamma = float(gamma or 0.0)
        if gamma > 0.0 and y_soft is not None:
            y_train = y_s6_bld + gamma * y_soft.to(dtype=y_s6_bld.dtype)
        else:
            y_train = y_s6_bld

        stats: Dict[str, torch.Tensor] = {
            "tau": tau.detach(),
            "gamma": torch.tensor(gamma, device=value.device),
        }
        if compute_losses and y_lin is not None:
            loss_handoff, cos_s6_lin = self._cos_loss_and_mean(y_s6_bld, y_lin.detach(), valid_mask)
            stats["loss_handoff"] = loss_handoff
            stats["cos_s6_lin"] = cos_s6_lin
            if y_soft is not None:
                loss_lin, cos_soft_lin = self._cos_loss_and_mean(y_lin, y_soft.detach(), valid_mask)
                stats["loss_lin"] = loss_lin
                stats["cos_soft_lin"] = cos_soft_lin

        return y_train.transpose(1, 2).to(dtype=y_s6.dtype), stats


def make_s6_attention_bridge(module: nn.Module) -> S6AttentionBridge:
    device, dtype = _module_device_dtype(module)
    bridge_dtype = dtype if getattr(dtype, "is_floating_point", False) else torch.float32
    bridge = S6AttentionBridge(
        d_state=int(module.d_state),
        d_inner=int(module.d_inner),
        device=device,
        dtype=bridge_dtype,
    )
    bridge.requires_grad_(False)
    return bridge


def ensure_s6_attention_bridge(module: nn.Module) -> S6AttentionBridge:
    bridge = getattr(module, "s6_attention_bridge", None)
    if bridge is None:
        bridge = make_s6_attention_bridge(module)
        module.s6_attention_bridge = bridge
    if not hasattr(module, "s6_bridge_runtime"):
        module.s6_bridge_runtime = {
            "enabled": False,
            "compute_soft": False,
            "compute_linear": False,
            "gamma": 0.0,
            "lambda_lin": 0.0,
            "lambda_handoff": 0.0,
            "progress": 0.0,
            "in_final_no_attention": False,
        }
    module.s6_bridge_last = None
    return bridge


def configure_s6_attention_bridge(
    model: nn.Module,
    enabled: bool = True,
    bridge_all_layers: bool = True,
    lambda_lin: float = 0.01,
    lambda_handoff: float = 0.03,
    bridge_use_soft_mixing: bool = True,
    bridge_mix_ratio_init: float = 1.0,
    bridge_mix_decay_portion: float = 0.3,
    final_no_attention_portion: float = 0.1,
    bridge_log_interval: int = 10,
    no_print: bool = False,
) -> nn.Module:
    if enabled and not bridge_all_layers:
        raise ValueError("Version A is an all-layer bridge: bridge_all_layers must be true when bridge_enabled=true.")

    count = 0
    for module in model.modules():
        if hasattr(module, "d_state") and hasattr(module, "d_inner") and hasattr(module, "selective_scan_fn"):
            bridge = ensure_s6_attention_bridge(module)
            bridge.requires_grad_(bool(enabled))
            module.s6_bridge_enabled = bool(enabled)
            module.s6_bridge_all_layers = bool(bridge_all_layers)
            module.s6_bridge_lambda_lin = float(lambda_lin)
            module.s6_bridge_lambda_handoff = float(lambda_handoff)
            module.s6_bridge_use_soft_mixing = bool(bridge_use_soft_mixing)
            module.s6_bridge_mix_ratio_init = float(bridge_mix_ratio_init)
            module.s6_bridge_mix_decay_portion = float(bridge_mix_decay_portion)
            module.s6_bridge_final_no_attention_portion = float(final_no_attention_portion)
            module.s6_bridge_log_interval = int(bridge_log_interval or 10)
            count += 1

    model.s6_attention_bridge_config = {
        "enabled": bool(enabled),
        "bridge_all_layers": bool(bridge_all_layers),
        "lambda_lin": float(lambda_lin),
        "lambda_handoff": float(lambda_handoff),
        "bridge_use_soft_mixing": bool(bridge_use_soft_mixing),
        "bridge_mix_ratio_init": float(bridge_mix_ratio_init),
        "bridge_mix_decay_portion": float(bridge_mix_decay_portion),
        "final_no_attention_portion": float(final_no_attention_portion),
        "bridge_log_interval": int(bridge_log_interval or 10),
        "num_layers": count,
    }
    if not no_print:
        status = "enabled" if enabled else "disabled"
        print(f"[S6 bridge] {status}; all_layers={bridge_all_layers}; layers={count}; eval path stays original S6.")
    return model


def set_s6_attention_bridge_runtime(model: nn.Module, progress: float = 0.0) -> None:
    progress = min(1.0, max(0.0, float(progress)))
    for module in _iter_s6_bridge_hosts(model):
        enabled = bool(getattr(module, "s6_bridge_enabled", False)) and bool(module.training)
        final_portion = min(1.0, max(0.0, _as_float(getattr(module, "s6_bridge_final_no_attention_portion", 0.1), 0.1)))
        final_start = 1.0 - final_portion
        in_final = enabled and final_portion > 0.0 and progress >= final_start

        mix_decay = min(1.0, max(0.0, _as_float(getattr(module, "s6_bridge_mix_decay_portion", 0.3), 0.3)))
        mix_init = _as_float(getattr(module, "s6_bridge_mix_ratio_init", 1.0), 1.0)
        use_mixing = bool(getattr(module, "s6_bridge_use_soft_mixing", True))
        gamma = 0.0
        if enabled and use_mixing and not in_final and mix_decay > 0.0 and progress < mix_decay:
            gamma = mix_init * max(0.0, 1.0 - progress / mix_decay)

        lambda_lin = _as_float(getattr(module, "s6_bridge_lambda_lin", 0.01), 0.01) if enabled and not in_final else 0.0
        lambda_handoff = _as_float(getattr(module, "s6_bridge_lambda_handoff", 0.03), 0.03) if enabled else 0.0
        if in_final and final_portion > 0.0:
            lambda_handoff *= max(0.0, (1.0 - progress) / final_portion)

        compute_soft = enabled and not in_final and (gamma > 0.0 or lambda_lin > 0.0)
        compute_linear = enabled and (compute_soft or lambda_handoff > 0.0)
        module.s6_bridge_runtime = {
            "enabled": enabled,
            "compute_soft": compute_soft,
            "compute_linear": compute_linear,
            "gamma": float(gamma),
            "lambda_lin": float(lambda_lin),
            "lambda_handoff": float(lambda_handoff),
            "progress": float(progress),
            "in_final_no_attention": bool(in_final),
        }


def reset_s6_attention_bridge_losses(model: nn.Module) -> None:
    for module in _iter_s6_bridge_hosts(model):
        module.s6_bridge_last = None


def is_s6_attention_bridge_active(module: nn.Module, inference_params=None) -> bool:
    if inference_params is not None or not bool(module.training):
        return False
    if not bool(getattr(module, "s6_bridge_enabled", False)):
        return False
    runtime = getattr(module, "s6_bridge_runtime", None) or {}
    return bool(runtime.get("compute_soft", False) or runtime.get("compute_linear", False) or runtime.get("gamma", 0.0) > 0.0)


def record_s6_attention_bridge_loss(module: nn.Module, stats: Dict[str, torch.Tensor]) -> None:
    module.s6_bridge_last = stats


def s6_attention_bridge_is_enabled(model: nn.Module) -> bool:
    return any(bool(getattr(module, "s6_bridge_enabled", False)) for module in _iter_s6_bridge_hosts(model))


def get_s6_attention_bridge_log_interval(model: nn.Module, default: int = 10) -> int:
    for module in _iter_s6_bridge_hosts(model):
        return int(getattr(module, "s6_bridge_log_interval", default) or default)
    return int(default)


def compute_s6_attention_bridge_loss(model: nn.Module):
    loss_lin = []
    loss_handoff = []
    cos_soft_lin = []
    cos_s6_lin = []
    tau_values = []
    gammas = []
    lambda_lin_values = []
    lambda_handoff_values = []
    per_layer_tau = {}

    for index, module in enumerate(_iter_s6_bridge_hosts(model)):
        stats = getattr(module, "s6_bridge_last", None)
        runtime = getattr(module, "s6_bridge_runtime", {}) or {}
        if stats is None:
            continue
        layer_idx = getattr(module, "layer_idx", index)
        tau = stats.get("tau")
        if torch.is_tensor(tau):
            tau_detached = tau.detach()
            tau_values.append(tau_detached)
            per_layer_tau[f"bridge_tau/layer_{layer_idx}"] = tau_detached
        if "loss_lin" in stats:
            loss_lin.append(stats["loss_lin"])
        if "loss_handoff" in stats:
            loss_handoff.append(stats["loss_handoff"])
        if "cos_soft_lin" in stats:
            cos_soft_lin.append(stats["cos_soft_lin"])
        if "cos_s6_lin" in stats:
            cos_s6_lin.append(stats["cos_s6_lin"])
        gammas.append(float(runtime.get("gamma", 0.0)))
        lambda_lin_values.append(float(runtime.get("lambda_lin", 0.0)))
        lambda_handoff_values.append(float(runtime.get("lambda_handoff", 0.0)))

    if not loss_lin and not loss_handoff:
        return None, {}

    device = None
    if loss_handoff:
        device = loss_handoff[0].device
    elif loss_lin:
        device = loss_lin[0].device

    zero = torch.zeros((), device=device)
    loss_lin_mean = torch.stack(loss_lin).mean() if loss_lin else zero
    loss_handoff_mean = torch.stack(loss_handoff).mean() if loss_handoff else zero
    lambda_lin_mean = sum(lambda_lin_values) / max(1, len(lambda_lin_values))
    lambda_handoff_mean = sum(lambda_handoff_values) / max(1, len(lambda_handoff_values))
    total = lambda_lin_mean * loss_lin_mean + lambda_handoff_mean * loss_handoff_mean

    log = {
        "bridge_loss_lin": loss_lin_mean.detach(),
        "bridge_loss_handoff": loss_handoff_mean.detach(),
        "bridge_lambda_lin": lambda_lin_mean,
        "bridge_lambda_handoff": lambda_handoff_mean,
        "bridge_aux_loss_total": total.detach(),
        "bridge_gamma": sum(gammas) / max(1, len(gammas)),
        "bridge_layer_count": len(gammas),
    }
    if cos_soft_lin:
        log["bridge_cos_soft_lin"] = torch.stack(cos_soft_lin).mean().detach()
    if cos_s6_lin:
        log["bridge_cos_s6_lin"] = torch.stack(cos_s6_lin).mean().detach()
    if tau_values:
        log["bridge_tau_mean"] = torch.stack([tau.reshape(()) for tau in tau_values]).mean().detach()
        log.update(per_layer_tau)
    return total, log
