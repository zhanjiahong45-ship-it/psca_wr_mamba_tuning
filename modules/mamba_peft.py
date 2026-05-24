# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
from modules.s6_attention_bridge import (
    S6AttentionBridge,
    is_s6_attention_bridge_active,
    record_s6_attention_bridge_loss,
)

from modules.selective_scan_cuda_torch import SelectiveScanCudaTorch
from modules.selective_scan_split import SelectiveScanSplit

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref, selective_scan_naiv, \
        mamba_inner_fn
except ImportError:
    pass

from modules.mamba_peft_utils import ParameterProcessor
from modules.selective_scan_torch import SelectiveScanTorch

# try:
#     from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
# except ImportError:
causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update_cuda = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MultiLinearLayer(nn.Module):
    def __init__(self, linear: nn.Linear, names_dims, cat_output=False) -> None:
        super().__init__()

        assert linear.bias is None

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.names_dims = names_dims

        weights = torch.split(linear.weight, list(names_dims.values()), dim=0)

        for i, (name, dim) in enumerate(names_dims.items()):
            l = nn.Linear(linear.in_features, dim, bias=False, device=linear.weight.device, dtype=linear.weight.dtype)

            with torch.no_grad():
                l.weight[:] = weights[i]
            setattr(self, name, l)

        self.cat_output = cat_output

    @property
    def weight(self):
        return next(getattr(self, name).weight for name in self.names_dims.keys())

    @property
    def device(self):
        return getattr(self, next(iter(self.names_dims.keys()))).weight.device

    @property
    def dtype(self):
        return getattr(self, next(iter(self.names_dims.keys()))).weight.dtype

    @property
    def bias(self):
        return None

    def to_linear(self):
        layers = [getattr(self, name) for name in self.names_dims.keys()]
        weights = [l.weight for l in layers]
        weight = torch.cat(weights, dim=0)

        l = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=layers[0].weight.device,
                      dtype=layers[0].weight.dtype)

        with torch.no_grad():
            l.weight[:] = weight

        return l

    def forward(self, x):
        outputs = [getattr(self, name)(x) for name in self.names_dims.keys()]
        if self.cat_output:
            outputs = torch.concat(outputs, -1)
        return outputs


class MambaPeft(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=False,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            backend="cuda",
            attn_implementation=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.selective_scan_fn = {
            "cuda": lambda: selective_scan_fn,
            "cuda_torch": lambda: SelectiveScanCudaTorch(),
            "ref": lambda: selective_scan_ref,
            "naiv": lambda: selective_scan_naiv,
            "torch_logcumsumexp": lambda: SelectiveScanTorch("logcumsumexp"),
            "torch_logcumsumexp_compile": lambda: torch.compile(SelectiveScanTorch("logcumsumexp")),
            "split": lambda: SelectiveScanSplit(),
            "split_compile": lambda: torch.compile(SelectiveScanSplit()),
        }[backend]()

        assert not self.use_fast_path

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.s6_attention_bridge = S6AttentionBridge(
            self.d_state,
            self.d_inner,
            device=device,
            dtype=dtype,
        )
        self.s6_attention_bridge.requires_grad_(False)

        all_dims = {
            "b": -1, "d": self.d_inner, "l": -1, "n": self.d_state
        }
        self.parameter_processors = nn.ModuleDict({
            "A_log": ParameterProcessor("A_log", shape=[self.d_inner, self.d_state], dim_names="dn",
                                        dtype=torch.float32, device=device, all_dims=all_dims, param=A),
            "A": ParameterProcessor("A", [self.d_inner, self.d_state], dim_names="dn", dtype=torch.float32,
                                    device=device, all_dims=all_dims),
            "B": ParameterProcessor("B", [-1, self.d_state, -1], dim_names="bnl", dtype=dtype, device=device,
                                    all_dims=all_dims),
            "C": ParameterProcessor("C", [-1, self.d_state, -1], dim_names="bnl", dtype=dtype, device=device,
                                    all_dims=all_dims),
            "D": ParameterProcessor("D", [self.d_inner], dim_names="d", dtype=torch.float32, device=device,
                                    all_dims=all_dims, param=self.D),
            "dt": ParameterProcessor("dt", [-1, self.d_state, -1], dim_names="bdl", dtype=dtype, device=device,
                                     all_dims=all_dims),
            # "z": ParameterProcessor("z", [-1, self.d_inner, -1], dim_names="bdl", dtype=dtype, device=device),
            "x_after_conv": ParameterProcessor("x_after_conv", [-1, self.d_inner, -1], dim_names="bdl", dtype=dtype,
                                               device=device, all_dims=all_dims),
            "x_after_ssm": ParameterProcessor("x_after_ssm", [-1, self.d_inner, -1], dim_names="bdl", dtype=dtype,
                                              device=device, all_dims=all_dims)
        })

    def split_layers(self):
        if not isinstance(self.x_proj, MultiLinearLayer):
            self.x_proj = MultiLinearLayer(self.x_proj, {
                "x_proj_dt": self.dt_rank,
                "x_proj_B": self.d_state,
                "x_proj_C": self.d_state,
            }, cat_output=False)

        if not isinstance(self.in_proj, MultiLinearLayer):
            self.in_proj = MultiLinearLayer(self.in_proj, {
                "in_proj_x": self.d_inner,
                "in_proj_z": self.d_inner,
            }, cat_output=True)

    def combine_layers(self):
        if isinstance(self.x_proj, MultiLinearLayer):
            self.x_proj = self.x_proj.to_linear()

        if isinstance(self.in_proj, MultiLinearLayer):
            self.in_proj = self.in_proj.to_linear()

    def process_parameter(self, name, param, **kwargs):
        if name in self.parameter_processors:
            return self.parameter_processors[name](param, **kwargs)
        else:
            return param

    def has_sft_delta_tuning(self):
        if "x_after_ssm" not in self.parameter_processors:
            return False
        processor = self.parameter_processors["x_after_ssm"]
        has_sft = getattr(processor, "has_sft_delta_tuning", None)
        return bool(has_sft()) if has_sft is not None else False

    def has_driver_adapter(self):
        return (
            (hasattr(self, "sdft_adapter") and self.sdft_adapter is not None)
            or (hasattr(self, "pd_dft_adapter") and self.pd_dft_adapter is not None)
        )

    def has_snoft_adapter(self):
        return bool(getattr(self, "snoft_enabled", False)) and getattr(self, "snoft", None) is not None

    def has_psca_wr_adapter(self):
        return getattr(self, "psca_wr", None) is not None

    def project_snoft_inputs(self, v_bc, v_delta):
        if isinstance(self.x_proj, MultiLinearLayer):
            dt = F.linear(v_delta, self.x_proj.x_proj_dt.weight)
            B = F.linear(v_bc, self.x_proj.x_proj_B.weight)
            C = F.linear(v_bc, self.x_proj.x_proj_C.weight)
            return dt, B, C

        W = self.x_proj.weight
        W_dt = W[: self.dt_rank]
        W_B = W[self.dt_rank : self.dt_rank + self.d_state]
        W_C = W[self.dt_rank + self.d_state :]
        dt = F.linear(v_delta, W_dt)
        B = F.linear(v_bc, W_B)
        C = F.linear(v_bc, W_C)
        return dt, B, C

    def maybe_run_snoft_sanity_check(self, v, v_bc, v_delta, dt_raw, B, C):
        if not getattr(self, "snoft_sanity_check", False) or getattr(self, "snoft_sanity_checked", False):
            return

        with torch.no_grad():
            if v_bc.shape != v.shape:
                raise RuntimeError(f"SNOFT-E v_bc shape mismatch: got {tuple(v_bc.shape)}, expected {tuple(v.shape)}")
            if v_delta.shape != v.shape:
                raise RuntimeError(f"SNOFT-E v_delta shape mismatch: got {tuple(v_delta.shape)}, expected {tuple(v.shape)}")
            if dt_raw.shape[-1] != self.dt_rank:
                raise RuntimeError(f"SNOFT-E dt_raw dim mismatch: got {dt_raw.shape[-1]}, expected {self.dt_rank}")
            if B.shape[-1] != self.d_state:
                raise RuntimeError(f"SNOFT-E B dim mismatch: got {B.shape[-1]}, expected {self.d_state}")
            if C.shape[-1] != self.d_state:
                raise RuntimeError(f"SNOFT-E C dim mismatch: got {C.shape[-1]}, expected {self.d_state}")
            if not torch.isfinite(v_bc).all():
                raise RuntimeError("SNOFT-E v_bc contains NaN or Inf.")
            if not torch.isfinite(v_delta).all():
                raise RuntimeError("SNOFT-E v_delta contains NaN or Inf.")

            max_abs_vbc = (v_bc - v).detach().float().abs().max().item()
            max_abs_vdelta = (v_delta - v).detach().float().abs().max().item()
            tau_mean = torch.sigmoid(self.snoft.tau_logit.detach().float()).mean().item()
            alpha_bc = self.snoft.alpha_bc.detach().float().item()
            alpha_delta = self.snoft.alpha_delta.detach().float().item()
            print(
                "[SNOFT-E sanity] "
                f"v={tuple(v.shape)} v_bc={tuple(v_bc.shape)} v_delta={tuple(v_delta.shape)} "
                f"dt_raw={tuple(dt_raw.shape)} B={tuple(B.shape)} C={tuple(C.shape)} "
                f"alpha_bc={alpha_bc:.6g} alpha_delta={alpha_delta:.6g} "
                f"tau_mean={tau_mean:.6f} max_abs(v_bc-v)={max_abs_vbc:.6g} "
                f"max_abs(v_delta-v)={max_abs_vdelta:.6g}"
            )

        self.snoft_sanity_checked = True

    def apply_sft_delta_tuning(self, delta):
        if "x_after_ssm" not in self.parameter_processors:
            return delta
        processor = self.parameter_processors["x_after_ssm"]
        apply_sft = getattr(processor, "apply_sft_delta_tuning", None)
        return apply_sft(delta) if apply_sft is not None else delta

    def forward(self, hidden_states, inference_params=None, attention_mask=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # batch, seqlen, dim = hidden_states.shape
        batch = hidden_states.shape[0]

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # xz = rearrange(
        #     self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        #     "d (b l) -> b d l",
        #     l=seqlen,
        # )

        xz = self.in_proj(hidden_states)

        if isinstance(xz, (tuple, list)):
            x, z = xz
            x = rearrange(x, "b l d -> b d l")
            z = rearrange(z, "b l d -> b d l")
        else:
            xz = rearrange(xz, "b l d -> b d l")
            x, z = xz.chunk(2, dim=1)

        A_log = self.A_log
        A_log = self.process_parameter("A_log", A_log)

        A = -torch.exp(A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path
            and inference_params is None
            and not self.has_sft_delta_tuning()
            and not self.has_driver_adapter()
            and not self.has_snoft_adapter()
            and not self.has_psca_wr_adapter()
            and not is_s6_attention_bridge_active(self, inference_params=inference_params)
        ):  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            causal_conv1d_fn = None
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.conv1d(x)
                # x = self.act(x[..., :seqlen])
                x = self.act(x[..., :-(self.d_conv - 1)])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            x = self.process_parameter("x_after_conv", x)
            x_param = x_scan = x
            if self.has_snoft_adapter():
                if self.has_driver_adapter():
                    raise RuntimeError("SNOFT-E cannot be mixed with SDFT or PD-DFT adapters.")
                v = rearrange(x, "b d l -> b l d")
                v_bc, v_delta = self.snoft(v)
                dt, B, C = self.project_snoft_inputs(v_bc, v_delta)
                self.maybe_run_snoft_sanity_check(v, v_bc, v_delta, dt, B, C)
            elif hasattr(self, "pd_dft_adapter") and self.pd_dft_adapter is not None:
                x_param, x_scan = self.pd_dft_adapter(x, layout="bdl")
            else:
                if hasattr(self, "sdft_adapter") and self.sdft_adapter is not None:
                    x = self.sdft_adapter(x, z=z, layout="bdl")
                    x_param = x_scan = x

                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
                x_dbl = self.x_proj(rearrange(x_param, "b d l -> b l d"))  # (bl d)

                if isinstance(x_dbl, (tuple, list)):
                    dt, B, C = x_dbl
                else:
                    dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            # dt = self.dt_proj.weight @ dt.t()
            dt = self.dt_proj(dt)
            dt = rearrange(dt, "b l d -> b d l", b=batch)
            # dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "b l dstate -> b dstate l", b=batch).contiguous()
            C = rearrange(C, "b l dstate -> b dstate l", b=batch).contiguous()
            D = self.D

            dt = self.process_parameter("dt", dt)
            A = self.process_parameter("A", A)
            B = self.process_parameter("B", B)
            C = self.process_parameter("C", C)
            D = self.process_parameter("D", D)

            if B.ndim == 4 and C.ndim == 3:
                C = repeat(C, "b n l -> b d n l", d=B.shape[1])

            if B.ndim == 3 and C.ndim == 4:
                B = repeat(B, "b n l -> b d n l", d=C.shape[1])

            if self.has_psca_wr_adapter():
                psca_source = rearrange(x_param, "b d l -> b l d")
                psca_delta_u = self.psca_wr.compute_delta_u(psca_source)
                psca_gate = self.psca_wr.compute_gate(psca_delta_u)
                x_scan = self.psca_wr.apply_projector_residual(x_scan, psca_delta_u, layout="bdl")
                if self.psca_wr.fallback_lite:
                    x_scan = self.psca_wr.apply_lite(
                        x_scan, psca_source, layout="bdl", delta_u=psca_delta_u, gate=psca_gate
                    )
                else:
                    B, C = self.psca_wr.adapt_bc(B, C, psca_source, delta_u=psca_delta_u, gate=psca_gate)

            # add peft cat here, ensure requires grad for columns
            # B D has batch, apply to x_proj instead
            # integrate split in x_proj layer

            rem_prefix_with_lora = False
            if x_scan.shape[2] > z.shape[2]:
                assert self.out_proj.__class__.__name__ == "RemoveSeqPrefixLayer" or self.out_proj.base_layer.__class__.__name__ == "RemoveSeqPrefixLayer"
                num_pad_tokens = x_scan.shape[2] - z.shape[2]

                if self.out_proj.__class__.__name__ != "RemoveSeqPrefixLayer":
                    # compability
                    rem_prefix_with_lora = True

                z = torch.nn.functional.pad(z, (num_pad_tokens, 0), value=0)

            assert self.activation in ["silu", "swish"]
            delta_softplus = True
            if self.has_sft_delta_tuning():
                # SFT: task-specific forgetting timescale calibration.
                # delta' = delta * exp(scale * r), applied after softplus to
                # preserve positivity and pretrained dynamics.
                dt = F.softplus(dt)
                dt = self.apply_sft_delta_tuning(dt)
                delta_softplus = False

            bridge_active = is_s6_attention_bridge_active(self, inference_params=inference_params)
            scan_z = None if bridge_active else z
            y = self.selective_scan_fn(
                x_scan,
                dt,
                A,
                B,
                C,
                D,
                z=scan_z,
                delta_bias=None,  # self.dt_proj.bias.float(),
                delta_softplus=delta_softplus,
                return_last_state=ssm_state is not None,
            )

            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            if bridge_active:
                runtime = self.s6_bridge_runtime
                y, bridge_stats = self.s6_attention_bridge(
                    x=x_scan,
                    B=B,
                    C=C,
                    y_s6=y,
                    attention_mask=attention_mask,
                    gamma=runtime.get("gamma", 0.0),
                    compute_soft=runtime.get("compute_soft", False),
                    compute_linear=runtime.get("compute_linear", False),
                    compute_losses=True,
                )
                record_s6_attention_bridge_loss(self, bridge_stats)
                y = y * self.act(z)
            y = self.process_parameter("x_after_ssm", y, z=z, A=A, B=B, C=C, D=D, dt=dt)

            y = rearrange(y, "b d l -> b l d")

            if rem_prefix_with_lora:
                rem_prefix = self.out_proj.base_layer
                linear = rem_prefix.base_layer
                self.out_proj.base_layer = linear
                out = self.out_proj(rem_prefix.rem_prefix(y))
                self.out_proj.base_layer = rem_prefix
            else:
                out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        # assert False, "dont use step() with peft"
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)

        if isinstance(xz, (tuple, list)):
            x, z = xz
        else:
            x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None or not isinstance(conv_state, nn.Conv1d):
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            if isinstance(self.conv1d, nn.Conv1d):
                x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
                if self.conv1d.bias is not None:
                    x = x + self.conv1d.bias
            else:
                x = self.conv1d(conv_state)
                x = x[..., x.shape[-1] // 2]

            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x = self.process_parameter("x_after_conv", x)
        x_param = x_scan = x
        if self.has_snoft_adapter():
            if self.has_driver_adapter():
                raise RuntimeError("SNOFT-E cannot be mixed with SDFT or PD-DFT adapters.")
            v = x.unsqueeze(1)
            v_bc, v_delta = self.snoft(v)
            dt, B, C = self.project_snoft_inputs(v_bc, v_delta)
            self.maybe_run_snoft_sanity_check(v, v_bc, v_delta, dt, B, C)
            dt = dt.squeeze(1)
            B = B.squeeze(1)
            C = C.squeeze(1)
        elif hasattr(self, "pd_dft_adapter") and self.pd_dft_adapter is not None:
            x_param, x_scan = self.pd_dft_adapter(x, layout="bd")
        else:
            if hasattr(self, "sdft_adapter") and self.sdft_adapter is not None:
                x = self.sdft_adapter(x, z=z, layout="bd")
                x_param = x_scan = x

            x_db = self.x_proj(x_param)  # (B dt_rank+2*d_state)

            if isinstance(x_db, (tuple, list)):
                dt, B, C = x_db
            else:
                dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        # dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        dt = self.dt_proj(dt)

        A_log = self.A_log
        A_log = self.process_parameter("A_log", A_log)

        A = -torch.exp(A_log.float())  # (d_inner, d_state)
        D = self.D

        dt = self.process_parameter("dt", dt)
        A = self.process_parameter("A", A)
        B = self.process_parameter("B", B)
        C = self.process_parameter("C", C)
        D = self.process_parameter("D", D)

        if self.has_psca_wr_adapter():
            psca_source = x_param.unsqueeze(1)
            psca_delta_u = self.psca_wr.compute_delta_u(psca_source)
            psca_gate = self.psca_wr.compute_gate(psca_delta_u)
            x_scan = self.psca_wr.apply_projector_residual(x_scan, psca_delta_u, layout="bd")
            if self.psca_wr.fallback_lite:
                x_scan = self.psca_wr.apply_lite(
                    x_scan, psca_source, layout="bd", delta_u=psca_delta_u, gate=psca_gate
                )
            else:
                B, C = self.psca_wr.adapt_bc(B, C, psca_source, delta_u=psca_delta_u, gate=psca_gate)

        has_sft_delta_tuning = self.has_sft_delta_tuning()
        if B.ndim == 3 or C.ndim == 3 or selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt)  # + self.dt_proj.bias.to(dtype=dt.dtype)
            if has_sft_delta_tuning:
                # SFT: task-specific forgetting timescale calibration.
                # delta' = delta * exp(scale * r), applied after softplus to
                # preserve positivity and pretrained dynamics.
                dt = self.apply_sft_delta_tuning(dt)
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn" if B.ndim == 2 else "bd,bdn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x_scan, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd" if C.ndim == 2 else "bdn,bdn->bd", ssm_state.to(dtype), C)
            y = y + D.to(dtype) * x_scan
            y = y * self.act(z)  # (B D)
        else:
            dt_softplus = True
            if has_sft_delta_tuning:
                # SFT: task-specific forgetting timescale calibration.
                # delta' = delta * exp(scale * r), applied after softplus to
                # preserve positivity and pretrained dynamics.
                dt = F.softplus(dt)
                dt = self.apply_sft_delta_tuning(dt)
                dt_softplus = False
            y = selective_state_update(
                ssm_state, x_scan, dt, A, B, C, D, z=z, dt_bias=None, dt_softplus=dt_softplus
            )

        y = self.process_parameter("x_after_ssm", y, z=z, A=A, B=B, C=C, D=D, dt=dt)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None

        if hasattr(self.x_proj, "x_proj_B") and hasattr(self.x_proj.x_proj_B, "out_features"):
            d_state = self.x_proj.x_proj_B.out_features
        else:
            d_state = self.d_state

        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, **kwargs,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim, **kwargs)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, attention_mask=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, attention_mask=attention_mask)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
