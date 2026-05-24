# SDFT Core Code

This note collects the minimum code path for the SDFT method in this repo:
where the adapter is defined, where it is injected into Mamba, how it is
frozen, and how training logs prove that it is changing the conv driver `v`.

## 1. Formula

For the current experiments, use:

```text
sdft_gate_mode: none

v_sdft = v + delta_v
delta_v = rho * U(dropout(SiLU(V(LN(v)))))
```

Code names:

```text
V   -> adapter.down
U   -> adapter.up
LN  -> adapter.ln
rho -> adapter.rho
```

Default hyperparameters:

```yaml
use_sdft: true
sdft_rank: 4
sdft_rho_init: 0.05
sdft_gate_mode: none
sdft_dropout: 0.05
sdft_target_layers: all
sdft_freeze_base_model: true
sdft_train_classifier: true
sdft_log_stats: true
sdft_log_per_layer: false
sdft_log_grad: true
```

## 2. Adapter Definition

File: `modules/sdft.py`

```python
class SDFTDriverAdapter(nn.Module):
    def __init__(
        self,
        d_inner: int,
        rank: int = 4,
        rho_init: float = 0.05,
        gate_mode: str = "none",
        dropout: float = 0.05,
        layer_idx: Optional[int] = None,
        stats_enabled: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_inner = int(d_inner)
        self.rank = int(rank)
        self.layer_idx = layer_idx
        self.gate_mode = str(gate_mode).lower()
        self.stats_enabled = bool(stats_enabled)
        self.eps = 1e-8

        self.ln = nn.LayerNorm(self.d_inner, device=device, dtype=dtype)
        self.down = nn.Linear(self.d_inner, self.rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(self.rank, self.d_inner, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(float(dropout))
        self.rho = nn.Parameter(torch.tensor(float(rho_init), device=device, dtype=dtype))

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-4)

        self._stats_accumulator = defaultdict(list)
```

The adapter supports `[B, D, L]`, `[B, L, D]`, and single-token `[B, D]`
layouts through `_as_bld()` and `_restore_layout()`.

Core forward:

```python
def forward(self, v: torch.Tensor, z: Optional[torch.Tensor] = None, layout: Optional[str] = None) -> torch.Tensor:
    v_bld, layout = self._as_bld(v, layout)

    hidden = self.down(self.ln(v_bld))
    hidden = self.dropout(F.silu(hidden))
    update = self.up(hidden)

    # With sdft_gate_mode == "none", gate is None.
    gate = self._compute_gate(z, layout)
    if gate is None:
        delta = self.rho.to(dtype=v_bld.dtype, device=v_bld.device) * update
    else:
        delta = (
            self.rho.to(dtype=v_bld.dtype, device=v_bld.device)
            * gate.to(dtype=v_bld.dtype, device=v_bld.device)
            * update
        )

    delta = delta.to(dtype=v_bld.dtype, device=v_bld.device)
    self._collect_forward_stats(v_bld, delta, gate)
    return self._restore_layout(v_bld + delta, layout)
```

Important: stats are collected under `torch.no_grad()`, use `detach().float()`,
and only save scalar values.

## 3. Mamba Insertion Point

File: `modules/mamba_peft.py`

SDFT is inserted after the `in_proj` split and after the x branch has gone
through `conv1d + activation`. At this point the conv driver is the tensor that
the request calls `v`.

Training/full-sequence forward:

```python
x = self.process_parameter("x_after_conv", x)
if hasattr(self, "sdft_adapter") and self.sdft_adapter is not None:
    x = self.sdft_adapter(x, z=z, layout="bdl")

x_dbl = self.x_proj(rearrange(x, "b d l -> b l d"))
dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
dt = self.dt_proj(dt)

y = self.selective_scan_fn(
    x,
    dt,
    A,
    B,
    C,
    D,
    z=z,
    delta_bias=self.dt_proj.bias.float(),
    delta_softplus=True,
    return_last_state=ssm_state is not None,
)
```

Therefore the same `x == v_sdft` affects both:

```text
1. x_proj/dt_proj/B/C generation
2. selective_scan input
```

Single-token decode/step path:

```python
x = self.process_parameter("x_after_conv", x)
if hasattr(self, "sdft_adapter") and self.sdft_adapter is not None:
    x = self.sdft_adapter(x, z=z, layout="bd")

x_db = self.x_proj(x)
dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
dt = self.dt_proj(dt)
```

SDFT is not added to hidden states, SSM output `y`, `lm_head`, or classifier
outputs.

## 4. Adapter Injection

File: `modules/sdft.py`

Each target Mamba layer receives its own adapter instance. No adapter weights
are shared between layers.

```python
def inject_sdft_adapters(model: nn.Module, config: SDFTConfig) -> List[int]:
    if not config.use_sdft:
        return []

    blocks = _find_mamba_blocks(model)
    target_layers = _resolve_target_layers(config.sdft_target_layers, len(blocks))

    for layer_idx in target_layers:
        block = blocks[layer_idx]
        ref = block.conv1d.weight if hasattr(block, "conv1d") else next(block.parameters())
        block.sdft_adapter = SDFTDriverAdapter(
            d_inner=block.d_inner,
            rank=config.sdft_rank,
            rho_init=config.sdft_rho_init,
            gate_mode=config.sdft_gate_mode,
            dropout=config.sdft_dropout,
            layer_idx=layer_idx,
            stats_enabled=config.sdft_log_stats,
            device=ref.device,
            dtype=ref.dtype,
        )

    setattr(model, "use_sdft", True)
    setattr(model, "sdft_config", config.to_dict())
    setattr(model, "sdft_target_layers", target_layers)
    return target_layers
```

`sdft_target_layers: all` resolves to every Mamba block.

## 5. No SOT Path

File: `modules/sdft.py`

SDFT config validation rejects SOT/suffix fields:

```python
_SOT_KEYS = (
    "bias_type",
    "bias_init",
    "use_sft",
    "use_tf_sot",
    "use_cf_sot",
    "C_scale_shape",
    "C_bias_shape",
)

def validate_sdft_config_dict(cfg: Optional[Dict[str, Any]]) -> None:
    if not is_sdft_config_dict(cfg):
        return
    if str((cfg or {}).get("peft_type", "")).upper() == "SUFFIX_TUNING":
        raise ValueError("SDFT must not use the SUFFIX_TUNING PEFT path.")
    forbidden = [key for key in _SOT_KEYS if cfg.get(key) not in (None, False)]
    if forbidden:
        raise ValueError(f"SDFT config must not enable SOT/suffix fields: {forbidden}")
```

`train.py` also skips deep supervision for SDFT.

## 6. Freezing Rule

File: `modules/sdft.py`

For `method=sdft`, only SDFT parameters and the small classifier head stay
trainable. Full `lm_head.weight` is explicitly frozen.

```python
def freeze_for_sdft(model: nn.Module, train_classifier: bool = True) -> None:
    for name, param in model.named_parameters():
        lowered = name.lower()
        if "sdft" in lowered:
            param.requires_grad = True
        elif train_classifier and _is_small_classifier_param(name):
            param.requires_grad = True
        else:
            param.requires_grad = False


def freeze_lm_head_weight_for_sdft(model: nn.Module) -> None:
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None) if lm_head is not None else None
    if weight is not None and weight.requires_grad:
        print("[SDFT][WARNING] lm_head.weight was trainable; freezing it for SDFT.")
        weight.requires_grad = False
```

## 7. Forward Stats

File: `modules/sdft.py`

The adapter stores scalar forward stats in `_stats_accumulator`:

```python
self._push_stat("v_rms", v_rms)
self._push_stat("v_mean", torch.mean(v_f).item())
self._push_stat("v_std", torch.std(v_f, unbiased=False).item())
self._push_stat("v_abs_mean", v_abs)
self._push_stat("delta_rms", d_rms)
self._push_stat("delta_mean", torch.mean(d_f).item())
self._push_stat("delta_std", torch.std(d_f, unbiased=False).item())
self._push_stat("delta_abs_mean", d_abs)
self._push_stat("delta_to_v_rms_ratio", d_rms / (v_rms + self.eps))
self._push_stat("delta_to_v_abs_ratio", d_abs / (v_abs + self.eps))
self._push_stat("rho", self.rho.detach().float().item())
```

The public methods are:

```python
adapter.peek_stats()   # read means without clearing
adapter.pop_stats()    # read means and clear
adapter.clear_stats()  # clear only
```

Global collection:

```python
sdft_stats = collect_sdft_stats(model, clear=True, log_per_layer=log_per_layer)
sdft_grad_stats = collect_sdft_grad_stats(model, log_per_layer=log_per_layer)
```

The global summary includes:

```text
sdft/global/mean_delta_to_v_rms_ratio
sdft/global/max_delta_to_v_rms_ratio
sdft/global/min_delta_to_v_rms_ratio
sdft/global/mean_rho
sdft/global/trainable_param_count
sdft/global/adapter_count
```

## 8. Gradient Stats

File: `modules/sdft.py`

Gradient stats read adapter parameter grads only:

```python
adapter.up.weight.grad       -> up_grad_norm
adapter.down.weight.grad     -> down_grad_norm
adapter.rho.grad             -> rho_grad_abs
adapter.gate_scale.grad      -> gate_scale_grad_abs, only if gate exists
adapter.gate_bias.grad       -> gate_bias_grad_norm, only if gate exists
```

With `sdft_gate_mode: none`, only `up`, `down`, and `rho` gradient stats are
expected.

## 9. Training Log Hook

File: `trainer/mamba_trainer.py`

`MambaTrainer.log()` merges SDFT stats into normal Trainer logs and prints one
compact console line:

```text
[SDFT] step=500 ratio_mean=0.034 ratio_max=0.081 rho_mean=0.049 grad_up_mean=1.2e-4 grad_down_mean=3.5e-5 trainable=1.23M
```

Full per-layer logs can be written to the logger/wandb/tensorboard path when:

```yaml
sdft_log_per_layer: true
```

When SDFT is disabled, no adapters exist, no stats are collected, and no SDFT
log line is printed.

## 10. Minimal Config Example

```yaml
method: sdft
use_sdft: true
sdft_rank: 4
sdft_rho_init: 0.05
sdft_gate_mode: none
sdft_dropout: 0.05
sdft_target_layers: all
sdft_freeze_base_model: true
sdft_train_classifier: true
sdft_log_stats: true
sdft_log_grad: true
sdft_log_per_layer: false
```
