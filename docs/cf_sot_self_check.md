# CF-SOT Self-Check Report

## Scope

Implemented CF-SOT as a clean branch on the existing SOT path for Mamba-130M GLUE MRPC/RTE/CoLA. Original SOT remains the default path. TF-SOT remains available. `use_tf_sot` and `use_cf_sot` are mutually exclusive and raise `ValueError` if both are enabled.

## Modified Files

- `modules/suffix_tuning.py`: CF-SOT config fields, Fourier basis builder, scale computation, parameter registration, trainable parameter marking, and forward integration on the original `SILU_Z_C` SOT path.
- `train.py`: CF-SOT method naming and result metadata.
- `cfg/final/peft/mamba-130m/glue_mrpc/cf_sot.json`: MRPC CF-SOT PEFT config.
- `cfg/final/peft/mamba-130m/glue_rte/cf_sot.json`: RTE CF-SOT PEFT config.
- `cfg/final/peft/mamba-130m/glue_cola/cf_sot.json`: CoLA CF-SOT PEFT config.
- `cfg/final/exps/mamba-130m/glue_mrpc/cf_sot.yaml`: MRPC CF-SOT experiment config.
- `cfg/final/exps/mamba-130m/glue_rte/cf_sot.yaml`: RTE CF-SOT experiment config.
- `cfg/final/exps/mamba-130m/glue_cola/cf_sot.yaml`: CoLA CF-SOT experiment config.
- `scripts/run_cf_sot_glue.sh`: MRPC/RTE/CoLA run script.
- `scripts/sanity_check_cf_sot.py`: local implementation sanity checks for init equivalence, params, gradients, shapes, and scale bounds.
- `tools/inspect_cf_sot.py`: checkpoint inspection for CF-SOT coefficients, context alpha, and scale statistics.

## Parameter Registration

- `cf_sot_tf_coeff` is registered in `SuffixTuningBiasProcessor.update_layer` as a per-adapter `nn.ParameterDict` entry with zeros of shape `[cf_sot_num_basis]`.
- `cf_sot_context_alpha` is registered in `SuffixTuningBiasProcessor.update_layer` as a per-adapter `nn.ParameterDict` entry with zeros of shape `[1]`.
- Both names are included in `_mark_only_adapters_as_trainable`, so they are trainable and should enter the optimizer.

## Forward Path

- Original SOT offset location: `SuffixTuningBiasProcessor.forward`, `SILU_Z_C` branch.
- Existing SOT computation remains:

```python
gate = F.silu(z_readout)
y_add = torch.einsum("bdl,bnl,dn -> bdl", gate, C_readout, param)
```

- CF-SOT applies a bounded scalar time scale to that original offset:

```python
scale = self.compute_cf_sot_scale(gate=gate, active_adapter=active_adapter, attention_mask=None)
y_add = y_add * scale
```

- This keeps the original gate, injection point, and SOT offset parameter unchanged. Runtime tensor layout in this repo is `[B, D, L]`, so `scale` is `[B, 1, L]`.

## Formula Code

```python
freq_raw = phi @ tf_coeff.to(device=gate_seq.device, dtype=gate_seq.dtype)
context_raw = context_alpha.to(device=gate_seq.device, dtype=gate_seq.dtype) * context_scalar
traj_raw = freq_raw.view(1, L, 1) + context_raw
scale_seq = 1.0 + eps * torch.tanh(traj_raw)
```

For this repo's SOT layout:

```python
y_add = torch.einsum("bdl,bnl,dn -> bdl", gate, C_readout, param)
y_add = y_add * scale
```

This is the implemented equivalent of `u_t = u_t + sigma(z_t) * (r_t h')` on the existing SOT offset.

## Config Alignment

No SOT MRPC/RTE/CoLA experiment YAML currently defines a dataset `max_seq_length`. CF-SOT sets `cf_sot_max_seq_len: 256` for the Fourier basis only and dynamically rebuilds the basis when runtime sequence length is longer.

| Task | Base SOT config | CF-SOT config | LR | Batch | Epochs | Seed |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MRPC | `cfg/final/exps/mamba-130m/glue_mrpc/state_tuning.yaml` | `cfg/final/exps/mamba-130m/glue_mrpc/cf_sot.yaml` | 0.0002 | 4 | 10 | script default 88 |
| RTE | `cfg/final/exps/mamba-130m/glue_rte/state_tuning.yaml` | `cfg/final/exps/mamba-130m/glue_rte/cf_sot.yaml` | 0.001 | 4 | 10 | script default 88 |
| CoLA | `cfg/final/exps/mamba-130m/glue_cola/state_tuning.yaml` | `cfg/final/exps/mamba-130m/glue_cola/cf_sot.yaml` | 0.0002 | 4 | 10 | script default 88 |

## Run Commands

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --cfg cfg/final/exps/mamba-130m/glue_mrpc/cf_sot.yaml --seed 88 --overwrite --output_dir outputs/cf_sot/mamba-130m/glue_mrpc
CUDA_VISIBLE_DEVICES=0 python train.py --cfg cfg/final/exps/mamba-130m/glue_rte/cf_sot.yaml --seed 88 --overwrite --output_dir outputs/cf_sot/mamba-130m/glue_rte
CUDA_VISIBLE_DEVICES=0 python train.py --cfg cfg/final/exps/mamba-130m/glue_cola/cf_sot.yaml --seed 88 --overwrite --output_dir outputs/cf_sot/mamba-130m/glue_cola
```

Or:

```bash
bash scripts/run_cf_sot_glue.sh
```

## Local Checks

- Static Python compile passed for `modules/suffix_tuning.py`, `train.py`, `scripts/sanity_check_cf_sot.py`, and `tools/inspect_cf_sot.py`.
- CF PEFT JSON validation passed: `use_cf_sot=true`, `use_tf_sot=false`, `cf_sot_max_seq_len=256`.
- Config alignment check passed for MRPC/RTE/CoLA learning rate, batch size, epochs, data, model, and precision.
- Full tensor sanity check was not run locally because this desktop Python environment does not have `torch` installed. Run this on the server:

```bash
python scripts/sanity_check_cf_sot.py --output outputs/cf_sot/sanity_check.json
```

Expected initial checks:

```text
max_abs_diff between SOT and CF-SOT at init: 0.0
initial scale min: 1.0
initial scale max: 1.0
initial mean|scale-1|: 0.0
cf_sot_tf_coeff grad norm: > 0
cf_sot_context_alpha grad norm: > 0
```

## Open Items

- `attention_mask` is not available in `SuffixTuningBiasProcessor.forward`, so CF-SOT currently uses unmasked sequence centering.
- Training was not run locally.
- `tools/inspect_cf_sot.py` can inspect checkpoint parameters directly; exact context contribution and full scale statistics require gate trajectories or a provided context proxy.
