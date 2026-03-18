# Architecture: vLLM Quantization Dispatch

## How vLLM Routes Layers to Quantization Methods

### The Config Chain

1. **Checkpoint detection** (`config/model.py:_verify_quantization`):
   - Reads `quantize_config.json` from the model directory
   - Iterates `QUANTIZATION_METHODS` + custom `overrides` list
   - Calls `override_quantization_method(hf_quant_cfg, user_quant)` on each
   - First match wins → sets `model_config.quantization`

2. **Config construction** (`config/vllm.py:_get_quantization_config`):
   - `get_quant_config()` → `from_config(quantize_config_dict)` → instance
   - `maybe_update_config(model_name)` — scans safetensors metadata
   - Returns the `QuantizationConfig` instance stored as `vllm_config.quant_config`

3. **Multi-process delivery**:
   - `VllmConfig` (including `quant_config`) is **pickled** and sent to the engine core subprocess
   - All custom attributes must survive pickle. Standard Python types (dict, set, str) survive fine.

4. **Model initialization** (`model_executor/model_loader/utils.py:initialize_model`):
   - `configure_quant_config(quant_config, model_class)` — sets `packed_modules_mapping` from the model class
   - Model `__init__` constructs layers, each calling `quant_config.get_quant_method(layer, prefix)`

### The Dispatch Point

`LinearBase.__init__()` (~line 249 of `linear.py`):

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
else:
    self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
```

This is the SINGLE routing point for all linear layers. The `quant_method` determines:
- How weight buffers are allocated (`create_weights`)
- How weights are post-processed after loading (`process_weights_after_loading`)
- How matmuls execute at inference time (`apply`)

### MoE Expert Path (Separate)

MoE expert weights do NOT flow through `LinearBase`. They use `FusedMoE` → `FusedMoEMethodBase`, which has its own `get_quant_method` dispatch:

```python
if isinstance(layer, FusedMoE):
    return GPTQMarlinMoEMethod(...)  # fused Marlin MoE kernels
```

This means per-layer dispatch for dense layers is completely orthogonal to MoE quantization.

## The Hybrid Config's Intervention

`GPTQMarlinHybridFp8Config.get_quant_method()` replaces the parent's single-method routing with a three-way dispatch:

```
LinearBase layer arrives
        │
        ▼
  ┌─ is_layer_gptq_quantized()?
  │     YES ──► GPTQMarlinLinearMethod (parent, GPTQ Marlin kernels)
  │      NO
  │     ▼
  ├─ all constituent HF names in fp8_layer_set?
  │     YES ──► Fp8LinearMethod(block_quant=True, block_size=[128,128])
  │      NO        └─ CUTLASS FP8 block GEMMs, FP8 weight storage
  │     ▼
  └─ UnquantizedLinearMethod (BF16 native)
         └─ embeddings, norms, small gates (in_proj_a/b)
```

## Prefix Naming Conventions

### The Mismatch Problem

HF checkpoint tensor names:
```
model.language_model.layers.0.self_attn.q_proj.weight
model.language_model.layers.0.linear_attn.in_proj_qkv.weight
```

vLLM layer prefixes received in `get_quant_method()`:
```
language_model.model.layers.0.self_attn.qkv_proj
language_model.model.layers.0.linear_attn.in_proj_qkvz
```

Two transformations:
1. **WeightsMapper** swaps `model.language_model.` → `language_model.model.`
2. **Module fusion** packs `q_proj+k_proj+v_proj` → `qkv_proj`

### Resolution Strategy

1. `_build_fp8_layer_set()` stores BOTH naming conventions using `_HF_TO_VLLM_PREFIXES`
2. `_resolve_fused_to_hf_names()` unpacks fused names using `packed_modules_mapping`
3. Check: ALL constituent HF names must be in the FP8 set

This correctly handles mixed fusions like `in_proj_ba = [in_proj_b, in_proj_a]` where
`in_proj_b` and `in_proj_a` are BF16 (not in FP8 set) despite matching `.*attn.*`.

## Key Files

| File | Role |
|------|------|
| `layers/linear.py:LinearBase.__init__` | The dispatch point |
| `layers/quantization/base_config.py:QuantizationConfig` | Base class with `get_quant_method()` |
| `layers/quantization/gptq_marlin.py:GPTQMarlinConfig` | Parent — GPTQ Marlin routing |
| `layers/quantization/fp8.py:Fp8LinearMethod` | Target for dense FP8 layers |
| `layers/quantization/fp8.py:Fp8Config` | Config companion for Fp8LinearMethod |
| `layers/fused_moe/layer.py:FusedMoE` | MoE expert path (separate from LinearBase) |
| `model_executor/model_loader/utils.py:configure_quant_config` | Sets packed_modules_mapping |
| `config/model.py:_verify_quantization` | Override detection chain |
| `config/vllm.py:_get_quantization_config` | Config construction + maybe_update_config |
