# PR Preparation: Upstream vLLM Contribution

## Status: Working Prototype → Needs Polish for PR

The core implementation is proven (21.6 tok/s, +44% on Qwen3.5-122B-A10B), but several items need attention before submitting to vLLM.

## Must-Fix for PR

### 1. Remove Hardcoded `_HF_TO_VLLM_PREFIXES`

**Current state:** The prefix mapping is hardcoded for Qwen3.5:
```python
_HF_TO_VLLM_PREFIXES = {
    "model.visual.": "visual.",
    "lm_head.": "language_model.lm_head.",
    "model.language_model.": "language_model.model.",
}
```

**Target:** Use the model's actual `hf_to_vllm_mapper` (a `WeightsMapper` instance). This is available via `apply_vllm_mapper()` which vLLM already calls on quant configs during `configure_quant_config()`.

**Approach:** Override `apply_vllm_mapper()` to capture the mapper's `orig_to_new_prefix` dict, then use it in `_build_fp8_layer_set()`. This makes the config fully model-agnostic.

**Challenge:** `apply_vllm_mapper()` is called in `initialize_model()`, which runs AFTER `maybe_update_config()`. The FP8 layer set is built in `maybe_update_config()`. Options:
  - (a) Defer FP8 layer set construction to `_ensure_fp8_initialized()` (lazy, on first `get_quant_method()` call when mapper is available)
  - (b) Build with HF names only, apply mapper transformation in `get_quant_method()` by reverse-mapping the vLLM prefix back to HF

Option (a) is cleaner — `_ensure_fp8_initialized()` already exists for pickle resilience and runs at the right time (after `configure_quant_config()` sets `packed_modules_mapping`).

### 2. Eliminate `deepcopy(self)` in `get_quant_method()`

**Current state:** `get_dynamic_override(cloned_config, ...)` mutates the config, so we deepcopy on every call.

**Target:** Cache the dynamic override decisions once in `maybe_update_config()` or on first call. Build a `set[str]` of prefixes that should be skipped by dynamic rules.

### 3. Add Unit Tests

**Required tests:**
- `test_from_config()` — parses quantize_config.json correctly
- `test_pickle_roundtrip()` — `_fp8_hf_layers`, `_model_name`, `_fp8_config` survive pickle
- `test_routing_fused_fp8()` — fused prefix (`qkv_proj`) with all-FP8 constituents → Fp8LinearMethod
- `test_routing_fused_mixed()` — fused prefix (`in_proj_ba`) with BF16 constituents → UnquantizedLinearMethod
- `test_routing_gptq()` — expert layer prefix → GPTQMarlinLinearMethod (parent)
- `test_routing_moe()` — FusedMoE layer → GPTQMarlinMoEMethod (parent)
- `test_override_quantization_method()` — intercepts correctly for our quant_method string

**Test infrastructure needed:** Mock `get_safetensors_params_metadata()` to avoid requiring an actual checkpoint.

### 4. Generalize Beyond Qwen3.5

The config should work for ANY MoE model with a hybrid checkpoint. Currently Qwen-specific aspects:
- `_HF_TO_VLLM_PREFIXES` (fix via item 1)
- `fp8_config.weight_block_size` — already configurable via quantize_config.json
- `dynamic` negative-match patterns — already configurable, model-specific by design

After fixing item 1, the config is model-agnostic.

### 5. Documentation

- Update vLLM's quantization docs to mention `gptq_marlin_hybrid_fp8`
- Add to the supported quantization methods table
- Example checkpoint creation workflow (reference `frankenstein.py`)

## Nice-to-Have for PR

### 6. Support `--quantization` Auto-Detection

Currently requires either:
- `quant_method: "gptq_marlin_hybrid_fp8"` in quantize_config.json, OR
- `--quantization gptq_marlin_hybrid_fp8` CLI flag

Could add auto-detection: if `quant_method: "gptq"` AND safetensors contain FP8 tensors AND GPTQ dynamic rules exclude those layers → auto-promote to hybrid. This would make `frankenstein.py` checkpoints work without modifying quantize_config.json.

### 7. Benchmark Suite

- Automated benchmark comparing GPTQ-only vs hybrid on multiple model sizes
- Quality evaluation (GPQA, BFCL) to confirm no degradation
- Memory profiling showing per-component VRAM breakdown

### 8. `frankenstein.py` Integration

Include the checkpoint builder in the repo or as a companion tool. Currently lives in `~/inference/bin/frankenstein.py` on the deployment host.

## PR Metadata

- **Target:** `vllm-project/vllm` main branch
- **Type:** Feature — new quantization config
- **Breaking changes:** None. Purely additive — new quant method string, new config class, no changes to existing behavior.
- **Dependencies:** Requires pre-existing `gptq_marlin.py` unquant_dtypes patch (line 314) — should be included in the PR or submitted as a prerequisite.
- **License:** Apache 2.0 (matches vLLM)

## Estimated Effort

| Item | Effort | Priority |
|------|--------|----------|
| Remove hardcoded prefixes | Medium | Must |
| Eliminate deepcopy | Low | Must |
| Unit tests | Medium | Must |
| Generalize | Low (mostly done) | Must |
| Documentation | Low | Must |
| Auto-detection | Medium | Nice |
| Benchmark suite | High | Nice |
| frankenstein.py | Low | Nice |

Total must-fix: ~1-2 sessions.
