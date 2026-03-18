# Debugging Journal

Chronological record of every pitfall, wrong turn, and key insight from the development session. These are the things that cost hours and aren't in any documentation.

## 1. UnquantizedLinearMethod Silently Casts FP8→BF16

**Symptom:** Hybrid checkpoint loads, serves correctly, uses less VRAM than pure GPTQ-INT4, but decode throughput is identical (15 tok/s).

**Root cause:** `UnquantizedLinearMethod.create_weights()` allocates at `params_dtype` (BF16). When the weight loader encounters an FP8 tensor, it casts to BF16 via `copy_()`. The FP8 data is gone — matmuls run at BF16 bandwidth.

**Lesson:** VRAM savings ≠ bandwidth savings. FP8 weights save disk and VRAM (1 B vs 2 B per param) but if the GEMM kernel operates on BF16, you're still reading 2 B per param worth of BF16 data through the memory bus.

## 2. `validate_fp8_block_shape` Rejects Small Layers

**Symptom:** First launch with FP8 routing crashes immediately:
```
ValueError: Weight output_partition_size = 64 is not divisible by 
weight quantization block_n = 128.
```

**Root cause:** Qwen3.5's `in_proj_a` and `in_proj_b` projections have output dimension 64 — smaller than the FP8 block size of 128. vLLM fuses them into `in_proj_ba` (MergedColumnParallelLinear with partitions [64, 64]). `Fp8LinearMethod.create_weights()` calls `validate_fp8_block_shape()` which requires each partition to be divisible by block_n.

**Fix:** The initial regex-based approach (`-:.*attn.*` matches `linear_attn.in_proj_ba`) was too broad. Replaced with safetensors metadata scanning — only layers that actually have `float8_e4m3fn` weight tensors get routed to FP8. `in_proj_a`/`in_proj_b` are BF16 in the checkpoint, so they're correctly excluded.

**Lesson:** Never trust regex patterns for layer routing. The checkpoint's actual tensor dtypes are the ground truth.

## 3. HF vs vLLM Prefix Naming Mismatch

**Symptom:** `get_quant_method()` is called (confirmed via diagnostics), `_fp8_hf_layers` has 307 entries, `packed_modules_mapping` is populated — but ZERO layers match the FP8 set. All fall through to `UnquantizedLinearMethod`.

**Root cause:** vLLM's `WeightsMapper` renames checkpoint prefixes:
```
HF checkpoint: model.language_model.layers.0.self_attn.q_proj
vLLM prefix:   language_model.model.layers.0.self_attn.qkv_proj
```

The swap `model.language_model.` → `language_model.model.` happens in the model class's `hf_to_vllm_mapper`. The FP8 layer set was built from checkpoint metadata (HF naming) but `get_quant_method()` receives vLLM-mapped prefixes.

**Fix:** `_build_fp8_layer_set()` now stores BOTH HF and vLLM-mapped prefixes using `_HF_TO_VLLM_PREFIXES`.

**Lesson:** Any code that compares checkpoint tensor names with vLLM layer prefixes MUST account for the `hf_to_vllm_mapper` transformation. This is a pervasive source of bugs in custom quantization configs.

## 4. Fused Module Names Don't Match Checkpoint

**Symptom:** Even with prefix normalization, fused names like `qkv_proj` don't appear in the checkpoint (which has separate `q_proj`, `k_proj`, `v_proj`).

**Fix:** `_resolve_fused_to_hf_names()` uses `packed_modules_mapping` to unpack fused names back to their constituent HF names. The check becomes: ALL constituents must be in the FP8 set.

**Lesson:** vLLM's module fusion is model-specific. `packed_modules_mapping` is the canonical source for mapping fused→constituent names.

## 5. `packed_modules_mapping` Is Empty at Test Time

**Symptom:** Unit test shows `packed_modules_mapping = {}` after `from_config()` + `maybe_update_config()`.

**Root cause:** `packed_modules_mapping` is set by `configure_quant_config()` in `model_executor/model_loader/utils.py`, which runs during `initialize_model()` — AFTER `maybe_update_config()` but BEFORE `get_quant_method()`. The unit test didn't call `configure_quant_config()`.

**Impact:** Not a bug in production — just a test artifact. In the actual vLLM flow, `packed_modules_mapping` IS populated by the time `get_quant_method()` runs.

**Lesson:** Always simulate the full vLLM initialization chain in tests: `from_config()` → `maybe_update_config()` → `configure_quant_config()` → `get_quant_method()`.

## 6. vLLM Plugin Entry Points Don't Load in Engine Core Subprocess

**Symptom:** Plugin registered via `vllm.general_plugins` entry point works in the API server process but the engine core subprocess uses `GPTQMarlinConfig` instead of `GPTQMarlinHybridFp8Config`. No routing logs appear.

**Root cause:** The engine core subprocess may reconstruct `VllmConfig` independently. If `load_general_plugins()` doesn't run before `get_quantization_config()` in the child process, the custom method isn't in the registry and `override_quantization_method()` never fires.

**Fix:** Abandoned the plugin approach. Placed the module directly in vLLM's quantization package directory and added it to `__init__.py`'s imports and `method_to_config` dict. This ensures the class is always available regardless of process.

**Lesson:** For quantization configs that must work in vLLM's multi-process architecture, prefer direct source integration over entry-point plugins. The plugin system works for tools/processors that run in the API server process, but quantization configs must be available in the engine core subprocess.

## 7. Pip-Installed Plugin Overrides vLLM-Internal Module

**Symptom:** After placing the module in vLLM's source AND having the pip plugin installed, a warning appears:
```
The quantization method 'gptq_marlin_hybrid_fp8' already exists and will be 
overwritten by the quantization config <class 'vllm_hybrid_fp8_plugin...'>
```

The pip plugin's `@register_quantization_config` decorator fires during `load_general_plugins()`, overwriting the vLLM-internal version with the (possibly stale) plugin version.

**Fix:** Uninstalled the pip plugin package.

**Lesson:** Don't have both a pip plugin AND a vLLM-internal module registered for the same quantization method. The plugin will overwrite the internal one.

## 8. `.pyc` Cache Serves Stale Code

**Symptom:** Edited the source file, restarted vLLM, but the old behavior persists.

**Root cause:** Python's `__pycache__/*.pyc` files cache the compiled bytecode. An editable pip install (`pip install -e .`) uses `.pth` files that point to the source directory, but the source directory's `__pycache__` may have stale `.pyc` from before the edit.

**Fix:** `find /path -name "__pycache__" -exec rm -rf {} +` before every launch during development.

**Lesson:** Always clear `__pycache__` after editing source files in an editable install.

## 9. DGX Spark Unified Memory Takes 60-90s to Reclaim After SIGKILL

**Symptom:** After `kill -9` on a vLLM process using ~70 GB of unified memory, `free -h` shows only 40 GB free even 30 seconds later. Relaunching vLLM fails with:
```
ValueError: Free memory on device cuda:0 (40.23/119.63 GiB) is less than
desired GPU memory utilization (0.85, 101.69 GiB)
```

**Root cause:** The GB10's unified memory (shared CPU/GPU) is managed by the kernel's memory management subsystem. After SIGKILL, the CUDA driver and kernel need to reclaim mapped GPU memory pages. This is much slower than discrete GPU memory deallocation because the pages are in the system's unified address space.

**Fix:** Wait 60-90 seconds after SIGKILL, then `sync; echo 3 > /proc/sys/vm/drop_caches`. Also `fuser /dev/nvidia*` to find zombie processes still holding the device.

**Lesson:** On unified memory systems (DGX Spark, Jetson), always gracefully stop vLLM (`inference-stop`) rather than SIGKILL. If forced to SIGKILL, budget 90 seconds before relaunching.

## 10. `deepcopy(self)` in get_quant_method() Is Expensive

**Identified but not fixed in this session.**

`get_dynamic_override()` mutates the config object, so `get_quant_method()` deep-copies `self` before calling it. This copies the entire `_fp8_hf_layers` set (614 strings) and `_fp8_config` object on EVERY layer construction call (~300 calls during init).

**Fix for PR:** Cache the dynamic override results in `maybe_update_config()` instead of re-evaluating per layer. Or restructure `get_dynamic_override()` to not mutate.
