# vllm-hybrid-quant

**Hybrid GPTQ-INT4 + FP8 per-layer dispatch for vLLM** — GPTQ-Marlin INT4 for MoE experts, FP8 CUTLASS block GEMMs for dense layers. Auto-detected, no new CLI flags.

## Results

NVIDIA DGX Spark (GB10, 128 GB unified, 273 GB/s) serving Qwen3.5-122B-A10B at 256k context:

| Metric | GPTQ-INT4 (before) | Hybrid GPTQ+FP8 (after) | Change |
|---|---|---|---|
| **Decode** | 15.0 tok/s | **21.5 tok/s** | **+43%** |
| **ITL P50** | 67 ms | **46.4 ms** | **-31%** |
| **ITL P99** | 67 ms | **48.4 ms** | **-28%** |
| **Model memory** | 68.4 GiB | **64.1 GiB** | **-4.3 GiB** |
| **KV cache** | 25.8 GiB | **31.4 GiB** | **+21%** |

FP8 weights sourced from the official calibrated `Qwen/Qwen3.5-122B-A10B-FP8` checkpoint.

## The Problem

Large MoE models like Qwen3.5-122B-A10B have two categories of weights:

1. **MoE expert weights** — bulk of parameters, but only a fraction active per token. GPTQ-INT4 (0.5 B/param via Marlin kernels) works great here.
2. **Dense layers** — attention projections (Q/K/V/O), shared experts. Active on *every* token and dominate per-token memory bandwidth.

The GPTQ-INT4 checkpoint only quantizes experts. Dense layers stay at BF16 (2 B/param), creating a bandwidth bottleneck: ~18.2 GB of the ~20.6 GB per-token read comes from dense layers.

The official FP8 checkpoint has calibrated FP8 weights for these dense layers (1 B/param, block-128 scales) — but the full FP8 checkpoint (127 GB) won't fit on a single Spark at useful context lengths.

**The hybrid approach:** combine GPTQ-INT4 experts with FP8 dense layers. Per-token bandwidth drops from ~20.6 GB to ~11.5 GB → theoretical **24 tok/s**, measured **21.5 tok/s**.

### Why the naive approach fails

Placing FP8 tensors into a GPTQ checkpoint and loading with `--quantization gptq_marlin` doesn't help — vLLM routes all non-GPTQ layers to `UnquantizedLinearMethod`, which casts FP8→BF16 at load time. You save VRAM but get zero bandwidth improvement.

## The Fix

A patch to two files in vLLM v0.17.1 (+116 lines) that makes `GPTQMarlinConfig` auto-detect FP8 layers and route them to `Fp8LinearMethod`:

**`gptq_marlin.py`** — `maybe_update_config()` scans safetensors metadata for `float8_e4m3fn` weight + `weight_scale_inv` tensor pairs. Infers block size from shape ratios. Constructs `Fp8Config`. Maps HF checkpoint names to vLLM runtime names via the model's `WeightsMapper` (model-agnostic — works with any MoE architecture).

**`gptq_utils.py`** — `get_linear_quant_method()` checks for FP8 layers before falling back to `UnquantizedLinearMethod`. Handles fused modules (qkv_proj, gate_up_proj) via `packed_modules_mapping`. Defers `deepcopy` to the GPTQ mutation path only — non-GPTQ layers skip it entirely.

No new config names, no new CLI flags. Just `--quantization gptq_marlin` as usual.

## Quick Start

### 1. Build the hybrid checkpoint

Downloads only the needed shards (~8 GB) from the FP8 checkpoint, extracts the 307 non-expert FP8 tensors, and grafts them into your GPTQ-INT4 checkpoint:

```bash
python3 build-hybrid-checkpoint.py \
    --gptq-dir ~/models/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --fp8-repo Qwen/Qwen3.5-122B-A10B-FP8 \
    --output ~/models/qwen3.5-122b-a10b-fp8hybrid
```

### 2. Apply the vLLM patch

```bash
cd /path/to/vllm-src  # v0.17.1
git apply /path/to/vllm-hybrid-quant/vllm-patch/hybrid-fp8-dispatch.patch
```

### 3. Launch

```bash
vllm serve ~/models/qwen3.5-122b-a10b-fp8hybrid \
    --served-model-name Qwen/Qwen3.5-122B-A10B \
    --quantization gptq_marlin \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 \
    --max-model-len 262144 \
    --max-num-seqs 1
```

You should see in the logs:
```
Hybrid GPTQ+FP8: detected 307 FP8 layers (block_size=[128, 128]). These will use Fp8LinearMethod.
```

## Repository Contents

| File | Description |
|---|---|
| `build-hybrid-checkpoint.py` | Builds the hybrid GPTQ-INT4 + FP8 checkpoint from existing checkpoints |
| `vllm-patch/hybrid-fp8-dispatch.patch` | vLLM v0.17.1 patch — per-layer FP8 dispatch (+116 lines, 2 files) |
| `tests/test_gptq_fp8_hybrid.py` | 16 unit tests (layer matching, FP8 detection, WeightsMapper, pickle, deepcopy deferral) |

## How It Works

```
Checkpoint load (maybe_update_config)
  │
  ├─ Scan safetensors metadata
  │   ├─ FP8 weight + weight_scale_inv? → add to fp8_layers set
  │   └─ Infer block_size from weight/scale shape ratio
  │
  ├─ apply_vllm_mapper() → map HF names to vLLM names
  │
  └─ Per-layer dispatch (get_linear_quant_method)
      ├─ FusedMoE layer? → GPTQ-Marlin MoE kernels (unchanged)
      ├─ GPTQ-quantized linear? → GPTQMarlinLinearMethod (unchanged)
      ├─ In fp8_layers set? → Fp8LinearMethod (FP8 CUTLASS block GEMMs)
      └─ Otherwise → UnquantizedLinearMethod (BF16, for norms/gates/embeddings)
```

192 dense layers routed to `Fp8LinearMethod` across 48 transformer blocks. The approach generalizes to any MoE model where dense layers bottleneck decode throughput.

## Tested On

- **Hardware:** NVIDIA DGX Spark GB10 (128 GB unified LPDDR5X, 273 GB/s, SM 12.1)
- **Software:** vLLM v0.17.1, CUDA 13.0, FlashInfer attention backend
- **Model:** Qwen3.5-122B-A10B (122B total / 10B active per token)
- **Context:** 256k max, FP8 KV cache

## License

Apache 2.0 — same as vLLM.
