# vllm-hybrid-quant

**Hybrid GPTQ-INT4 + FP8 per-layer quantization for vLLM** — GPTQ-Marlin INT4 for MoE experts, FP8 CUTLASS block GEMMs for dense layers. Auto-detected, no new CLI flags.

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

**The hybrid approach:** combine GPTQ-INT4 experts with FP8 dense layers. Per-token bandwidth drops from ~20.6 GB to ~11.5 GB → theoretical 24 tok/s, measured **21.5 tok/s**.

### Why the naive approach fails

Placing FP8 tensors into a GPTQ checkpoint and loading with `--quantization gptq_marlin` doesn't help — vLLM routes all non-GPTQ layers to `UnquantizedLinearMethod`, which casts FP8→BF16 at load time. You save VRAM but get zero bandwidth improvement.

## Components

This project has two parts:

| Component | Location | Description |
|---|---|---|
| **vLLM patch** | [`rmstxrx/vllm`](https://github.com/rmstxrx/vllm/tree/v0.17.1-hybrid-fp8) | Fork of vLLM v0.17.1 with per-layer FP8 dispatch (+130 lines, 2 files, 19 tests) |
| **Checkpoint builder** | `build-hybrid-checkpoint.py` (this repo) | Builds the hybrid GPTQ-INT4 + FP8 checkpoint from existing HuggingFace checkpoints |

## Quick Start

### 1. Build the hybrid checkpoint

Downloads only the needed shards (~8 GB) from the FP8 checkpoint, extracts the non-expert FP8 tensors, and grafts them into your GPTQ-INT4 checkpoint:

```bash
python3 build-hybrid-checkpoint.py \
    --gptq-dir ~/models/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --fp8-repo Qwen/Qwen3.5-122B-A10B-FP8 \
    --output ~/models/qwen3.5-122b-a10b-fp8hybrid
```

### 2. Install the patched vLLM

```bash
git clone https://github.com/rmstxrx/vllm.git
cd vllm
git checkout v0.17.1-hybrid-fp8
pip install -e .
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

## How It Works

```
Checkpoint load (maybe_update_config)
  │
  ├─ Scan safetensors metadata
  │   ├─ FP8 weight + weight_scale_inv? → add to fp8_layers set
  │   └─ Validate uniform block_size across all FP8 layers
  │
  ├─ apply_vllm_mapper() → map HF names to vLLM runtime names
  │
  └─ Per-layer dispatch (get_linear_quant_method)
      ├─ FusedMoE layer? → GPTQ-Marlin MoE kernels (unchanged)
      ├─ GPTQ-quantized linear? → GPTQMarlinLinearMethod (unchanged)
      ├─ In fp8_layers set? → Fp8LinearMethod (FP8 CUTLASS block GEMMs)
      └─ Otherwise → UnquantizedLinearMethod (BF16, for norms/gates/embeddings)
```

The 307 detected FP8 checkpoint tensors (weight + scale pairs) map to 192 runtime `Fp8LinearMethod` linear layers across 48 transformer blocks, because vLLM fuses some projections at runtime (e.g., `q_proj + k_proj + v_proj → qkv_proj`).

## Tested On

- **Hardware:** NVIDIA DGX Spark GB10 (128 GB unified LPDDR5X, 273 GB/s, SM 12.1)
- **Software:** vLLM v0.17.1 ([patched fork](https://github.com/rmstxrx/vllm/tree/v0.17.1-hybrid-fp8)), CUDA 13.0, FlashInfer attention backend
- **Model:** Qwen3.5-122B-A10B (122B total / 10B active per token)
- **Context:** 256k max, FP8 KV cache

## Known Limitations

- The checkpoint builder assumes the FP8 source uses blockwise FP8 with `weight_scale_inv` tensors. Per-tensor FP8 (with `weight_scale`) is not supported.
- FP8 detection requires that all FP8 layers share the same block size. Mixed block sizes are rejected.
- The `--quantization gptq_marlin` flag is required even though the model is hybrid — vLLM's primary quantization config must be GPTQ for the dispatch to work.
- Non-expert tensor identification uses the heuristic `".experts." not in tensor_name`, which works for Qwen, Mixtral, DeepSeek, and DBRX naming conventions but may need adjustment for other architectures.

## License

Apache 2.0 — same as vLLM.
