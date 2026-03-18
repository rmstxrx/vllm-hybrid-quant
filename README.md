# vllm-hybrid-quant

**Per-layer mixed-precision quantization dispatch for vLLM** — GPTQ-Marlin INT4 for MoE experts + FP8 CUTLASS GEMMs for dense layers.

## Results

On NVIDIA DGX Spark (GB10, 128 GB unified, SM 12.1) serving Qwen3.5-122B-A10B:

| Metric | GPTQ-INT4 only | Hybrid GPTQ+FP8 | Change |
|--------|---------------|------------------|--------|
| **Decode** | 15.0 tok/s | **21.6 tok/s** | **+44%** |
| **ITL P50** | 66 ms | **46 ms** | -30% |
| **ITL P99** | 133 ms | **48 ms** | -64% |
| **KV cache** | 25.7 GiB | **31.2 GiB** | +21% |

No quality degradation — all FP8 weights are from the official `Qwen/Qwen3.5-122B-A10B-FP8` checkpoint with calibrated scales.

## The Problem

Large MoE models like Qwen3.5-122B-A10B have two categories of weights:

1. **MoE expert weights** — the bulk of parameters, but only a fraction are active per token. GPTQ-INT4 (0.5 B/param via Marlin kernels) works great here.
2. **Dense layers** — attention projections (Q/K/V/O), shared experts, embeddings. These are active on *every* token and dominate per-token memory bandwidth.

The GPTQ-INT4 checkpoint only quantizes the experts. Dense layers stay at BF16 (2 B/param), creating a bandwidth bottleneck: ~18.2 GB of the ~20.6 GB per-token bandwidth comes from dense layers.

The official FP8 checkpoint quantizes these dense layers to FP8 (1 B/param) with calibrated block-128 scales. The idea: combine GPTQ-INT4 experts with FP8 dense layers to get the best of both.

### Why naive approaches fail

Simply replacing BF16 tensors with FP8 in the GPTQ checkpoint (what `frankenstein.py` does) puts FP8 data on disk, but vLLM's `GPTQMarlinConfig` routes **all** non-GPTQ layers to `UnquantizedLinearMethod`, which:
1. Allocates weight buffers at `params_dtype` (BF16)
2. Casts FP8→BF16 at load time
3. Runs BF16 GEMMs at BF16 bandwidth

FP8 saves disk/VRAM but the matmuls run at identical bandwidth. No throughput improvement.

## The Solution

`GPTQMarlinHybridFp8Config` — a `GPTQMarlinConfig` subclass that implements **per-layer quantization dispatch** in `get_quant_method()`:

```
┌──────────────────────────────────────────────────────┐
│              get_quant_method(layer, prefix)          │
├──────────────┬───────────────────┬───────────────────┤
│  FusedMoE    │  LinearBase +     │  LinearBase +     │
│  layers      │  FP8 checkpoint   │  everything else  │
│              │  weights          │                   │
├──────────────┼───────────────────┼───────────────────┤
│  GPTQ Marlin │  Fp8LinearMethod  │  Unquantized      │
│  MoE kernels │  (CUTLASS FP8     │  LinearMethod     │
│  (parent)    │   block GEMMs)    │  (BF16 native)    │
└──────────────┴───────────────────┴───────────────────┘
```

Key design decisions:
- **FP8 layer detection via checkpoint metadata** — scans safetensors for `float8_e4m3fn` tensors at config init time. No hardcoded layer lists.
- **Dual-prefix storage** — stores both HF (`model.language_model.layers.N...`) and vLLM-mapped (`language_model.model.layers.N...`) prefixes because vLLM's `WeightsMapper` renames them.
- **Fused module resolution** — vLLM fuses `q_proj+k_proj+v_proj→qkv_proj`. Uses `packed_modules_mapping` to resolve back to HF names for FP8 set lookup. Correctly rejects mixed fusions (e.g. `in_proj_ba` where `in_proj_a`/`in_proj_b` are BF16 despite matching `.*attn.*`).
- **Multi-process pickle resilience** — `_ensure_fp8_initialized()` lazily rebuilds FP8 state if lost during pickle to the engine core subprocess.
- **Zero core vLLM changes** — no modifications to `LinearBase`, `FusedMoE`, `Fp8LinearMethod`, or any kernel code. Pure `QuantizationConfig` subclass.

## Repository Structure

```
vllm-hybrid-quant/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0
├── src/vllm_hybrid_quant/
│   └── gptq_marlin_hybrid_fp8.py      # The config class (production version)
├── configs/
│   └── quantize_config.json           # Example checkpoint config
├── patches/
│   ├── __init__.py.patch              # vLLM quantization __init__.py additions
│   ├── model.py.patch                 # vLLM config/model.py override list
│   └── gptq_marlin.py.patch           # unquant_dtypes FP8 addition (prerequisite)
├── docs/
│   ├── ARCHITECTURE.md                # Deep-dive on vLLM's quantization dispatch
│   ├── DEBUGGING_JOURNAL.md           # Session findings and pitfalls
│   └── PR_PREPARATION.md             # Upstream PR checklist and design notes
├── tests/
│   └── test_hybrid_routing.py         # Routing correctness tests
└── HANDOFF.md                         # Session handoff for continuation
```

## Installation (current: local vLLM source patch)

```bash
# 1. Copy the config module into vLLM's quantization package
cp src/vllm_hybrid_quant/gptq_marlin_hybrid_fp8.py \
   $VLLM_SRC/vllm/model_executor/layers/quantization/

# 2. Apply the integration patches
cd $VLLM_SRC && git apply patches/*.patch

# 3. Update checkpoint's quantize_config.json
cp configs/quantize_config.json /path/to/hybrid-checkpoint/

# 4. Launch
vllm serve /path/to/hybrid-checkpoint \
    --quantization gptq_marlin_hybrid_fp8 \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8
```

## Prerequisites

1. **Hybrid checkpoint** built by `frankenstein.py`:
   - MoE expert weights: GPTQ-INT4 (from `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`)
   - Dense layer weights: FP8 E4M3 with block-128 scales (from `Qwen/Qwen3.5-122B-A10B-FP8`)
   - Norms/gates/embeddings: original dtype

2. **vLLM 0.17.x** with CUDA support and CUTLASS FP8 block kernels (SM ≥ 89)

3. **Pre-existing patch**: `gptq_marlin.py:314` must include `torch.float8_e4m3fn` in `unquant_dtypes` for the hybrid checkpoint to load at all

## License

Apache 2.0 — matching vLLM's license for upstream compatibility.
