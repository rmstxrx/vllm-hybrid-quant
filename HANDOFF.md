# Session Handoff — vllm-hybrid-quant

**Date:** 2026-03-18
**Status:** Working prototype deployed on Apollyon, repo scaffolded, PR prep documented.

## What Was Accomplished

### Core Implementation (Complete)
- `GPTQMarlinHybridFp8Config` — per-layer quantization dispatch for vLLM
- Hybrid checkpoint: GPTQ-INT4 MoE experts + FP8 block-128 dense layers
- **Result: 15 → 21.6 tok/s (+44%)** on Qwen3.5-122B-A10B / DGX Spark

### Repo Structure (Complete)
- `vllm-hybrid-quant/` — standalone repo with source, patches, docs, tests
- `docs/ARCHITECTURE.md` — vLLM quantization dispatch deep-dive
- `docs/DEBUGGING_JOURNAL.md` — 10 pitfalls documented
- `docs/PR_PREPARATION.md` — upstream PR checklist with estimated effort
- `tests/test_hybrid_routing.py` — routing correctness tests

### Deployment (Complete)
- Module installed in vLLM source on Apollyon
- `quantize_config.json` updated in hybrid checkpoint
- systemd unit updated with `--quantization gptq_marlin_hybrid_fp8`
- HANDOFF.md updated with benchmark results
- Server running at 21.6 tok/s

## What Needs to Happen Next

### Priority 1: Push to GitHub
```bash
cd ~/vllm-hybrid-quant   # or wherever the repo lands
git init && git add -A && git commit -m "Initial: hybrid GPTQ+FP8 quantization for vLLM"
gh repo create rmstxrx/vllm-hybrid-quant --public --source=.
git push -u origin main
```

### Priority 2: PR Polish (see docs/PR_PREPARATION.md for details)

**Must-fix items:**

1. **Remove hardcoded `_HF_TO_VLLM_PREFIXES`**
   - File: `src/vllm_hybrid_quant/gptq_marlin_hybrid_fp8.py`, lines 70-75
   - Approach: Override `apply_vllm_mapper()` to capture the mapper, defer FP8 layer set construction to `_ensure_fp8_initialized()` (which runs after `configure_quant_config` sets the mapper)
   - This makes the config work with ANY model, not just Qwen3.5

2. **Eliminate `deepcopy(self)` in `get_quant_method()`**
   - File: same, around line 345
   - `get_dynamic_override()` mutates the config — cache results instead
   - Minor perf issue (~300 deep copies during init) but reviewers will flag it

3. **Run full test suite**
   - `pytest tests/test_hybrid_routing.py -v`
   - Need to mock `get_safetensors_params_metadata` since tests shouldn't need a 70 GB checkpoint

4. **Quality validation**
   - Run GPQA Diamond eval on the hybrid model to confirm no quality degradation vs pure GPTQ-INT4
   - Benchmark with different prompt lengths (prefill-heavy vs decode-heavy)

### Priority 3: Upstream PR

- Target: `vllm-project/vllm` main branch
- Include: the config file + __init__.py patch + model.py patch + gptq_marlin.py unquant_dtypes patch
- PR description should reference the benchmark results and explain the architectural approach
- Link to DGX Spark forum thread: `forums.developer.nvidia.com/t/qwen-qwen3-5-122b-a10b/361639`

## Key Files on Apollyon

| File | Description |
|------|-------------|
| `/home/rmstxrx/inference/vllm-src/vllm/model_executor/layers/quantization/gptq_marlin_hybrid_fp8.py` | Deployed production version |
| `/home/rmstxrx/inference/models/hf/qwen3.5-122b-a10b-fp8hybrid/quantize_config.json` | Checkpoint config |
| `/home/rmstxrx/inference/HANDOFF.md` | Inference stack handoff (updated) |
| `/home/rmstxrx/inference/bin/frankenstein.py` | Hybrid checkpoint builder |
| `~/.config/systemd/user/vllm-inference-fp8hybrid.service` | systemd unit |

## Key Files in Repo

| File | Description |
|------|-------------|
| `src/vllm_hybrid_quant/gptq_marlin_hybrid_fp8.py` | Clean copy of the config |
| `patches/*.patch` | vLLM integration patches |
| `configs/quantize_config.json` | Example checkpoint config |
| `docs/ARCHITECTURE.md` | How vLLM quantization dispatch works |
| `docs/DEBUGGING_JOURNAL.md` | 10 pitfalls and their fixes |
| `docs/PR_PREPARATION.md` | PR checklist with effort estimates |
| `tests/test_hybrid_routing.py` | Routing correctness tests |

## Context for New Session

The most important thing to understand: vLLM picks ONE `QuantizationConfig` per model, and that config's `get_quant_method()` is called for every layer during model initialization. Our config subclasses `GPTQMarlinConfig` and overrides `get_quant_method()` to add a three-way dispatch (GPTQ/FP8/Unquantized) instead of the parent's two-way (GPTQ/Unquantized).

The trickiest parts were:
1. HF↔vLLM prefix mapping (different naming conventions between checkpoint and runtime)
2. Fused module resolution (vLLM packs q_proj+k_proj+v_proj into qkv_proj)
3. Multi-process delivery (config is pickled to engine core subprocess)

All three are documented in `docs/DEBUGGING_JOURNAL.md`.
