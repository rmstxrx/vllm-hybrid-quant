# Publication Handoff — vllm-hybrid-quant

**Date:** 2026-03-18
**Repo:** https://github.com/rmstxrx/vllm-hybrid-quant (1 commit, `a564331`)
**Server:** Apollyon fp8hybrid currently down (was 21.5 tok/s when last running)

---

## What We Built and Why It Matters

We implemented **per-layer mixed-precision quantization dispatch for vLLM** — a
`GPTQMarlinConfig` subclass (`GPTQMarlinHybridFp8Config`) that routes MoE expert
layers to GPTQ-Marlin INT4 kernels and dense layers (attention, shared experts) to
`Fp8LinearMethod` with CUTLASS FP8 block GEMMs. This is a genuinely novel
contribution: vLLM assumes one quantization method per model, and nobody has
cleanly broken that assumption before.

**Result: 15 → 21.5 tok/s (+43%) on Qwen3.5-122B-A10B / DGX Spark**, with -30% ITL
P50, -64% ITL P99, and +21% KV cache capacity. Zero quality degradation — FP8
weights come from the official calibrated checkpoint.

This is publishable because:
- The technique generalizes to any MoE model where dense layers bottleneck decode
- The DGX Spark is NVIDIA's own inference appliance — compelling showcase
- It's zero changes to vLLM core, pure config subclass — reviewers love that
- Nobody else has solved this; the vLLM + DGX Spark forum thread has no solutions

---

## Two Publication Tracks

### Track 1: vLLM Upstream PR

**Target:** `vllm-project/vllm` main branch

**What to include in the PR:**
1. `gptq_marlin_hybrid_fp8.py` — the config class
2. `__init__.py` patch — adds to registry, Literal type, method_to_config
3. `model.py` patch — adds to override probe list
4. `gptq_marlin.py` patch — `float8_e4m3fn` in `unquant_dtypes` (prerequisite)
5. Unit tests (need to be written/completed)
6. Documentation update for vLLM's quantization docs

**Must-fix before PR submission (4 items):**

#### 1. Remove hardcoded `_HF_TO_VLLM_PREFIXES` → model-agnostic

The prefix mapping on line 70 is hardcoded for Qwen3.5:
```python
_HF_TO_VLLM_PREFIXES = {
    "model.visual.": "visual.",
    "lm_head.": "language_model.lm_head.",
    "model.language_model.": "language_model.model.",
}
```

The model class's `hf_to_vllm_mapper` has the real mapping, available via
`apply_vllm_mapper()` (called by `configure_quant_config()` in
`model_executor/model_loader/utils.py`).

**Recommended approach:** Override `apply_vllm_mapper()` to capture the mapper's
`orig_to_new_prefix` dict. Defer FP8 layer set construction to
`_ensure_fp8_initialized()`, which already runs at the right time (after
`configure_quant_config()` sets both `packed_modules_mapping` AND the mapper).
This eliminates the TODO and makes the config work with Mixtral, DeepSeek-V3,
or any future MoE model — not just Qwen3.5.

**Key insight from debugging:** `apply_vllm_mapper()` runs in `initialize_model()`
AFTER `maybe_update_config()` but BEFORE `get_quant_method()`. The existing
`_ensure_fp8_initialized()` lazy-init pattern was built for pickle resilience —
repurposing it for deferred prefix mapping is the same pattern.

#### 2. Eliminate `deepcopy(self)` in `get_quant_method()`

Around line 265, every `get_quant_method()` call does `deepcopy(self)` because
`get_dynamic_override()` mutates the config. This deep-copies the 614-entry
`_fp8_hf_layers` set + `Fp8Config` on every layer (~300 calls during init).

**Fix:** Cache the dynamic override decisions once. Build a `set[str]` of prefixes
that should be skipped by negative-match dynamic rules in `maybe_update_config()`.
Then `get_quant_method()` becomes a simple set lookup.

#### 3. Unit tests with mocked checkpoint metadata

`tests/test_hybrid_routing.py` has routing correctness tests but they use hardcoded
mock data. For the PR:
- Mock `get_safetensors_params_metadata()` so tests don't need a 70 GB checkpoint
- Add a test that actually calls `get_quant_method()` with mock `LinearBase` /
  `FusedMoE` layers (not just the helper functions)
- Verify the pickle round-trip end-to-end

#### 4. PR description and documentation

The PR description should:
- Lead with the benchmark (21.5 tok/s, +43%)
- Explain the architectural approach (diagram from README.md's dispatch table)
- Reference the DGX Spark forum thread
- Note that this is purely additive — no breaking changes

vLLM docs need:
- Entry in the supported quantization methods table
- Example `quantize_config.json`
- Brief note on how to build a hybrid checkpoint (reference `frankenstein.py`)

### Track 2: Blog Post / Technical Write-up

The debugging journey is the real story. Content that doesn't exist anywhere:

1. **The silent FP8→BF16 cast** — `UnquantizedLinearMethod` saves VRAM but
   doesn't change GEMM bandwidth. This catches everyone who puts FP8 tensors
   in a non-FP8 quantization config.

2. **HF ↔ vLLM prefix mapping** — `model.language_model.` vs
   `language_model.model.` — the `WeightsMapper` transformation that breaks
   any code comparing checkpoint names to runtime prefixes.

3. **Fused module resolution** — vLLM packs `q_proj+k_proj+v_proj` into
   `qkv_proj`, but the checkpoint has them separate. `packed_modules_mapping`
   is the resolution mechanism, and mixed fusions (`in_proj_ba` where
   `in_proj_a`/`in_proj_b` are BF16) must be handled correctly.

4. **Multi-process pickle resilience** — the config is pickled to the engine
   core subprocess. Custom attributes survive, but the plugin system doesn't —
   entry-point registered configs may not be available in child processes.

5. **Unified memory reclaim on DGX Spark** — 60-90 seconds after SIGKILL for
   70 GB to be freed. Unique to unified memory systems.

All of this is documented in `docs/DEBUGGING_JOURNAL.md` in the repo.

---

## Current State of Everything

### Deployed on Apollyon

| File | Status |
|------|--------|
| `~/inference/vllm-src/.../gptq_marlin_hybrid_fp8.py` | Deployed, matches repo |
| `~/inference/vllm-src/.../quantization/__init__.py` | Patched (3 additions) |
| `~/inference/vllm-src/.../config/model.py` | Patched (1 addition) |
| `~/inference/vllm-src/.../gptq_marlin.py:327-328` | Patched (float8_e4m3fn) |
| `~/inference/models/hf/qwen3.5-122b-a10b-fp8hybrid/quantize_config.json` | Updated |
| `~/.config/systemd/user/vllm-inference-fp8hybrid.service` | `--quantization gptq_marlin_hybrid_fp8` |
| Server | **Down** — needs `inference-serve fp8hybrid` |

### GitHub Repo

| File | Purpose |
|------|---------|
| `src/vllm_hybrid_quant/gptq_marlin_hybrid_fp8.py` | Production source (identical to deployed) |
| `patches/*.patch` | Integration patches for vLLM source |
| `configs/quantize_config.json` | Example checkpoint config |
| `docs/ARCHITECTURE.md` | vLLM quantization dispatch deep-dive |
| `docs/DEBUGGING_JOURNAL.md` | 10 pitfalls and fixes (blog post material) |
| `docs/PR_PREPARATION.md` | Full PR checklist with effort estimates |
| `tests/test_hybrid_routing.py` | Routing correctness tests (need expansion) |

### What the Intermediate Session Changed

Between the original implementation session and now, another session:
- Confirmed the deployed code works (server was serving 21.5 tok/s)
- Restored the `.py` source file (it was accidentally deleted, only `.pyc` remained)
- Re-applied the `__init__.py` and `model.py` patches (they were lost)
- Verified all three vLLM patches are currently applied

No changes were made to the actual `gptq_marlin_hybrid_fp8.py` logic — the code
that's deployed and in the repo is the same code that was written in the original
session.

---

## Recommended Session Plan

### If doing PR work:

1. **Read** `docs/PR_PREPARATION.md` and `docs/ARCHITECTURE.md` for full context
2. **Fix item 1** (hardcoded prefixes) — this is the biggest code change
   - Override `apply_vllm_mapper()` to store the prefix map
   - Move FP8 layer set construction into `_ensure_fp8_initialized()`
   - Remove `_HF_TO_VLLM_PREFIXES` constant entirely
3. **Fix item 2** (deepcopy) — cache dynamic override results
4. **Expand tests** — mock `get_safetensors_params_metadata`, test full `get_quant_method()` flow
5. **Draft PR description** — benchmark table, architecture diagram, forum link
6. **Test on Apollyon** — `inference-serve fp8hybrid` and verify it still hits 21+ tok/s

### If doing blog post:

1. **Read** `docs/DEBUGGING_JOURNAL.md` — this is the skeleton
2. **Structure:** Problem → naive approach → why it fails → the fix → results
3. **Key diagrams needed:**
   - MoE model bandwidth breakdown (expert vs dense layers)
   - vLLM quantization dispatch flow (before and after)
   - Benchmark comparison chart
4. **Audience:** vLLM practitioners running MoE models on constrained hardware

### Quick restart to verify:

```bash
# On Apollyon:
export PATH="$HOME/inference/bin:$PATH"
inference-serve fp8hybrid
# Wait ~9 min for load + compile
inference-bench fp8hybrid-verify
# Should show ~21.5 tok/s
```
