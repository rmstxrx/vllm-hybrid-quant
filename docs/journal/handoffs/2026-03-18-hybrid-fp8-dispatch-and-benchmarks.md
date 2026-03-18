# Handoff — Hybrid FP8 Dispatch & Benchmarks

**Date:** 2026-03-18
**Author:** Claude (Opus 4.6) + Rômulo
**Status:** Code complete, forum post drafted, GPQA Diamond benchmark needs redo

---

## What was accomplished

### Per-layer FP8 dispatch for vLLM (complete)

Patched 2 files in vLLM v0.17.1 (+116 lines) so that `GPTQMarlinConfig` auto-detects FP8 layers in GPTQ checkpoints and routes them to `Fp8LinearMethod` instead of `UnquantizedLinearMethod`.

**Branch:** `feat/hybrid-gptq-fp8` on `~/inference/vllm-src` (1 squashed commit, `4b995a5`)

**Files changed:**
- `vllm/model_executor/layers/quantization/gptq_marlin.py` — FP8 detection in `maybe_update_config()`, block size inference, `apply_vllm_mapper()` for HF→vLLM name mapping
- `vllm/model_executor/layers/quantization/utils/gptq_utils.py` — FP8 dispatch in `get_linear_quant_method()`, `_is_layer_fp8()` fused module helper, deepcopy deferred to GPTQ mutation path only

**Tests:** 16 passing (`tests/quantization/test_gptq_fp8_hybrid.py`)

### Throughput benchmarks (complete, definitive)

Measured at 256k max context, CUDA graphs, FP8 KV cache, single DGX Spark:

| Metric | GPTQ-INT4 (before) | Hybrid GPTQ+FP8 (after) | Delta |
|---|---|---|---|
| Decode | 15.0 tok/s | **21.5 tok/s** | **+43%** |
| ITL P50 | 67 ms | **46.4 ms** | **-31%** |
| ITL P99 | 67 ms | **48.4 ms** | **-28%** |
| Model memory | 68.4 GiB | **64.1 GiB** | **-4.3 GiB** |
| KV cache | 25.8 GiB | **31.4 GiB** | **+21%** |

Full `inference-bench` results at `/tmp/bench_256k.txt` on Apollyon.

### Repo published (complete)

`https://github.com/rmstxrx/vllm-hybrid-quant` — pushed with:
- `build-hybrid-checkpoint.py` (renamed from frankenstein.py)
- `vllm-patch/hybrid-fp8-dispatch.patch`
- `tests/test_gptq_fp8_hybrid.py`
- Updated README with 256k benchmark numbers

### Forum post drafted (ready to post)

Draft at `/home/claude/forum_post.md` (also in claude.ai outputs). Targets the NVIDIA DGX Spark forum thread: `forums.developer.nvidia.com/t/qwen-qwen3-5-122b-a10b/361639`. Humble tone, acknowledges Intel AutoRound approach, no quality claims.

---

## What needs work

### GPQA Diamond benchmark (blocked — methodology issue)

**Problem:** `lm_eval` with `local-completions` model type uses loglikelihood scoring, which feeds raw text to the completions API. Instruction-tuned models (like Qwen3.5-122B-A10B) expect chat-formatted input, so the loglikelihood scores are meaningless.

**Results so far:**
- `max_length=2048` (truncated): 54.55% — invalid, questions were cut off
- `max_length=8192` (full context): 52.53% — still essentially random, confirming the methodology is wrong, not the model

**Comparison:** Forum user trystan1 got **84.85%** on Intel/Qwen3.5-122B-A10B-int4-AutoRound using 256 concurrent requests with thinking enabled — likely a custom eval script, not lm_eval loglikelihood.

**Options for next session:**
1. **Use `--model vllm` instead of `--model local-completions`** — loads the model directly in-process, bypasses the API. This is how the vLLM docs recommend running lm_eval. Requires enough memory to load the model + run lm_eval, which should be fine on Spark.
2. **Write a custom GPQA evaluator** using the chat completions API with `enable_thinking=True`, parse the model's answer from the response. This matches what the forum poster likely did.
3. **Use `lm_eval` with `--model local-chat-completions` and a generative task** like `gpqa_diamond_cot_zeroshot` instead of `gpqa_diamond_zeroshot`. The CoT variant uses generation + regex parsing instead of loglikelihood.

**Recommended: option 3** — least work, stays within lm_eval framework. The command would be:

```bash
~/inference/venv/bin/python3 -m lm_eval \
    --model local-chat-completions \
    --model_args "model=Qwen/Qwen3.5-122B-A10B,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,tokenized_requests=False,max_length=8192" \
    --tasks gpqa_diamond_cot_zeroshot \
    --num_fewshot 0 \
    --batch_size 1 \
    --output_path ~/inference/benchmarks/gpqa-diamond-cot \
    --log_samples
```

**Warning:** CoT tasks generate long reasoning chains — this will take hours, not minutes. But it's the correct methodology for instruction-tuned models.

### vLLM upstream PR (deferred)

Code is ready. PR would target `vllm-project/vllm` main. Key considerations:
- Rebase onto current main (we're on v0.17.1)
- RFC #30136 is deprecating legacy quant formats — our approach (extending existing `gptq_marlin`) aligns with consolidation direction
- The `__init__.py` and `model.py` deletions in our commit are just cleaning up old experiment artifacts — not needed for upstream
- File an issue first to socialize, or go straight to PR

### Systemd unit cleanup

`~/.config/systemd/user/vllm-inference-fp8hybrid.service` updated to `--quantization gptq_marlin` and `--max-model-len 262144`. The `quantize_config.json` in the hybrid checkpoint was also updated to `"quant_method": "gptq"` (standard, auto-promoted to gptq_marlin).

---

## Key bugs found and fixed this session

1. **`UnquantizedLinearMethod` FP8→BF16 cast** — FP8 tensors loaded but cast to BF16 at load time. Root cause: vLLM's one-quant-method-per-model assumption. Fix: per-layer dispatch via `Fp8LinearMethod`.

2. **HF↔vLLM prefix mismatch** — `fp8_layers` set had checkpoint names (`model.language_model.layers.N`), runtime uses mapped names (`language_model.model.layers.N`). 0/307 layers matched → 192/307 after fix. Fix: `apply_vllm_mapper()` on `fp8_layers`.

3. **Unnecessary deepcopy** — `get_linear_quant_method()` deepcopied entire config (including 307-entry `fp8_layers`) for every layer (~340 calls). Fix: defer deepcopy to GPTQ mutation path only.

4. **GPQA Diamond max_length** — `lm_eval` defaults to `max_length=2048`, but GPQA questions are ~2757 tokens. Fix: pass `max_length=8192` in model_args. (Underlying loglikelihood methodology issue remains.)

---

## Server state on Apollyon

- **Running:** fp8hybrid at 256k context, 21.5 tok/s, port 8000
- **Process:** launched via nohup, log at `/tmp/vllm-256k.log`
- **To restart:** `export PATH="$HOME/inference/bin:$PATH" && inference-serve fp8hybrid`

---

## File locations on Apollyon

| Path | Description |
|---|---|
| `~/inference/vllm-src/` | vLLM source, branch `feat/hybrid-gptq-fp8` |
| `~/inference/bin/build-hybrid-checkpoint.py` | Checkpoint builder (renamed from frankenstein.py) |
| `~/inference/bin/run-gpqa-diamond.sh` | GPQA Diamond benchmark script (max_length=8192) |
| `~/inference/bin/inference-bench` | Streaming throughput benchmark |
| `~/inference/models/hf/qwen3.5-122b-a10b-fp8hybrid/` | The hybrid checkpoint (70 GB) |
| `~/inference/benchmarks/gpqa-diamond/` | lm_eval results (truncated run — invalid) |
| `~/Development/vllm-hybrid-quant/` | Public repo (pushed) |
| `~/.config/systemd/user/vllm-inference-fp8hybrid.service` | Systemd unit (updated) |
