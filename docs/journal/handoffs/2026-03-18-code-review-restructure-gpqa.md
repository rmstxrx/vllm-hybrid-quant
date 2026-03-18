# Handoff — Code Review, Repo Restructure, GPQA Benchmark

**Date:** 2026-03-18
**Author:** Claude (Opus 4.6) + Rômulo
**Status:** GPQA Diamond benchmark running in tmux on Apollyon (~2.5h remaining)

---

## What was accomplished

### Three-way code review (complete)

Three agents reviewed `vllm-hybrid-quant`: Claude (Opus 4.6), Gemini, GPT-5.4-Pro. GPT produced the strongest review — found two bugs the others missed. Cross-review reports at `/mnt/user-data/outputs/` on the Claude.ai sandbox (not on Apollyon).

**Bugs found and fixed:**
1. **Stale patch hunk (P0, GPT):** Diff minus-side had `float8_e4m3fn` already in `unquant_dtypes`, but pristine v0.17.1 doesn't. `git apply` would fail on clean checkout.
2. **`modules_in_block_to_quantize` early-return (P1, GPT):** `maybe_update_config()` returned before FP8 detection for checkpoints with pre-set module lists. Fixed by extracting `_detect_fp8_layers()` and calling from both paths.
3. **Substring matching false-positive (P1, Claude):** `_is_layer_fp8()` used `any(fp8_layer in prefix ...)` — `"gate"` matched `"gate_proj"`. Fixed with exact set membership.
4. **Block-size nondeterminism (P2, Claude+GPT):** `next(iter(set))` picked arbitrary layer for block-size inference. Fixed by validating all layers share uniform block size.

**Gemini phantom bugs (verified false):**
- Layer 1/11 collision — doesn't exist in Python (`"layers.1.mlp" in "layers.11.mlp"` → False)
- Missing activation scales — wrong for `activation_scheme="dynamic"`
- Deepcopy mutability bleeding — `Fp8LinearMethod` receives `Fp8Config`, not `GPTQMarlinConfig`

### vLLM fork created (complete, pushed)

`https://github.com/rmstxrx/vllm` branch `v0.17.1-hybrid-fp8` — commit `740433a`

- Created from pristine v0.17.1 tag (no local patches carried forward)
- All 4 bug fixes applied
- 19 tests (16 original + 3 new: substring regression, early-return path, heterogeneous block-size rejection)
- GitHub Actions DISABLED on fork (avoid billing overrun)
- GPT-5.4-xhigh (Codex) wrote the code; dispatched via Maestro

**Unrequested but valid change by GPT:** Added `marlin_input_dtype` attribute guard in `get_quant_method()` to prevent dangling attribute on `Fp8LinearMethod`. Also added `config.get_name() == "gptq_marlin"` guard on the FP8 dispatch path.

### Standalone repo restructured (complete, pushed)

`https://github.com/rmstxrx/vllm-hybrid-quant` — 2 commits:
- `566905e` — Remove `vllm-patch/` and `tests/` (now in fork), clean 5 stale "frankenstein" refs, rewrite README to point to fork, add Known Limitations section, clarify 307→192 layer count
- `bcdad3f` — Builder hardening: unplaced tensor validation, output dir safety (`--force`), shape validation, single-file checkpoint support, early input validation, `print()` → `logging` migration, type annotations

### Forum post updated (complete)

`forums.developer.nvidia.com/t/.../363941` — edited with: ~100→~150 lines, 16→19 tests, "Apply the vLLM patch" → clone-the-fork instructions, repo description updated, edit log appended.

### GPQA Diamond benchmark (IN PROGRESS)

Running in tmux session `gpqa` on Apollyon. Two panes: left = lm_eval, right = server log.

**Run configuration (v4 — the correct one):**
- Server: `--default-chat-template-kwargs '{"enable_thinking": false}'` — no reasoning parser, no tool parser
- lm_eval: `--model local-chat-completions --apply_chat_template --tasks gpqa_diamond_cot_zeroshot --max_gen_toks=4096`
- Script: `/tmp/gpqa-full-run.sh`, log: `/tmp/gpqa-pipeline-v4.log`, server log: `/tmp/vllm-gpqa-run.log`
- Results will land in: `~/inference/benchmarks/gpqa-diamond-cot-hybrid-v4/`

**Previous failed runs and why:**
- v1: `--reasoning-parser qwen3` caused model output to go into `reasoning_content`, lm_eval read `content` → all null → 0%
- v2: Removed reasoning parser but default `max_gen_toks=2047` truncated CoT mid-sentence → 11.1% flexible / 0% strict
- v3: Added `max_gen_toks=4096` but server still lacked `--default-chat-template-kwargs`, model generated endless thinking → 190s/question → server killed when setsid wrapper exited
- v4 (current): Thinking disabled, max_gen_toks=4096, tmux-isolated. Running at ~60s/question average, 21.5 tok/s confirmed.

---

## What needs work when benchmark completes

### 1. Read the results

```bash
tmux attach -t gpqa
# Or check directly:
tail -20 /tmp/gpqa-pipeline-v4.log
# Look for exact_match scores in flexible-extract and strict-match
```

Results dir: `~/inference/benchmarks/gpqa-diamond-cot-hybrid-v4/`

### 2. Interpret the score

The forum comparison point is **84.85%** on `Intel/Qwen3.5-122B-A10B-int4-AutoRound` (trystan1, likely custom eval with thinking enabled). Our hybrid checkpoint uses FP8 for dense layers (higher precision than INT4), so we'd expect similar or better quality — but thinking is disabled for this run, which will lower the score vs thinking-enabled evals.

If the score is reasonable (>50%), post it on the forum thread as a reply to AoE's quality question.

If the score is poor (<40%), it may be because:
- Thinking disabled hurts CoT quality (consider re-running with thinking enabled + a custom eval script that parses reasoning_content)
- The `gpqa_diamond_cot_zeroshot` task prompt ("Let's think step by step:") may not be ideal for this model without thinking mode
- Consider running `gpqa_diamond_zeroshot` (no CoT) as a simpler baseline

### 3. Re-enable the systemd service

The `vllm-inference-fp8hybrid.service` was disabled during this session to prevent auto-restart interfering with benchmark launches. Re-enable after benchmark:

```bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
systemctl --user enable vllm-inference-fp8hybrid.service
systemctl --user start vllm-inference-fp8hybrid.service
```

**WARNING:** The service file may still have `--reasoning-parser qwen3`. Update it if you want thinking disabled for general use.

### 4. Upstream PR (deferred)

All three reviewers agree the approach needs a `HybridQuantizationConfig` or similar abstraction for upstream acceptance. File an issue on `vllm-project/vllm` first to socialize. The fork branch is ready as a proof-of-concept reference.

---

## Spark unified memory lessons learned

These bit us repeatedly during this session:

1. **Page cache IS GPU memory.** After killing vLLM, the model shards stay in page cache. `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` is mandatory before relaunching.

2. **`VLLM::EngineCore` renames itself.** `pkill -f "vllm"` doesn't match it. Must also `pkill -f "VLLM::EngineCore"` or kill by PID.

3. **`setsid` + `nohup` inside Maestro exec doesn't fully detach.** The server died when the wrapper script's process tree was cleaned up. **Use tmux** for long-running processes.

4. **Memory check before launch.** Need ~102 GiB free for 0.85 GPU utilization on the 128 GB Spark. The benchmark script should verify `MemAvailable > 100G` before attempting server launch.

---

## File locations on Apollyon

| Path | Description |
|---|---|
| `~/inference/vllm-src/` | vLLM fork, branch `v0.17.1-hybrid-fp8` (commit `740433a`) |
| `~/Development/vllm-hybrid-quant/` | Standalone repo (builder + docs) |
| `~/inference/models/hf/qwen3.5-122b-a10b-fp8hybrid/` | The hybrid checkpoint (70 GB) |
| `~/inference/bin/run-gpqa-benchmark.sh` | Benchmark launcher (v3 — partially outdated, see notes) |
| `/tmp/gpqa-full-run.sh` | Current benchmark script (v4 — the correct one) |
| `/tmp/gpqa-pipeline-v4.log` | Current benchmark output |
| `/tmp/vllm-gpqa-run.log` | Current server log |
| `~/inference/benchmarks/gpqa-diamond-cot-hybrid-v4/` | Results destination |
| `~/inference/benchmarks/gpqa-diamond-cot-hybrid/` | v1 results (all null — invalid) |
| `~/inference/benchmarks/gpqa-diamond-cot-hybrid-v2/` | v2 results (truncated — may not exist, run was killed early) |
| `~/.config/systemd/user/vllm-inference-fp8hybrid.service` | Systemd unit (DISABLED) |

## GitHub state

| Repo | Branch | Last commit | Pushed |
|---|---|---|---|
| `rmstxrx/vllm` | `v0.17.1-hybrid-fp8` | `740433a` | Yes |
| `rmstxrx/vllm-hybrid-quant` | `main` | `bcdad3f` | Yes |
| `rmstxrx/vllm` | Actions | Disabled | — |
