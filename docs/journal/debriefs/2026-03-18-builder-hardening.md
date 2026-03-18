# Debrief — Builder Hardening

## Task Summary

Hardened `build-hybrid-checkpoint.py` with early GPTQ input validation, safer output-directory handling, single-file checkpoint support, tensor shape validation during FP8 replacement, guarded handling for unmatched FP8 tensors, and a logging/type-annotation pass required by the task spec.

## Files Changed

- `build-hybrid-checkpoint.py` now validates inputs before download, refuses unsafe output reuse unless `--force` is set, supports single-file `model.safetensors` checkpoints, validates replacement shapes, warns or errors on unexpected unmatched FP8 tensors, and routes progress through `logging` with annotated/documented functions.
- `docs/journal/debriefs/2026-03-18-builder-hardening.md` records the requested task summary, file-level change notes, and the main implementation decisions from this task.

## Surprises Or Decisions

- `model.safetensors-*` shard names do not end with the plain `.safetensors` suffix, so the startup validation had to check for `*.safetensors*` and then separately enforce the supported `model.safetensors` or `model.safetensors-*` layouts.
- For unexpected unmatched FP8 tensors, the builder now warns and leaves them out of the output rather than appending dead tensors to the final shard; only expected `weight_scale_inv` tensors are auto-added.
