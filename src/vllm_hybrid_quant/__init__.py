# SPDX-License-Identifier: Apache-2.0
"""vllm-hybrid-quant: Per-layer mixed-precision quantization for vLLM."""

from .gptq_marlin_hybrid_fp8 import GPTQMarlinHybridFp8Config

__all__ = ["GPTQMarlinHybridFp8Config"]
