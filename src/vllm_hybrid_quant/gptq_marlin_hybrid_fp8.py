# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid GPTQ-Marlin + FP8 quantization config for vLLM.

Handles checkpoints where MoE expert weights use GPTQ-INT4 (Marlin kernels)
and dense layers (attention, shared experts) use calibrated FP8 block-128
weights from an official FP8 checkpoint.

Without this, vLLM's GPTQMarlinConfig routes non-GPTQ layers to
UnquantizedLinearMethod, which casts FP8 tensors to BF16 at load time —
same bandwidth, no speedup. This config routes those layers to
Fp8LinearMethod instead, keeping weights in FP8 and using FP8 tensor core
GEMMs for actual bandwidth savings.

Usage:
    # In the hybrid checkpoint's quantize_config.json:
    {
        "quant_method": "gptq_marlin_hybrid_fp8",
        "bits": 4,
        "group_size": 128,
        ...
        "fp8_config": {
            "weight_block_size": [128, 128],
            "activation_scheme": "dynamic"
        }
    }

    # vLLM launch:
    vllm serve model_path --quantization gptq_marlin_hybrid_fp8
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Known HF → vLLM prefix mappings.
#
# vLLM's WeightsMapper renames checkpoint prefixes during weight loading.
# We need both HF and vLLM naming conventions in the FP8 layer set so that
# prefix matching works in get_quant_method() (which receives vLLM prefixes).
#
# TODO(upstream): Instead of hardcoding these, obtain them from the model
# class's hf_to_vllm_mapper via apply_vllm_mapper(). This would make the
# config fully model-agnostic.
# ---------------------------------------------------------------------------
_HF_TO_VLLM_PREFIXES = {
    "model.visual.": "visual.",
    "lm_head.": "language_model.lm_head.",
    "model.language_model.": "language_model.model.",
}


def _build_fp8_layer_set(model_name: str,
                         revision: str | None = None) -> set[str]:
    """Scan the safetensors checkpoint metadata to find layers that have
    FP8 (float8_e4m3fn) weight tensors.

    Returns a set of layer prefixes in BOTH HF and vLLM naming conventions.
    """
    from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

    from vllm.transformers_utils.config import get_safetensors_params_metadata

    metadata = get_safetensors_params_metadata(model_name, revision=revision)
    fp8_layers: set[str] = set()
    for param_name, info in metadata.items():
        dtype_str = info.get("dtype", None)
        if (
            dtype_str
            and _SAFETENSORS_TO_TORCH_DTYPE.get(dtype_str)
            == torch.float8_e4m3fn
            and param_name.endswith(".weight")
        ):
            hf_prefix = param_name.rsplit(".", 1)[0]
            fp8_layers.add(hf_prefix)

            # Also add vLLM-mapped version
            for hf_pfx, vllm_pfx in _HF_TO_VLLM_PREFIXES.items():
                if hf_prefix.startswith(hf_pfx):
                    vllm_name = vllm_pfx + hf_prefix[len(hf_pfx):]
                    fp8_layers.add(vllm_name)
                    break

    return fp8_layers


def _resolve_fused_to_hf_names(
    vllm_prefix: str,
    packed_modules_mapping: dict[str, list[str]],
) -> list[str]:
    """Resolve a vLLM fused layer prefix to its constituent HF layer names.

    For example:
        "model.layers.0.self_attn.qkv_proj"
        → ["model.layers.0.self_attn.q_proj",
           "model.layers.0.self_attn.k_proj",
           "model.layers.0.self_attn.v_proj"]

    If the prefix doesn't match any fused mapping, returns [vllm_prefix].
    """
    parts = vllm_prefix.rsplit(".", 1)
    if len(parts) == 2:
        parent, module_name = parts
    else:
        return [vllm_prefix]

    if module_name in packed_modules_mapping:
        return [f"{parent}.{hf_name}"
                for hf_name in packed_modules_mapping[module_name]]

    return [vllm_prefix]


class GPTQMarlinHybridFp8Config(GPTQMarlinConfig):
    """Hybrid quantization: GPTQ-Marlin for MoE experts, FP8 for dense layers.

    Inherits all GPTQ-Marlin behavior for expert layers and MoE dispatch.
    Overrides only the linear layer routing to use Fp8LinearMethod for layers
    that were excluded from GPTQ quantization (attention, shared experts).
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
        dynamic: dict[str, dict[str, int | bool]],
        full_config: dict[str, Any],
        modules_in_block_to_quantize: list[str] | None = None,
        fp8_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            weight_bits=weight_bits,
            group_size=group_size,
            desc_act=desc_act,
            is_sym=is_sym,
            lm_head_quantized=lm_head_quantized,
            dynamic=dynamic,
            full_config=full_config,
            modules_in_block_to_quantize=modules_in_block_to_quantize,
        )
        self._fp8_config_dict = fp8_config or {}
        self._fp8_config: Fp8Config | None = None
        self._fp8_hf_layers: set[str] = set()
        self._model_name: str | None = None

    def maybe_update_config(
        self, model_name: str, revision: str | None = None
    ) -> None:
        """Called by vLLM after from_config() with the actual model path."""
        super().maybe_update_config(model_name, revision=revision)
        self._model_name = model_name
        self._fp8_hf_layers = _build_fp8_layer_set(model_name, revision)

        weight_block_size = self._fp8_config_dict.get(
            "weight_block_size", [128, 128]
        )
        activation_scheme = self._fp8_config_dict.get(
            "activation_scheme", "dynamic"
        )
        self._fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=activation_scheme,
            ignored_layers=None,
            weight_block_size=weight_block_size,
        )
        self._fp8_config.packed_modules_mapping = self.packed_modules_mapping

        logger.info(
            "GPTQMarlinHybridFp8Config: GPTQ-INT4 for experts, "
            "FP8 block-%s for dense layers (activation=%s), "
            "%d FP8 layers detected in checkpoint",
            weight_block_size, activation_scheme, len(self._fp8_hf_layers),
        )

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin_hybrid_fp8"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GPTQMarlinHybridFp8Config":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic
        return cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=cls.get_from_keys(config, ["group_size"]),
            desc_act=cls.get_from_keys(config, ["desc_act"]),
            is_sym=cls.get_from_keys(config, ["sym"]),
            lm_head_quantized=cls.get_from_keys_or(
                config, ["lm_head"], default=False
            ),
            dynamic=dynamic,
            full_config=config,
            modules_in_block_to_quantize=cls.get_from_keys_or(
                config, ["modules_in_block_to_quantize"], default=None
            ),
            fp8_config=cls.get_from_keys_or(
                config, ["fp8_config"], default=None
            ),
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        quant_method = hf_quant_cfg.get("quant_method", "").lower()
        if quant_method == "gptq_marlin_hybrid_fp8":
            return "gptq_marlin_hybrid_fp8"
        if user_quant == "gptq_marlin_hybrid_fp8":
            return "gptq_marlin_hybrid_fp8"
        return None

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    def _ensure_fp8_initialized(self) -> bool:
        """Lazily (re)build FP8 state if lost during pickling."""
        if self._fp8_config is not None and self._fp8_hf_layers:
            return True
        if not self._model_name:
            logger.warning(
                "GPTQMarlinHybridFp8Config: _model_name not set, "
                "cannot initialize FP8 layer set."
            )
            return False

        logger.debug(
            "GPTQMarlinHybridFp8Config: (re)initializing FP8 state "
            "for model '%s'", self._model_name,
        )
        self._fp8_hf_layers = _build_fp8_layer_set(self._model_name)
        weight_block_size = self._fp8_config_dict.get(
            "weight_block_size", [128, 128]
        )
        self._fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=self._fp8_config_dict.get(
                "activation_scheme", "dynamic"
            ),
            ignored_layers=None,
            weight_block_size=weight_block_size,
        )
        self._fp8_config.packed_modules_mapping = self.packed_modules_mapping
        logger.debug(
            "GPTQMarlinHybridFp8Config: FP8 state ready — "
            "%d FP8 layers, packed_mapping_keys=%s",
            len(self._fp8_hf_layers),
            list(self.packed_modules_mapping.keys()),
        )
        return True

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # MoE layers: delegate to parent (GPTQ Marlin MoE)
        if isinstance(layer, FusedMoE):
            return super().get_quant_method(layer, prefix)

        # Linear layers: route GPTQ vs FP8
        if isinstance(layer, LinearBase):
            from vllm.model_executor.layers.quantization.utils.gptq_utils import (
                is_layer_gptq_quantized,
            )

            is_gptq = is_layer_gptq_quantized(
                prefix=prefix,
                quantized_layers=self.modules_in_block_to_quantize,
                fused_mapping=self.packed_modules_mapping,
            )
            cloned_config = deepcopy(self)
            skip_by_dynamic = (
                get_dynamic_override(cloned_config, layer_name=prefix) is False
            )

            if is_gptq and not skip_by_dynamic:
                return super().get_quant_method(layer, prefix)

            # Check if it's an FP8 layer
            if self._ensure_fp8_initialized():
                hf_names = _resolve_fused_to_hf_names(
                    prefix, self.packed_modules_mapping
                )
                if all(n in self._fp8_hf_layers for n in hf_names):
                    fp8_method = Fp8LinearMethod(self._fp8_config)
                    fp8_method.marlin_input_dtype = get_marlin_input_dtype(
                        prefix
                    )
                    logger.debug(
                        "Hybrid: routing '%s' → Fp8LinearMethod "
                        "(block_quant=%s, hf_names=%s)",
                        prefix, fp8_method.block_quant, hf_names,
                    )
                    return fp8_method

            return UnquantizedLinearMethod()

        # Non-linear layers: let parent handle
        return super().get_quant_method(layer, prefix)

    def __repr__(self) -> str:
        fp8_bs = (self._fp8_config.weight_block_size
                  if self._fp8_config else "not yet initialized")
        return (
            f"GPTQMarlinHybridFp8Config("
            f"quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"fp8_block_size={fp8_bs}, "
            f"dynamic={self.dynamic})"
        )
