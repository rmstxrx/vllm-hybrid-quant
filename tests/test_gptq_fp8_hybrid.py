# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for hybrid GPTQ-INT4 + FP8 per-layer quantization dispatch.

When a GPTQ checkpoint contains non-GPTQ layers stored as FP8
(float8_e4m3fn) with block-scale tensors (weight_scale_inv),
those layers should be routed to Fp8LinearMethod instead of
UnquantizedLinearMethod.

Run `pytest tests/quantization/test_gptq_fp8_hybrid.py`.
"""

from unittest.mock import patch

import pytest

from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    _is_layer_fp8,
)
from vllm.model_executor.models.utils import WeightsMapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_gptq_config(**overrides) -> GPTQMarlinConfig:
    """Create a minimal GPTQMarlinConfig for testing."""
    defaults = dict(
        weight_bits=4,
        group_size=128,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={
            "-:.*attn.*": {},
            "-:.*shared_expert.*": {},
        },
        full_config={
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
        },
    )
    defaults.update(overrides)
    return GPTQMarlinConfig(**defaults)


# Simulated safetensors metadata for a tiny 2-layer MoE model with
# FP8 attention + shared experts and GPTQ INT4 MoE experts.
MOCK_METADATA = {
    # Layer 0 attention — FP8 weights + block scales
    "model.layers.0.self_attn.q_proj.weight": {
        "dtype": "F8_E4M3",
        "shape": [1024, 512],
    },
    "model.layers.0.self_attn.q_proj.weight_scale_inv": {
        "dtype": "BF16",
        "shape": [8, 4],  # block_size = [128, 128]
    },
    "model.layers.0.self_attn.k_proj.weight": {
        "dtype": "F8_E4M3",
        "shape": [256, 512],
    },
    "model.layers.0.self_attn.k_proj.weight_scale_inv": {
        "dtype": "BF16",
        "shape": [2, 4],
    },
    "model.layers.0.self_attn.v_proj.weight": {
        "dtype": "F8_E4M3",
        "shape": [256, 512],
    },
    "model.layers.0.self_attn.v_proj.weight_scale_inv": {
        "dtype": "BF16",
        "shape": [2, 4],
    },
    "model.layers.0.self_attn.o_proj.weight": {
        "dtype": "F8_E4M3",
        "shape": [512, 1024],
    },
    "model.layers.0.self_attn.o_proj.weight_scale_inv": {
        "dtype": "BF16",
        "shape": [4, 8],
    },
    # Layer 0 shared expert — FP8
    "model.layers.0.mlp.shared_expert.gate_proj.weight": {
        "dtype": "F8_E4M3",
        "shape": [2048, 512],
    },
    "model.layers.0.mlp.shared_expert.gate_proj.weight_scale_inv": {
        "dtype": "BF16",
        "shape": [16, 4],
    },
    "model.layers.0.mlp.shared_expert.up_proj.weight": {
        "dtype": "F8_E4M3",
        "shape": [2048, 512],
    },
    "model.layers.0.mlp.shared_expert.up_proj.weight_scale_inv": {
        "dtype": "BF16",
        "shape": [16, 4],
    },
    "model.layers.0.mlp.shared_expert.down_proj.weight": {
        "dtype": "F8_E4M3",
        "shape": [512, 2048],
    },
    "model.layers.0.mlp.shared_expert.down_proj.weight_scale_inv": {
        "dtype": "BF16",
        "shape": [4, 16],
    },
    # Layer 0 MoE expert — GPTQ INT4 (not FP8, not in unquant_dtypes)
    "model.layers.0.mlp.experts.0.gate_proj.qweight": {
        "dtype": "I32",
        "shape": [512, 256],
    },
    "model.layers.0.mlp.experts.0.gate_proj.qzeros": {
        "dtype": "I32",
        "shape": [4, 256],
    },
    # Layer 0 gate — BF16 (small, not quantized)
    "model.layers.0.mlp.gate.weight": {
        "dtype": "BF16",
        "shape": [8, 512],
    },
    # Norms — BF16
    "model.layers.0.input_layernorm.weight": {
        "dtype": "BF16",
        "shape": [512],
    },
}


# ---------------------------------------------------------------------------
# Unit tests: _is_layer_fp8
# ---------------------------------------------------------------------------

class TestIsLayerFp8:
    """Test the _is_layer_fp8 helper that handles fused module matching."""

    FP8_LAYERS = {
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.shared_expert.gate_proj",
        "layers.0.mlp.shared_expert.up_proj",
        "layers.0.mlp.shared_expert.down_proj",
    }

    PACKED = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def test_direct_match(self):
        assert _is_layer_fp8(
            "layers.0.self_attn.o_proj", self.FP8_LAYERS, self.PACKED
        )

    def test_fused_qkv_match(self):
        """qkv_proj should match when all 3 shards are FP8."""
        assert _is_layer_fp8(
            "layers.0.self_attn.qkv_proj", self.FP8_LAYERS, self.PACKED
        )

    def test_fused_gate_up_match(self):
        """gate_up_proj should match when both shards are FP8."""
        assert _is_layer_fp8(
            "layers.0.mlp.shared_expert.gate_up_proj",
            self.FP8_LAYERS,
            self.PACKED,
        )

    def test_non_fp8_layer(self):
        """Gate layer is not FP8."""
        assert not _is_layer_fp8(
            "layers.0.mlp.gate", self.FP8_LAYERS, self.PACKED
        )

    def test_empty_fp8_layers(self):
        assert not _is_layer_fp8(
            "layers.0.self_attn.o_proj", set(), self.PACKED
        )

    def test_partial_fused_no_match(self):
        """If only some shards of a fused layer are FP8, no match."""
        partial = {
            "layers.0.self_attn.q_proj",
            "layers.0.self_attn.k_proj",
            # v_proj missing
        }
        assert not _is_layer_fp8(
            "layers.0.self_attn.qkv_proj", partial, self.PACKED
        )

    def test_no_fused_mapping(self):
        """Without fused mapping, direct substring match only."""
        assert _is_layer_fp8(
            "layers.0.self_attn.q_proj", self.FP8_LAYERS, {}
        )
        assert not _is_layer_fp8(
            "layers.0.self_attn.qkv_proj", self.FP8_LAYERS, {}
        )


# ---------------------------------------------------------------------------
# Unit tests: FP8 detection in maybe_update_config
# ---------------------------------------------------------------------------

class TestFp8Detection:
    """Test that GPTQMarlinConfig detects FP8 layers from metadata."""

    def test_detect_fp8_layers(self):
        """FP8 layers with weight_scale_inv should be detected."""
        config = _make_gptq_config()
        with patch(
            "vllm.model_executor.layers.quantization.gptq_marlin"
            ".get_safetensors_params_metadata",
            return_value=MOCK_METADATA,
        ):
            config.maybe_update_config("fake/model")

        assert config.fp8_config is not None
        assert config.fp8_config.weight_block_size == [128, 128]
        assert config.fp8_config.is_checkpoint_fp8_serialized is True
        assert config.fp8_config.activation_scheme == "dynamic"
        assert len(config.fp8_layers) == 7  # 4 attn + 3 shared_expert

    def test_no_fp8_in_pure_gptq(self):
        """Pure GPTQ checkpoint without FP8 tensors."""
        pure_gptq_metadata = {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16",
                "shape": [1024, 512],
            },
            "model.layers.0.mlp.experts.0.gate_proj.qweight": {
                "dtype": "I32",
                "shape": [512, 256],
            },
        }
        config = _make_gptq_config()
        with patch(
            "vllm.model_executor.layers.quantization.gptq_marlin"
            ".get_safetensors_params_metadata",
            return_value=pure_gptq_metadata,
        ):
            config.maybe_update_config("fake/model")

        assert config.fp8_config is None
        assert len(config.fp8_layers) == 0

    def test_fp8_without_scale_not_detected(self):
        """FP8 weight without weight_scale_inv should NOT be detected."""
        no_scale_metadata = {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F8_E4M3",
                "shape": [1024, 512],
            },
            # No weight_scale_inv
        }
        config = _make_gptq_config()
        with patch(
            "vllm.model_executor.layers.quantization.gptq_marlin"
            ".get_safetensors_params_metadata",
            return_value=no_scale_metadata,
        ):
            config.maybe_update_config("fake/model")

        assert config.fp8_config is None

    def test_block_size_inference(self):
        """Block size should be inferred from weight/scale shape ratio."""
        block_size = GPTQMarlinConfig._infer_fp8_block_size(
            MOCK_METADATA, "model.layers.0.self_attn.q_proj"
        )
        assert block_size == [128, 128]


# ---------------------------------------------------------------------------
# Unit tests: WeightsMapper integration
# ---------------------------------------------------------------------------

class TestWeightsMapperIntegration:
    """Test that fp8_layers survives the HF-to-vLLM name mapping."""

    def test_apply_vllm_mapper(self):
        config = _make_gptq_config()
        config.fp8_layers = {
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.k_proj",
        }
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.language_model.": "language_model.model.",
            }
        )
        config.apply_vllm_mapper(mapper)

        assert "language_model.model.layers.0.self_attn.q_proj" in config.fp8_layers
        assert "language_model.model.layers.0.self_attn.k_proj" in config.fp8_layers
        # Old names should be gone
        assert "model.language_model.layers.0.self_attn.q_proj" not in config.fp8_layers

    def test_apply_vllm_mapper_empty(self):
        """Empty fp8_layers should not error."""
        config = _make_gptq_config()
        mapper = WeightsMapper(
            orig_to_new_prefix={"model.": "language_model.model."}
        )
        config.apply_vllm_mapper(mapper)
        assert len(config.fp8_layers) == 0


# ---------------------------------------------------------------------------
# Unit tests: Fp8Config pickle round-trip (V1 engine uses multiprocessing)
# ---------------------------------------------------------------------------

class TestPickleSurvival:
    """Ensure hybrid config attributes survive pickle for V1 engine."""

    def test_pickle_round_trip(self):
        import pickle

        config = _make_gptq_config()
        config.fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        config.fp8_layers = {"layers.0.attn.q_proj", "layers.0.attn.k_proj"}

        data = pickle.dumps(config)
        restored = pickle.loads(data)

        assert restored.fp8_config is not None
        assert restored.fp8_config.weight_block_size == [128, 128]
        assert len(restored.fp8_layers) == 2


# ---------------------------------------------------------------------------
# Unit tests: deepcopy deferral
# ---------------------------------------------------------------------------

class TestDeepcopyDeferral:
    """Verify that get_linear_quant_method avoids deepcopy for non-GPTQ layers.

    The deepcopy only exists for the override_config() mutation path.
    Non-GPTQ layers (FP8 or unquantized) should never trigger it.
    """

    def test_fp8_path_does_not_deepcopy(self):
        """FP8 dispatch should use the original config, not a clone."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
        )
        from vllm.model_executor.layers.quantization.gptq_marlin import (
            GPTQMarlinLinearMethod,
        )
        from vllm.model_executor.layers.quantization.utils.gptq_utils import (
            get_linear_quant_method,
        )

        config = _make_gptq_config()
        config.fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        config.fp8_layers = {"layers.0.self_attn.o_proj"}
        config.modules_in_block_to_quantize = ["layers.0.mlp.experts"]
        config.packed_modules_mapping = {}

        layer = ColumnParallelLinear.__new__(ColumnParallelLinear)

        deepcopy_called = []
        original_deepcopy = __import__("copy").deepcopy

        def tracking_deepcopy(obj, memo=None):
            if isinstance(obj, GPTQMarlinConfig):
                deepcopy_called.append(True)
            return original_deepcopy(obj, memo)

        # Mock Fp8LinearMethod to avoid needing a full vLLM config context
        mock_fp8 = MagicMock(spec=Fp8LinearMethod)

        with (
            mock_patch(
                "vllm.model_executor.layers.quantization.utils"
                ".gptq_utils.deepcopy",
                side_effect=tracking_deepcopy,
            ),
            mock_patch(
                "vllm.model_executor.layers.quantization.utils"
                ".gptq_utils.Fp8LinearMethod",
                return_value=mock_fp8,
            ),
        ):
            result = get_linear_quant_method(
                config, layer, "layers.0.self_attn.o_proj",
                GPTQMarlinLinearMethod,
            )

        assert result is mock_fp8
        assert len(deepcopy_called) == 0, (
            "deepcopy should not be called for FP8 path"
        )

    def test_gptq_path_still_deepcopies(self):
        """GPTQ layers with dynamic overrides must still deepcopy."""
        from unittest.mock import patch as mock_patch

        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
        )
        from vllm.model_executor.layers.quantization.gptq_marlin import (
            GPTQMarlinLinearMethod,
        )
        from vllm.model_executor.layers.quantization.utils.gptq_utils import (
            get_linear_quant_method,
        )

        config = _make_gptq_config(dynamic={})  # No negative matches
        config.modules_in_block_to_quantize = [
            "layers.0.mlp.experts.0.gate_proj"
        ]
        config.packed_modules_mapping = {}

        layer = ColumnParallelLinear.__new__(ColumnParallelLinear)

        deepcopy_called = []
        original_deepcopy = __import__("copy").deepcopy

        def tracking_deepcopy(obj, memo=None):
            if isinstance(obj, GPTQMarlinConfig):
                deepcopy_called.append(True)
            return original_deepcopy(obj, memo)

        with mock_patch(
            "vllm.model_executor.layers.quantization.utils.gptq_utils.deepcopy",
            side_effect=tracking_deepcopy,
        ):
            result = get_linear_quant_method(
                config, layer, "layers.0.mlp.experts.0.gate_proj",
                GPTQMarlinLinearMethod,
            )

        assert isinstance(result, GPTQMarlinLinearMethod)
        assert len(deepcopy_called) == 1, "deepcopy required for GPTQ override path"
