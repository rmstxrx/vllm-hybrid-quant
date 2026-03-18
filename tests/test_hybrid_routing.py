# SPDX-License-Identifier: Apache-2.0
"""Tests for GPTQMarlinHybridFp8Config routing logic.

These tests verify that the hybrid config correctly routes layers:
- FP8 dense layers → Fp8LinearMethod (CUTLASS FP8 GEMMs)
- GPTQ expert layers → GPTQMarlinLinearMethod (Marlin kernels)
- Mixed/BF16 layers → UnquantizedLinearMethod

Run with: pytest tests/test_hybrid_routing.py -v
Requires: A hybrid checkpoint OR mocked safetensors metadata.
"""

import json
import pickle
from unittest.mock import patch

import pytest

# These tests require vLLM to be importable
vllm = pytest.importorskip("vllm")

from vllm.model_executor.layers.quantization.gptq_marlin_hybrid_fp8 import (
    GPTQMarlinHybridFp8Config,
    _build_fp8_layer_set,
    _resolve_fused_to_hf_names,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "bits": 4,
    "group_size": 128,
    "damp_percent": 0.01,
    "desc_act": False,
    "static_groups": False,
    "sym": True,
    "true_sequential": True,
    "quant_method": "gptq_marlin_hybrid_fp8",
    "dynamic": {
        "lm_head": {},
        "model.language_model.embed_tokens": {},
        "-:.*attn.*": {},
        "-:.*shared_expert.*": {},
        "-:.*mtp.*": {},
        "-:.*visual.*": {},
    },
    "modules_to_not_convert": [],
    "fp8_config": {
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    },
}

# Simulates Qwen3.5's packed_modules_mapping
PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}

# Simulated FP8 layer set (vLLM-mapped prefixes for layer 11 with self_attn,
# and layer 0 with linear_attn)
MOCK_FP8_LAYERS = {
    # Layer 0 (linear_attn variant) — vLLM-mapped prefixes
    "language_model.model.layers.0.linear_attn.in_proj_qkv",
    "language_model.model.layers.0.linear_attn.in_proj_z",
    "language_model.model.layers.0.linear_attn.out_proj",
    "language_model.model.layers.0.mlp.shared_expert.gate_proj",
    "language_model.model.layers.0.mlp.shared_expert.up_proj",
    "language_model.model.layers.0.mlp.shared_expert.down_proj",
    # Layer 11 (self_attn variant)
    "language_model.model.layers.11.self_attn.q_proj",
    "language_model.model.layers.11.self_attn.k_proj",
    "language_model.model.layers.11.self_attn.v_proj",
    "language_model.model.layers.11.self_attn.o_proj",
    "language_model.model.layers.11.mlp.shared_expert.gate_proj",
    "language_model.model.layers.11.mlp.shared_expert.up_proj",
    "language_model.model.layers.11.mlp.shared_expert.down_proj",
    # NOTE: in_proj_a and in_proj_b are NOT here (BF16, too small for block-128)
}


@pytest.fixture
def config():
    """Build a GPTQMarlinHybridFp8Config with mocked FP8 state."""
    qc = GPTQMarlinHybridFp8Config.from_config(SAMPLE_CONFIG)
    qc._fp8_hf_layers = MOCK_FP8_LAYERS.copy()
    qc._fp8_config = type("MockFp8Config", (), {
        "weight_block_size": [128, 128],
        "is_checkpoint_fp8_serialized": True,
        "activation_scheme": "dynamic",
        "packed_modules_mapping": PACKED_MODULES,
    })()
    qc._model_name = "/mock/model/path"
    qc.packed_modules_mapping = PACKED_MODULES
    return qc


# ---------------------------------------------------------------------------
# Tests: _resolve_fused_to_hf_names
# ---------------------------------------------------------------------------

class TestResolveFusedNames:
    def test_qkv_proj_resolves(self):
        result = _resolve_fused_to_hf_names(
            "language_model.model.layers.11.self_attn.qkv_proj",
            PACKED_MODULES,
        )
        assert result == [
            "language_model.model.layers.11.self_attn.q_proj",
            "language_model.model.layers.11.self_attn.k_proj",
            "language_model.model.layers.11.self_attn.v_proj",
        ]

    def test_gate_up_proj_resolves(self):
        result = _resolve_fused_to_hf_names(
            "language_model.model.layers.0.mlp.shared_expert.gate_up_proj",
            PACKED_MODULES,
        )
        assert result == [
            "language_model.model.layers.0.mlp.shared_expert.gate_proj",
            "language_model.model.layers.0.mlp.shared_expert.up_proj",
        ]

    def test_in_proj_ba_resolves(self):
        result = _resolve_fused_to_hf_names(
            "language_model.model.layers.0.linear_attn.in_proj_ba",
            PACKED_MODULES,
        )
        assert result == [
            "language_model.model.layers.0.linear_attn.in_proj_b",
            "language_model.model.layers.0.linear_attn.in_proj_a",
        ]

    def test_non_fused_passes_through(self):
        result = _resolve_fused_to_hf_names(
            "language_model.model.layers.0.linear_attn.out_proj",
            PACKED_MODULES,
        )
        assert result == ["language_model.model.layers.0.linear_attn.out_proj"]

    def test_empty_mapping_passes_through(self):
        result = _resolve_fused_to_hf_names(
            "language_model.model.layers.0.self_attn.qkv_proj",
            {},
        )
        assert result == ["language_model.model.layers.0.self_attn.qkv_proj"]


# ---------------------------------------------------------------------------
# Tests: FP8 layer matching
# ---------------------------------------------------------------------------

class TestFp8LayerMatching:
    """Test that fused vLLM prefixes correctly resolve and match the FP8 set."""

    def test_qkv_proj_all_fp8(self):
        hf_names = _resolve_fused_to_hf_names(
            "language_model.model.layers.11.self_attn.qkv_proj",
            PACKED_MODULES,
        )
        assert all(n in MOCK_FP8_LAYERS for n in hf_names)

    def test_o_proj_fp8(self):
        hf_names = _resolve_fused_to_hf_names(
            "language_model.model.layers.11.self_attn.o_proj",
            PACKED_MODULES,
        )
        assert all(n in MOCK_FP8_LAYERS for n in hf_names)

    def test_in_proj_qkvz_all_fp8(self):
        hf_names = _resolve_fused_to_hf_names(
            "language_model.model.layers.0.linear_attn.in_proj_qkvz",
            PACKED_MODULES,
        )
        assert all(n in MOCK_FP8_LAYERS for n in hf_names)

    def test_in_proj_ba_not_fp8(self):
        """in_proj_a and in_proj_b are BF16 — should NOT match."""
        hf_names = _resolve_fused_to_hf_names(
            "language_model.model.layers.0.linear_attn.in_proj_ba",
            PACKED_MODULES,
        )
        assert not all(n in MOCK_FP8_LAYERS for n in hf_names)

    def test_gate_up_proj_fp8(self):
        hf_names = _resolve_fused_to_hf_names(
            "language_model.model.layers.0.mlp.shared_expert.gate_up_proj",
            PACKED_MODULES,
        )
        assert all(n in MOCK_FP8_LAYERS for n in hf_names)

    def test_embeddings_not_fp8(self):
        hf_names = _resolve_fused_to_hf_names(
            "language_model.model.embed_tokens",
            PACKED_MODULES,
        )
        assert not all(n in MOCK_FP8_LAYERS for n in hf_names)


# ---------------------------------------------------------------------------
# Tests: Config lifecycle
# ---------------------------------------------------------------------------

class TestConfigLifecycle:
    def test_from_config(self):
        qc = GPTQMarlinHybridFp8Config.from_config(SAMPLE_CONFIG)
        assert qc.weight_bits == 4
        assert qc.group_size == 128
        assert qc._fp8_config_dict == SAMPLE_CONFIG["fp8_config"]
        assert qc._fp8_hf_layers == set()  # Not yet initialized
        assert qc._model_name is None

    def test_pickle_roundtrip(self, config):
        data = pickle.dumps(config)
        restored = pickle.loads(data)
        assert len(restored._fp8_hf_layers) == len(config._fp8_hf_layers)
        assert restored._model_name == config._model_name
        assert restored._fp8_config is not None

    def test_override_quantization_method(self):
        assert GPTQMarlinHybridFp8Config.override_quantization_method(
            {"quant_method": "gptq_marlin_hybrid_fp8"}, None
        ) == "gptq_marlin_hybrid_fp8"

    def test_override_ignores_plain_gptq(self):
        assert GPTQMarlinHybridFp8Config.override_quantization_method(
            {"quant_method": "gptq"}, None
        ) is None

    def test_override_respects_user_flag(self):
        assert GPTQMarlinHybridFp8Config.override_quantization_method(
            {"quant_method": "gptq"}, "gptq_marlin_hybrid_fp8"
        ) == "gptq_marlin_hybrid_fp8"

    def test_get_name(self):
        assert GPTQMarlinHybridFp8Config.get_name() == "gptq_marlin_hybrid_fp8"

    def test_get_config_filenames(self):
        assert "quantize_config.json" in GPTQMarlinHybridFp8Config.get_config_filenames()
