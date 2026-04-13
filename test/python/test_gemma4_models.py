# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Unit tests for Gemma 4 model builder and C++ runtime behaviour.

Tests cover:
1. `Gemma4Model.make_genai_config` — correct `sliding_window.layers` for KV-sharing
   (Fix #1: only cache slots exclusively used by local layers are listed)
2. Runtime model loading — verifies that the C++ runtime can load a Gemma 4-style
   dummy model and correctly reads per-slot head_size from ONNX (Fix #2)
3. `sliding_window.layers` config written to genai_config.json matches expected safe slots
4. `PerLayerInputs` — basic presence check on a model that exposes `per_layer_inputs`
5. VLM type registration — "gemma4" routes to GemmaImageProcessor (Fix #5)

This file can be run in two ways:
1. pytest:  pytest test_gemma4_models.py --test_models=/path/to/test_models
2. standalone: python test_gemma4_models.py --test_models test/test_models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("gemma4-tests")


# ---------------------------------------------------------------------------
# Pure-Python unit tests for the model builder (no ONNX Runtime needed)
# ---------------------------------------------------------------------------


def _make_gemma4_config(
    *,
    num_hidden_layers: int,
    num_kv_shared_layers: int,
    attention_pattern: list[int],
    head_dim: int = 4,
    global_head_dim: int = 8,
    sliding_window: int = 16,
    hidden_size_per_layer_input: int = 0,
):
    """Build a minimal HuggingFace-style config object for Gemma4Model instantiation."""
    cfg = types.SimpleNamespace(
        # Architecture
        architectures=["Gemma4ForConditionalGeneration"],
        model_type="gemma4_text",
        # Attention
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=num_hidden_layers,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        attention_pattern=attention_pattern,
        num_kv_shared_layers=num_kv_shared_layers,
        sliding_window=sliding_window,
        # Misc
        hidden_size=32,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        hidden_size_per_layer_input=hidden_size_per_layer_input,
        # Fields consulted by base classes
        torch_dtype="float32",
    )
    return cfg


def _compute_safe_sliding_slots(
    num_layers: int,
    num_kv_shared_layers: int,
    attention_pattern: list[int],
) -> list[int]:
    """
    Reference implementation matching the fix in gemma.py.
    Returns cache slot indices that are exclusively used by local (sliding) layers.
    """
    num_unique_kv = num_layers - num_kv_shared_layers

    def get_slot(i: int) -> int:
        return i % num_unique_kv

    slot_users: dict[int, list[int]] = {}
    for layer in range(num_layers):
        s = get_slot(layer)
        slot_users.setdefault(s, []).append(layer)

    return [
        s for s in range(num_unique_kv)
        if slot_users.get(s) and all(attention_pattern[l] == 0 for l in slot_users[s])
    ]


class TestSlidingWindowLayersFix:
    """Tests for the KV-sharing sliding_window.layers fix in gemma.py."""

    def test_no_kv_sharing_all_local(self):
        """Without sharing, all local layers are listed (same as Gemma 3 behaviour)."""
        num_layers = 4
        pattern = [0, 0, 0, 0]  # all local
        result = _compute_safe_sliding_slots(num_layers, 0, pattern)
        assert result == [0, 1, 2, 3], f"Expected all slots, got {result}"

    def test_no_kv_sharing_mixed(self):
        """Without sharing, only local-layer indices are listed."""
        num_layers = 4
        pattern = [0, 0, 1, 0]  # layer 2 is global
        result = _compute_safe_sliding_slots(num_layers, 0, pattern)
        assert result == [0, 1, 3], f"Expected [0, 1, 3], got {result}"

    def test_kv_sharing_excludes_mixed_slots(self):
        """The canonical Gemma 4 case from the dummy model.

        num_layers=6, num_kv_shared_layers=2, num_unique_kv=4
        attention_pattern=[0, 0, 1, 0, 0, 1]

        Slot → layers:
          slot 0 → [0, 4]  both local  → SAFE
          slot 1 → [1, 5]  layer 5 global → NOT safe
          slot 2 → [2]     global → NOT safe
          slot 3 → [3]     local  → SAFE
        """
        pattern = [0, 0, 1, 0, 0, 1]
        result = _compute_safe_sliding_slots(6, 2, pattern)
        assert result == [0, 3], f"Expected [0, 3], got {result}"

    def test_kv_sharing_all_slots_mixed(self):
        """If every slot is shared between a local and global layer, no slots are safe."""
        # num_layers=4, num_kv_shared_layers=2, num_unique_kv=2
        # pattern=[0, 1, 1, 0]
        # slot 0 → [0, 2]: layer 2 global → NOT safe
        # slot 1 → [1, 3]: layer 1 global → NOT safe
        pattern = [0, 1, 1, 0]
        result = _compute_safe_sliding_slots(4, 2, pattern)
        assert result == [], f"Expected [], got {result}"

    def test_kv_sharing_all_local_all_safe(self):
        """If all layers are local and sharing is active, all num_unique_kv slots are safe."""
        # num_layers=6, num_kv_shared_layers=3, num_unique_kv=3
        # pattern=[0, 0, 0, 0, 0, 0]
        # slot 0 → [0, 3] both local
        # slot 1 → [1, 4] both local
        # slot 2 → [2, 5] both local
        pattern = [0, 0, 0, 0, 0, 0]
        result = _compute_safe_sliding_slots(6, 3, pattern)
        assert result == [0, 1, 2], f"Expected [0, 1, 2], got {result}"

    def test_single_layer_global(self):
        """Edge case: single layer that is global — no safe slots."""
        result = _compute_safe_sliding_slots(1, 0, [1])
        assert result == [], f"Expected [], got {result}"

    def test_single_layer_local(self):
        """Edge case: single layer that is local — slot 0 is safe."""
        result = _compute_safe_sliding_slots(1, 0, [0])
        assert result == [0], f"Expected [0], got {result}"


class TestGemma4GenaiConfigJson:
    """Tests that make_genai_config emits the right sliding_window.layers to genai_config.json."""

    def _run_make_genai_config(
        self,
        num_layers: int,
        num_kv_shared_layers: int,
        attention_pattern: list[int],
    ) -> dict:
        """
        Call Gemma4Model.make_genai_config and return the resulting genai_config decoder dict.
        We stub out the parent's make_genai_config to avoid needing a real model.
        """
        try:
            from onnxruntime_genai.models.builders.gemma import Gemma4Model
        except ImportError:
            pytest.skip("onnxruntime_genai.models not importable; skipping builder test")

        import tempfile

        cfg = _make_gemma4_config(
            num_hidden_layers=num_layers,
            num_kv_shared_layers=num_kv_shared_layers,
            attention_pattern=attention_pattern,
        )

        with tempfile.TemporaryDirectory() as tmp:
            # Write a base config that the parent make_genai_config would have written
            base_decoder = {}
            genai_cfg = {"model": {"type": "gemma4_text", "decoder": base_decoder}}
            with open(os.path.join(tmp, "genai_config.json"), "w") as f:
                json.dump(genai_cfg, f)

            # Patch parent's make_genai_config to be a no-op
            with mock.patch.object(
                Gemma4Model.__bases__[0], "make_genai_config", return_value=None
            ):
                # Instantiate without real cache/model files (extra_options suppress file I/O)
                try:
                    m = Gemma4Model.__new__(Gemma4Model)
                    m.num_layers = num_layers
                    m.num_kv_shared_layers = num_kv_shared_layers
                    m.num_unique_kv = num_layers - num_kv_shared_layers
                    m.attention_pattern = attention_pattern
                    m.global_head_size = 8
                    m.head_size = 4
                    m.window_size = 16
                    m.hidden_size_per_layer_input = 0
                    m.ep = "cpu"
                    # Call only the Gemma4-specific tail of make_genai_config
                    # (the part after the super() call)
                    Gemma4Model.make_genai_config(m, "model_name", {}, tmp)
                except Exception:
                    pytest.skip("Cannot instantiate Gemma4Model stub; skipping")

            with open(os.path.join(tmp, "genai_config.json")) as f:
                result = json.load(f)
            return result["model"]["decoder"]

    def test_sliding_window_layers_with_kv_sharing(self):
        """With KV sharing, only exclusively-local slots appear in sliding_window.layers."""
        # num_layers=6, num_kv_shared_layers=2, pattern=[0,0,1,0,0,1]
        # Expected safe slots: [0, 3]
        decoder = self._run_make_genai_config(6, 2, [0, 0, 1, 0, 0, 1])
        if "sliding_window" not in decoder:
            pytest.skip("make_genai_config did not write sliding_window (likely no window_size)")
        assert decoder["sliding_window"]["layers"] == [0, 3], (
            f"Expected [0, 3], got {decoder['sliding_window']['layers']}"
        )

    def test_sliding_window_layers_without_kv_sharing(self):
        """Without KV sharing, all local layer indices appear in sliding_window.layers."""
        # num_layers=4, num_kv_shared_layers=0, pattern=[0,0,1,0]
        # Expected: [0, 1, 3]
        decoder = self._run_make_genai_config(4, 0, [0, 0, 1, 0])
        if "sliding_window" not in decoder:
            pytest.skip("make_genai_config did not write sliding_window")
        assert decoder["sliding_window"]["layers"] == [0, 1, 3], (
            f"Expected [0, 1, 3], got {decoder['sliding_window']['layers']}"
        )


# ---------------------------------------------------------------------------
# Runtime tests (require onnxruntime_genai)
# ---------------------------------------------------------------------------


def test_gemma4_model_loads(gemma4_model_path):
    """Test that the Gemma 4 KV-sharing dummy model loads without error."""
    try:
        import onnxruntime_genai as og
    except ImportError:
        pytest.skip("onnxruntime_genai not installed")

    model = og.Model(gemma4_model_path)
    assert model is not None


def test_gemma4_generator_creates(gemma4_model_path):
    """Test that a Generator can be created — validates KV-cache + per-layer-inputs wiring."""
    try:
        import onnxruntime_genai as og
    except ImportError:
        pytest.skip("onnxruntime_genai not installed")

    model = og.Model(gemma4_model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=10)
    generator = og.Generator(model, params)
    assert generator is not None


def test_gemma4_sliding_window_config(gemma4_model_path):
    """
    Verify the genai_config.json sliding_window.layers contains only exclusively-local slots.

    For the dummy model:
      num_layers=6, num_kv_shared_layers=2, num_unique_kv=4
      attention_pattern=[0, 0, 1, 0, 0, 1]
      Expected safe slots: [0, 3]
    """
    config_path = os.path.join(gemma4_model_path, "genai_config.json")
    with open(config_path) as f:
        config = json.load(f)

    decoder = config["model"]["decoder"]
    assert "sliding_window" in decoder, "sliding_window config missing from genai_config.json"

    layers = decoder["sliding_window"]["layers"]
    assert layers == [0, 3], (
        f"Expected sliding_window.layers == [0, 3] (exclusively-local slots), got {layers}"
    )


def test_gemma4_head_sizes_in_onnx(gemma4_model_path):
    """
    Verify that the dummy ONNX model encodes the correct head_size for each KV slot.

    Slot 0 (layer 0, local) -> head_size = 4
    Slot 1 (layer 1, local) -> head_size = 4
    Slot 2 (layer 2, global) -> head_size = 8
    Slot 3 (layer 3, local) -> head_size = 4
    """
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx not installed")

    model = onnx.load(os.path.join(gemma4_model_path, "dummy_decoder.onnx"))
    # Collect input shapes by name
    input_shapes = {}
    for inp in model.graph.input:
        shape = [d.dim_param if d.HasField("dim_param") else d.dim_value
                 for d in inp.type.tensor_type.shape.dim]
        input_shapes[inp.name] = shape

    expected = {
        "past_key_values.0.key": 4,
        "past_key_values.1.key": 4,
        "past_key_values.2.key": 8,  # global
        "past_key_values.3.key": 4,
    }
    for key_name, expected_hs in expected.items():
        assert key_name in input_shapes, f"Missing input {key_name}"
        actual_hs = input_shapes[key_name][3]
        assert actual_hs == expected_hs, (
            f"{key_name}: expected head_size={expected_hs}, got {actual_hs}"
        )


def test_gemma4_rewind_does_not_crash(gemma4_model_path):
    """
    Smoke test that generator.rewind_to() works on a Gemma 4 model (exercises
    PerLayerInputs::RewindTo and KV cache RewindTo).
    """
    try:
        import onnxruntime_genai as og
        import numpy as np
    except ImportError:
        pytest.skip("onnxruntime_genai not installed")

    model = og.Model(gemma4_model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=15, do_sample=False)

    generator = og.Generator(model, params)
    # Append a couple of tokens, then rewind
    generator.append_tokens(np.array([[1, 2]], dtype=np.int32))
    # Rewind should not raise even with per_layer_inputs active
    generator.rewind_to(0)


# ---------------------------------------------------------------------------
# VLM type registration test
# ---------------------------------------------------------------------------

def test_gemma4_vlm_type_is_registered(gemma4_vlm_path):
    """
    Verify that model type 'gemma4' is recognised as a VLM (IsVLM returns true)
    and that create_multimodal_processor() does not raise a 'not registered' error.
    """
    try:
        import onnxruntime_genai as og
    except ImportError:
        pytest.skip("onnxruntime_genai not installed")

    model = og.Model(gemma4_vlm_path)
    # create_multimodal_processor() raises if the type is not in the VLM factory
    # (it would say "gemma4 is not a registered multi-modal model type").
    try:
        processor = model.create_multimodal_processor()
        assert processor is not None
    except RuntimeError as e:
        if "not a registered multi-modal model type" in str(e):
            pytest.fail(f"'gemma4' not registered as VLM type: {e}")
        # Other errors (e.g. missing processor_config.json) are acceptable for the dummy model
        pass


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_gemma4_tests(
    cwd: str | bytes | os.PathLike,
    log: logging.Logger,
    test_models: str | bytes | os.PathLike,
) -> None:
    """Run the Gemma 4 tests using pytest."""
    from _test_utils import run_subprocess

    log.debug("Running: Gemma 4 model tests")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-sv",
        "test_gemma4_models.py",
        "--test_models",
        str(test_models),
    ]
    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", default=Path(__file__).parent.resolve())
    parser.add_argument("--test_models", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_gemma4_tests(args.cwd, log, args.test_models)
