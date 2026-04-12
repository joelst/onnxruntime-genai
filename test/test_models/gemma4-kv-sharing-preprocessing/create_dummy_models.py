#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Generate dummy ONNX models for Gemma 4 KV-sharing tests.

Creates a minimal decoder model that exercises:
- KV cache sharing: num_layers=6, num_kv_shared_layers=2 -> num_unique_kv=4
  (layers 0-3 own their KV slots; layers 4-5 share slots 0 and 1)
- Variable head dimensions: local/sliding layers use head_size=4, global layers use
  global_head_size=8.  Attention pattern: [0, 0, 1, 0, 0, 1] (layers 2 and 5 are global)
- sliding_window config with correctly-identified exclusively-local cache slots
- per_layer_inputs tensor (zero-initialised, present in ONNX graph as an input)

Layout summary (6 layers, 4 unique KV slots, global period = every 3rd layer):
  Layer 0 -> slot 0  local  (head_size 4)
  Layer 1 -> slot 1  local  (head_size 4)
  Layer 2 -> slot 2  global (head_size 8)
  Layer 3 -> slot 3  local  (head_size 4)
  Layer 4 -> slot 0  local  (head_size 4, shares slot 0 with layer 0)
  Layer 5 -> slot 1  global (head_size 8, shares slot 1 with layer 1)

  Safe sliding-window slots (exclusively local):
    slot 0 -> layers 0, 4  both local -> SAFE
    slot 1 -> layers 1, 5  layer 5 is global -> NOT safe
    slot 2 -> layer  2     global -> NOT safe
    slot 3 -> layer  3     local  -> SAFE
  => sliding_window.layers = [0, 3]

Usage:
    python create_dummy_models.py [--output <dir>]
"""

import argparse
import json
import os

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:
    print("onnx package required: pip install onnx")
    exit(1)

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
NUM_LAYERS = 6
NUM_KV_SHARED_LAYERS = 2
NUM_UNIQUE_KV = NUM_LAYERS - NUM_KV_SHARED_LAYERS  # = 4
ATTENTION_PATTERN = [0, 0, 1, 0, 0, 1]  # 0=local, 1=global
HEAD_SIZE_LOCAL = 4   # sliding-window layers
HEAD_SIZE_GLOBAL = 8  # full-attention layers
NUM_KV_HEADS = 2
BATCH = "batch_size"
HIDDEN_SIZE = 32
VOCAB_SIZE = 100
MAX_SEQ_LEN = 64
HIDDEN_SIZE_PER_LAYER_INPUT = 16


def _kv_head_size(layer_idx: int) -> int:
    return HEAD_SIZE_GLOBAL if ATTENTION_PATTERN[layer_idx] == 1 else HEAD_SIZE_LOCAL


def _slot(layer_idx: int) -> int:
    return layer_idx % NUM_UNIQUE_KV


def create_decoder_model(output_path: str) -> None:
    """Create a minimal decoder with KV cache (KV sharing + variable head_size) + per_layer_inputs."""
    inputs = []
    outputs = []
    nodes = []
    initializers = []

    # input_ids: [batch, seq]
    inputs.append(helper.make_tensor_value_info("input_ids", TensorProto.INT64, [BATCH, "seq_len"]))
    # attention_mask: [batch, total_seq]
    inputs.append(helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [BATCH, "total_seq_len"]))
    # position_ids: [batch, seq]
    inputs.append(helper.make_tensor_value_info("position_ids", TensorProto.INT64, [BATCH, "seq_len"]))
    # per_layer_inputs: [batch, seq, num_layers, hidden_size_per_layer_input]
    inputs.append(helper.make_tensor_value_info(
        "per_layer_inputs", TensorProto.FLOAT,
        [BATCH, "seq_len", NUM_LAYERS, HIDDEN_SIZE_PER_LAYER_INPUT]))

    # KV cache inputs (num_unique_kv slots, 2 per slot)
    for slot in range(NUM_UNIQUE_KV):
        # Determine the head_size for this slot — use the first layer that maps to it
        first_layer = slot  # layer 'slot' is the canonical owner
        hs = _kv_head_size(first_layer)
        inputs.append(helper.make_tensor_value_info(
            f"past_key_values.{slot}.key", TensorProto.FLOAT,
            [BATCH, NUM_KV_HEADS, "past_seq_len", hs]))
        inputs.append(helper.make_tensor_value_info(
            f"past_key_values.{slot}.value", TensorProto.FLOAT,
            [BATCH, NUM_KV_HEADS, "past_seq_len", hs]))

    # logits output: [batch, seq, vocab_size]
    outputs.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, [BATCH, "seq_len", VOCAB_SIZE]))
    # KV cache outputs
    for slot in range(NUM_UNIQUE_KV):
        first_layer = slot
        hs = _kv_head_size(first_layer)
        outputs.append(helper.make_tensor_value_info(
            f"present.{slot}.key", TensorProto.FLOAT,
            [BATCH, NUM_KV_HEADS, "total_seq_len", hs]))
        outputs.append(helper.make_tensor_value_info(
            f"present.{slot}.value", TensorProto.FLOAT,
            [BATCH, NUM_KV_HEADS, "total_seq_len", hs]))

    # Dummy logits initializer (constant zeros — no real computation needed for testing)
    import numpy as np
    logits_data = np.zeros([1, 1, VOCAB_SIZE], dtype=np.float32)
    logits_init = onnx.numpy_helper.from_array(logits_data, name="logits_const")
    initializers.append(logits_init)

    # KV present outputs — just pass-through past as present (identity)
    present_inits = []
    for slot in range(NUM_UNIQUE_KV):
        first_layer = slot
        hs = _kv_head_size(first_layer)
        # key
        key_data = np.zeros([1, NUM_KV_HEADS, 1, hs], dtype=np.float32)
        key_init = onnx.numpy_helper.from_array(key_data, name=f"present_key_init_{slot}")
        initializers.append(key_init)
        present_inits.append((f"present.{slot}.key", f"present_key_init_{slot}"))
        # value
        val_data = np.zeros([1, NUM_KV_HEADS, 1, hs], dtype=np.float32)
        val_init = onnx.numpy_helper.from_array(val_data, name=f"present_val_init_{slot}")
        initializers.append(val_init)
        present_inits.append((f"present.{slot}.value", f"present_val_init_{slot}"))

    # Identity nodes to assign initializers to output names
    for out_name, init_name in present_inits:
        nodes.append(helper.make_node("Identity", inputs=[init_name], outputs=[out_name]))
    nodes.append(helper.make_node("Identity", inputs=["logits_const"], outputs=["logits"]))

    graph = helper.make_graph(nodes, "gemma4_dummy", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Created {output_path}")


def create_genai_config(output_path: str) -> None:
    """Create genai_config.json exercising all Gemma 4 runtime features."""
    # Compute the safe-to-constrain sliding-window cache slots.
    # Slot s is safe iff every model layer mapping to it (s, s+NUM_UNIQUE_KV, ...) is local.
    slot_users: dict = {}
    for layer in range(NUM_LAYERS):
        s = _slot(layer)
        slot_users.setdefault(s, []).append(layer)

    safe_sliding_slots = [
        s for s in range(NUM_UNIQUE_KV)
        if slot_users.get(s) and all(ATTENTION_PATTERN[l] == 0 for l in slot_users[s])
    ]

    config = {
        "model": {
            "bos_token_id": 1,
            "context_length": MAX_SEQ_LEN,
            "decoder": {
                "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
                "filename": "dummy_decoder.onnx",
                "head_size": HEAD_SIZE_LOCAL,
                "global_head_size": HEAD_SIZE_GLOBAL,
                "hidden_size": HIDDEN_SIZE,
                "num_attention_heads": NUM_KV_HEADS,
                "num_hidden_layers": NUM_LAYERS,
                "num_key_value_heads": NUM_KV_HEADS,
                "num_kv_shared_layers": NUM_KV_SHARED_LAYERS,
                "attention_pattern": ATTENTION_PATTERN,
                "hidden_size_per_layer_input": HIDDEN_SIZE_PER_LAYER_INPUT,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "sliding_window": {
                    "window_size": 8,
                    "slide_key_value_cache": False,
                    "slide_inputs": False,
                    "layers": safe_sliding_slots,
                },
            },
            "eos_token_id": 1,
            "pad_token_id": 0,
            "type": "gemma4_text",
            "vocab_size": VOCAB_SIZE,
        },
        "search": {
            "do_sample": False,
            "early_stopping": True,
            "max_length": MAX_SEQ_LEN,
            "min_length": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "top_k": 1,
        },
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Created {output_path}")


def copy_tokenizer_files(output_dir: str) -> None:
    """Copy minimal tokenizer files from an existing test model."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "hf-internal-testing", "tiny-random-gpt2-fp32")
    if not os.path.isdir(src_dir):
        print(f"  WARNING: tokenizer source not found at {src_dir}, skipping tokenizer copy")
        return
    for fname in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
    print("  Copied tokenizer files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy ONNX models for Gemma 4 KV-sharing tests")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.dirname(__file__),
        help="Output directory (default: same directory as this script)",
    )
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating Gemma 4 dummy test model in {output_dir}")
    print(f"  num_layers={NUM_LAYERS}, num_kv_shared_layers={NUM_KV_SHARED_LAYERS}, "
          f"num_unique_kv={NUM_UNIQUE_KV}")
    print(f"  attention_pattern={ATTENTION_PATTERN}")

    create_decoder_model(os.path.join(output_dir, "dummy_decoder.onnx"))
    create_genai_config(os.path.join(output_dir, "genai_config.json"))
    copy_tokenizer_files(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
