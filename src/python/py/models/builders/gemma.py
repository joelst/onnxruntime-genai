# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np

from .mistral import MistralModel


class GemmaModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.embed_attrs["scale"] = np.round(np.sqrt(self.hidden_size), decimals=2)
        self.layernorm_attrs["add_offset"] = 1


class Gemma2Model(GemmaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = False
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = False
        self.attention_attrs["scale"] = config.query_pre_attn_scalar**-0.5

    def is_local(self, layer_id):
        return layer_id % 2 == 1

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        if "final_norm" in location:
            # Set cast for final LayerNorm since it is a special case and not covered in `make_layer`
            self.layernorm_attrs["cast"]["root_input"] = False
        super().make_layernorm(layer_id, layernorm, skip, simple, location)

    def make_layer(self, layer_id, layer):
        # Gemma-2 decoder layer is typically defined as:
        # input_layernorm --> attention --> post_attention_layernorm --> pre_ffn_layernorm --> MLP --> post_ffn_layernorm

        # Adjust LayerNorm attributes because of extra LayerNorms inserted
        # 1. Only cast root_input if the first layer of LayerNorms are being created
        original_cast_root_input = self.layernorm_attrs["cast"]["root_input"]
        self.layernorm_attrs["cast"]["root_input"] = self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.layernorm_attrs["cast"]["root_input"] = original_cast_root_input

        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        # Adjust LayerNorm attributes for extra LayerNorm to insert
        # 1. Temporarily set root_input for LayerNorm to skip_input for post_attention_layernorm
        # 2. Set skip_input to output of post_attention_layernorm
        # 3. Do not cast outputs from post_attention_layernorm
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        # Adjust LayerNorm attributes because of extra LayerNorms inserted
        # 1. Only cast root_input if the first layer of LayerNorms are being created
        original_cast_root_input = self.layernorm_attrs["cast"]["root_input"]
        self.layernorm_attrs["cast"]["root_input"] = self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(
            layer_id,
            layer.pre_feedforward_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="pre_feedforward",
        )
        self.layernorm_attrs["cast"]["root_input"] = original_cast_root_input

        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        # Adjust LayerNorm attributes for extra LayerNorm to insert
        # 1. Temporarily set root_input for LayerNorm to skip_input for post_feedforward_layernorm
        # 2. Set skip_input to output of post_feedforward_layernorm
        # 3. Do not cast outputs from post_feedforward_layernorm
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id,
            layer.post_feedforward_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_feedforward",
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        original_window_size = self.window_size
        self.window_size = (
            original_window_size if self.is_local(layer_id) else -1
        )  # default is -1 in GroupQueryAttention kernel
        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.window_size = original_window_size


class Gemma3Model(Gemma2Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.rope_local_theta = config.rope_local_base_freq
        self.make_rotary_embedding_multi_cache()

    def is_local(self, layer_id):
        return bool((layer_id + 1) % 6)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()

    def make_rotary_embedding_multi_cache(self):
        self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_global_name, sin_cache_name=self.sin_cache_global_name
        )

        # Create the new cos/sin caches for local attention layers with its own theta value
        self.rope_attrs["create_caches"] = True
        self.rope_attrs["theta"] = self.rope_local_theta

        self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name
        )

    def make_rotary_embedding_caches(self, **kwargs):
        cos_cache_name = kwargs.get(
            "cos_cache_name", self.cos_cache_global_name if self.window_size == -1 else self.cos_cache_local_name
        )
        sin_cache_name = kwargs.get(
            "sin_cache_name", self.sin_cache_global_name if self.window_size == -1 else self.sin_cache_local_name
        )
        return super().make_rotary_embedding_caches(cos_cache_name=cos_cache_name, sin_cache_name=sin_cache_name)


class Gemma4Model(Gemma3Model):
    """
    Model builder for Gemma 4, which extends Gemma 3 with three new architectural features:

    1. Per-Layer Embeddings (PLE): the embed model produces a ``per_layer_inputs`` tensor of shape
       ``[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`` that is fed as an additional
       decoder input.
    2. Variable attention head dimensions: sliding-window layers use ``head_dim`` (e.g. 256) while
       full-attention layers (every 5th layer by default) use ``global_head_dim`` (e.g. 512).
    3. KV cache sharing: the first ``num_kv_shared_layers`` decoder layers share KV caches with the
       last ``num_hidden_layers - num_kv_shared_layers`` unique layers (accessed via modulo mapping).
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Variable head dimensions per layer (HuggingFace uses 'global_head_dim', C++ config uses 'global_head_size')
        self.global_head_size = (
            getattr(config, "global_head_dim", None)
            or getattr(config, "global_head_size", None)
            or self.head_size
        )

        # Per-layer attention pattern: 0 = sliding/local, 1 = full/global
        self.attention_pattern = list(getattr(config, "attention_pattern", []))

        # KV cache sharing
        self.num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        self.num_unique_kv = max(1, self.num_layers - self.num_kv_shared_layers)

        # Per-layer embeddings (PLE)
        self.hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", 0)

        # Remap KV cache tensor names to shared indices
        if self.num_kv_shared_layers > 0:
            for i in range(self.num_layers):
                cache_id = self._get_kv_cache_id(i)
                self.input_names["past_key_values.key"][i] = f"past_key_values.{cache_id}.key"
                self.input_names["past_key_values.value"][i] = f"past_key_values.{cache_id}.value"
                self.output_names["present.key"][i] = f"present.{cache_id}.key"
                self.output_names["present.value"][i] = f"present.{cache_id}.value"

        # Register per_layer_inputs as an additional model input when PLE is used
        if self.hidden_size_per_layer_input > 0:
            self.input_names["per_layer_inputs"] = "per_layer_inputs"
            self.input_types["per_layer_inputs"] = self.io_dtype
            self.input_shapes["per_layer_inputs"] = [
                "batch_size",
                "sequence_length",
                self.num_layers,
                self.hidden_size_per_layer_input,
            ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_kv_cache_id(self, layer_id):
        """Map a layer index to its KV cache index, accounting for sharing."""
        return layer_id % self.num_unique_kv

    def is_local(self, layer_id):
        """Return True for sliding/local attention layers using the per-layer attention pattern."""
        if self.attention_pattern:
            return self.attention_pattern[layer_id] == 0  # 0=sliding, 1=full
        return super().is_local(layer_id)

    # ------------------------------------------------------------------
    # Attention with per-layer head size
    # ------------------------------------------------------------------

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Switch head_size for full-attention layers before building the attention subgraph."""
        is_full_attn = bool(self.attention_pattern) and not self.is_local(layer_id)
        if is_full_attn and self.global_head_size != self.head_size:
            original_head_size = self.head_size
            self.head_size = self.global_head_size
            try:
                super().make_attention(layer_id, attention, root_input, **kwargs)
            finally:
                self.head_size = original_head_size
        else:
            super().make_attention(layer_id, attention, root_input, **kwargs)

    # ------------------------------------------------------------------
    # genai_config.json emission
    # ------------------------------------------------------------------

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        # Append Gemma 4-specific fields to the already-written config
        import json
        import os

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        decoder = genai_config["model"]["decoder"]
        if self.global_head_size and self.global_head_size != self.head_size:
            decoder["global_head_size"] = self.global_head_size
        if self.attention_pattern:
            decoder["attention_pattern"] = self.attention_pattern
        if self.num_kv_shared_layers > 0:
            decoder["num_kv_shared_layers"] = self.num_kv_shared_layers
        if self.hidden_size_per_layer_input > 0:
            decoder["hidden_size_per_layer_input"] = self.hidden_size_per_layer_input

        # For non-TRT EPs, the base class does not write a sliding_window config. Add it here so that
        # the C++ runtime can apply per-layer KV cache size constraints (sliding layers → window_size
        # tokens; full-attention layers → max_length tokens), saving significant memory for Gemma 4
        # where ~80 % of layers are sliding-window.
        if self.ep != "trt-rtx" and self.attention_pattern and self.window_size and self.window_size > 0:
            if "sliding_window" not in decoder:
                sliding_layer_idxs = [i for i in range(self.num_layers) if self.is_local(i)]
                decoder["sliding_window"] = {
                    "window_size": self.window_size,
                    "slide_key_value_cache": False,
                    "slide_inputs": False,
                    "layers": sliding_layer_idxs,
                }

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)
