#pragma once

namespace Generators {

struct PresetExtraInputs {
  PresetExtraInputs(State& state);
  void Add();

 private:
  using FuncType = std::function<std::unique_ptr<OrtValue>()>;
  State& state_;
  std::unordered_map<std::string, FuncType> registry_;
  std::vector<std::unique_ptr<OrtValue>> extra_inputs_;
  std::vector<std::string> extra_input_names_;
};

struct ExtraInputs {
  ExtraInputs(State& state);
  void Add(const std::vector<ExtraInput>& extra_inputs, const std::vector<std::string>& required_input_names = {});

 private:
  State& state_;
  const Model& model_{state_.model_};
  PresetExtraInputs registrar_{state_};
};

// Manages the per_layer_inputs tensor for models that use Per-Layer Embeddings (PLE), e.g. Gemma 4.
// The tensor has shape [batch_size, sequence_length, num_hidden_layers, hidden_size_per_layer_input]
// and is re-allocated each step to match the current sequence length.  Values are zero-initialised
// until a dedicated embed model supplies the actual per-layer embeddings.
struct PerLayerInputs {
  PerLayerInputs(State& state);

  bool IsActive() const { return is_active_; }

  // Wire the input into the state's input list.  Must be called once during state construction.
  void Add();

  // Resize the tensor to match the new sequence length and zero-initialise it.
  void Update(int seq_length);

 private:
  Ort::Allocator& Allocator() { return model_.allocator_cpu_; }

  State& state_;
  const Model& model_{state_.model_};
  bool is_active_{false};

  // [batch_size, sequence_length, num_hidden_layers, hidden_size_per_layer_input]
  std::array<int64_t, 4> shape_{};
  ONNXTensorElementDataType type_{};
  std::unique_ptr<OrtValue> tensor_;

  size_t index_{~0U};  // position of this input in state_.inputs_
};

}  // namespace Generators
