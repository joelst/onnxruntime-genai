#include "../generators.h"
#include "model.h"
#include "extra_inputs.h"

namespace Generators {

PresetExtraInputs::PresetExtraInputs(State& state)
    : state_(state),
      registry_{
          {"num_logits_to_keep", [&state = state_]() -> std::unique_ptr<OrtValue> {
             std::vector<int64_t> shape{1};
             auto num_logits_to_keep = OrtValue::CreateTensor<int64_t>(state.model_.allocator_cpu_, shape);
             *num_logits_to_keep->GetTensorMutableData<int64_t>() = 0;
             return num_logits_to_keep;
           }}} {}

void PresetExtraInputs::Add() {
  const auto input_names_vector = state_.model_.session_info_.GetInputNames();
  const std::unordered_set<std::string> input_names(state_.input_names_.begin(), state_.input_names_.end());
  std::vector<std::string> unclaimed_input_names;
  // Add any model input for which we don't have a corresponding input in the state to the unclaimed_input_names
  for (const auto& input_name : input_names_vector) {
    if (input_names.find(input_name) == input_names.end()) {
      unclaimed_input_names.push_back(input_name);
    }
  }

  // Try to claim the unclaimed inputs from the registry
  for (const auto& input_name : unclaimed_input_names) {
    auto it = registry_.find(input_name);
    if (it != registry_.end()) {
      extra_input_names_.push_back(input_name);
      extra_inputs_.push_back(it->second());
      state_.input_names_.push_back(extra_input_names_.back().c_str());
      state_.inputs_.push_back(extra_inputs_.back().get());
    } else if (input_name.rfind("onnx::Neg_", 0) == 0) {
      // The unclaimed input has a prefix of onnx::Neg_, which is a special case
      // We treat this as an alias to num_logits_to_keep
      extra_input_names_.push_back(input_name);
      extra_inputs_.push_back(registry_.at("num_logits_to_keep")());
      state_.input_names_.push_back(extra_input_names_.back().c_str());
      state_.inputs_.push_back(extra_inputs_.back().get());
    }
  }
}

ExtraInputs::ExtraInputs(State& state)
    : state_{state} {}

void ExtraInputs::Add(const std::vector<ExtraInput>& extra_inputs, const std::vector<std::string>& required_input_names) {
  std::unordered_set<std::string> required_input_names_set(required_input_names.begin(), required_input_names.end());
  // Add extra user inputs
  for (int i = 0; i < extra_inputs.size(); i++) {
    if (required_input_names_set.empty() || required_input_names_set.count(extra_inputs[i].name)) {
      state_.input_names_.push_back(extra_inputs[i].name.c_str());
      state_.inputs_.push_back(extra_inputs[i].tensor->ort_tensor_.get());
    }
  }

  registrar_.Add();
}

// ---------------------------------------------------------------------------
// PerLayerInputs
// ---------------------------------------------------------------------------

PerLayerInputs::PerLayerInputs(State& state)
    : state_{state} {
  const int hidden_size_per_layer = model_.config_->model.decoder.hidden_size_per_layer_input;
  if (hidden_size_per_layer <= 0) {
    return;
  }

  // Check whether the model session actually exposes this input (avoids activating for models
  // that have the config field but do not expose per_layer_inputs in the ONNX graph).
  const auto& session_input_names = model_.session_info_.GetInputNames();
  const bool in_session = std::any_of(session_input_names.begin(), session_input_names.end(),
                                      [](const std::string& n) { return n == "per_layer_inputs"; });
  if (!in_session) {
    return;
  }

  is_active_ = true;

  // Resolve data type from the session metadata (falls back to model io_dtype).
  type_ = model_.session_info_.GetInputDataType("per_layer_inputs");

  shape_ = {
      static_cast<int64_t>(state_.params_->BatchBeamSize()),
      0,  // sequence_length — updated on first Update() call
      static_cast<int64_t>(model_.config_->model.decoder.num_hidden_layers),
      static_cast<int64_t>(hidden_size_per_layer),
  };

  // Allocate an empty (seq_len=0) placeholder so that the input slot is never null.
  tensor_ = OrtValue::CreateTensor(Allocator(), shape_, type_);
}

void PerLayerInputs::Add() {
  if (!is_active_) return;

  index_ = state_.inputs_.size();
  state_.inputs_.push_back(tensor_.get());
  state_.input_names_.push_back("per_layer_inputs");
}

void PerLayerInputs::Update(int seq_length) {
  if (!is_active_) return;

  if (shape_[1] == seq_length) return;  // Nothing to do — shape unchanged.

  shape_[1] = static_cast<int64_t>(seq_length);
  tensor_ = OrtValue::CreateTensor(Allocator(), shape_, type_);

  // Zero-initialise the tensor.  The actual per-layer embeddings will come from a dedicated
  // embed model once that pipeline stage is wired up; until then zeros are safe because no
  // ONNX node in the current decoder graph consumes this input.
  // Use size_t for all multiplications to avoid 32-bit intermediate overflow.
  const size_t num_elements = static_cast<size_t>(shape_[0]) * static_cast<size_t>(shape_[1]) *
                              static_cast<size_t>(shape_[2]) * static_cast<size_t>(shape_[3]);
  const size_t num_bytes = num_elements * Ort::SizeOf(type_);
  std::memset(tensor_->GetTensorMutableRawData(), 0, num_bytes);
  std::memset(tensor_->GetTensorMutableRawData(), 0, num_bytes);

  state_.inputs_[index_] = tensor_.get();
}

}  // namespace Generators
