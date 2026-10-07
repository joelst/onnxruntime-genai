# Whisper timestamp decoding

ONNX Runtime GenAI can apply Whisper's token-level timestamp rules during generation.
The rules run before token selection and cover timestamp pairing and monotonicity,
the initial timestamp boundary, timestamp-versus-text probability mass, and Whisper
control-token suppression.

Timestamp decoding is opt-in:

```python
params = og.GeneratorParams(model)
params.set_search_options(
    whisper_timestamps=True,
    whisper_max_initial_timestamp_index=50,
)
```

Use a timestamp-compatible Whisper decoder prompt and do not include the
`<|notimestamps|>` token. The runtime rejects timestamp-enabled prompts containing that
token.

`whisper_max_initial_timestamp_index` is measured in 20 ms timestamp-token intervals.
The default value, 50, limits the first timestamp to the first second of the current
audio window. Set it to `-1` to disable the initial boundary.

Models must provide `model.timestamp_begin_token_id` and
`model.no_timestamps_token_id` in `genai_config.json`. The Whisper model builder emits
these values when the tokenizer contains the standard contiguous timestamp-token suffix.

Timestamp rule processing runs on the model's scoring device. CPU and CUDA scoring are
implemented. Providers that use CPU scoring follow the CPU path, but provider-specific
model execution must still be validated separately. NvTensorRtRtx inherits the CUDA
timestamp implementation, but provider-specific Whisper timestamp execution has not been
validated.

Timestamp decoding is not supported with speculative decoding, guidance, multiple EOS
token IDs, or the continuous batching Engine.

## Consuming timestamp tokens

Generated sequences retain timestamp token IDs so a transcription component can inspect
them, but tokenizer decoding and streaming decoding omit timestamp tokens from visible
text. Use the tokenizer timestamp APIs to distinguish text from timestamps and convert a
timestamp to seconds relative to the current audio window:

```python
if tokenizer.has_timestamp_tokens:
    prompt_length = len(decoder_prompt)
    for token_id in generator.get_sequence(0)[prompt_length:]:
        if tokenizer.is_timestamp_token(token_id):
            relative_seconds = tokenizer.timestamp_to_seconds(token_id)
```

The example inspects a completed generated suffix. Prompt timestamps used for
conditioning are not current-window output.

`timestamp_begin_token_id` raises an error when timestamp metadata is unavailable.
`timestamp_to_seconds` raises an error when passed a token outside the model's timestamp
token range.

Equivalent capability, classification, and conversion APIs are available in C, C++,
C#, and Java.

## Ownership boundary

These APIs expose low-level, relative-window decoding state. A higher-level
transcription component remains responsible for audio chunking, media duration probing,
VAD policy, seek/redecode behavior, adding each window's absolute offset, and final
segment assembly. Word-level cross-attention/DTW timestamps are not provided by this
feature. Create a fresh generator for each new audio window or redecode attempt; appending
another window after timestamp generation begins is rejected.

## Completed-window segment prototype

On the prototype branch, `examples/python/whisper.py --timestamps --segments`
demonstrates consumer-side segment extraction. This is not a library API or an
implementation of the metadata API proposed in #2591. Use a build containing the
Whisper timestamp primitives and a timestamp-compatible Whisper model export:

```bash
python examples/python/whisper.py -m path/to/whisper-tiny-fp32-cpu -e cpu -b 5 \
  --timestamps --segments
```

Enter `test/audios/1272-141231-0002.mp3` at the audio-path prompt.

The example emits one JSON record per completed batch/beam hypothesis alongside
its existing console output. Each record identifies the batch and beam, declares
`time_reference` as `audio_window`, and contains segments with `text`, `start_time`,
and `stop_time`. It verifies and removes the decoder prompt before examining the
generated suffix, including any timestamp tokens used for prompt conditioning.

The extractor supports the standard Whisper vocabulary, where text IDs precede
the single EOS token and control IDs precede the timestamp-token suffix. It
requires a separate opening boundary and a later closing boundary for every
text segment, permits repeated boundaries and gaps between segments, and rejects
decreasing timestamps, unexpected controls, or unfinished text. A truncated
hypothesis without a closing boundary raises an error identifying its batch/beam;
it does not return a partial list or invent an end time. Timestamp-only output
produces no segments. An empty list is not a silence-detection result.

Only finalized sequences are inspected. There is no assumption that intermediate
beam hypotheses can be irreversibly accumulated into streaming segments.
Successful extraction does not establish that all speech in the input was
transcribed. Window offsets, duration clipping, chunking, and redecode decisions
remain consumer responsibilities.

This prototype supplies segment text and window-relative seconds, not per-word
alignment or absolute acoustic-frame bounds. Those are the concrete fields to
compare with #2591's timestamp result records. Sharing the result envelope remains
a follow-up design question; this branch adds no dependency on its metadata state,
Extensions word aggregation, or public bindings.
