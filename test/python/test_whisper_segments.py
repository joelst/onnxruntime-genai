# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


@pytest.fixture(scope="module")
def segment_example():
    path = Path(__file__).resolve().parents[2] / "examples" / "python" / "whisper_segments.py"
    spec = importlib.util.spec_from_file_location("whisper_segments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tokenizer():
    return og.Tokenizer(str(Path(__file__).resolve().parents[1] / "models" / "whisper"))


@pytest.fixture
def whisper_example(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "examples" / "python"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("whisper_example", directory / "whisper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_segments_preserve_text_and_times(segment_example, tokenizer):
    begin = tokenizer.timestamp_begin_token_id
    first = tokenizer.encode(" Hello, world!")
    second = tokenizer.encode("  Caf\u00e9 \u65e5\u672c\u8a9e.")
    tokens = [begin, *first, begin + 35, begin + 35, *second, begin + 100, tokenizer.eos_token_ids[0]]

    segments = segment_example.extract_segments(tokenizer, np.array(tokens, dtype=np.int32))

    assert [(segment.text, segment.start_time, segment.stop_time) for segment in segments] == [
        (tokenizer.decode(first), 0.0, 0.7),
        (tokenizer.decode(second), 0.7, 2.0),
    ]


def test_segment_gap_and_sequence_isolation(segment_example, tokenizer):
    begin = tokenizer.timestamp_begin_token_id
    text = tokenizer.encode(" One window.")
    first = segment_example.extract_segments(tokenizer, [begin, *text, begin + 10])
    second = segment_example.extract_segments(tokenizer, [begin + 20, *text, begin + 30])

    assert first[0].start_time == 0.0
    assert first[0].stop_time == 0.2
    assert second[0].start_time == 0.4
    assert second[0].stop_time == 0.6
    assert len(first) == len(second) == 1


@pytest.mark.parametrize("kind", ["empty", "eos", "timestamps", "repeated_eos"])
def test_no_text_produces_no_segments(segment_example, tokenizer, kind):
    begin = tokenizer.timestamp_begin_token_id
    eos = tokenizer.eos_token_ids[0]
    cases = {"empty": [], "eos": [eos], "timestamps": [begin, begin + 1], "repeated_eos": [begin, eos, eos]}
    assert segment_example.extract_segments(tokenizer, cases[kind]) == []


@pytest.mark.parametrize("ending", [[], ["eos"]])
def test_incomplete_text_rejects_entire_result(segment_example, tokenizer, ending):
    begin = tokenizer.timestamp_begin_token_id
    text = tokenizer.encode(" Complete.")
    suffix = [tokenizer.eos_token_ids[0]] if ending else []
    tokens = [begin, *text, begin + 10, begin + 10, *tokenizer.encode(" Incomplete"), *suffix]

    with pytest.raises(ValueError, match="no closing timestamp"):
        segment_example.extract_segments(tokenizer, tokens)


def test_each_segment_needs_its_own_opening_boundary(segment_example, tokenizer):
    begin = tokenizer.timestamp_begin_token_id
    text = tokenizer.encode(" Text.")
    with pytest.raises(ValueError, match="opening timestamp"):
        segment_example.extract_segments(tokenizer, [begin, *text, begin + 10, *text, begin + 20])


def test_text_before_a_boundary_is_rejected(segment_example, tokenizer):
    with pytest.raises(ValueError, match="opening timestamp"):
        segment_example.extract_segments(tokenizer, tokenizer.encode(" No boundary."))


@pytest.mark.parametrize("with_text", [False, True])
def test_decreasing_timestamps_are_rejected(segment_example, tokenizer, with_text):
    begin = tokenizer.timestamp_begin_token_id
    text = tokenizer.encode(" Text.") if with_text else []
    with pytest.raises(ValueError, match="monotonic"):
        segment_example.extract_segments(tokenizer, [begin + 10, *text, begin])


def test_zero_duration_text_is_rejected(segment_example, tokenizer):
    begin = tokenizer.timestamp_begin_token_id
    with pytest.raises(ValueError, match="end after"):
        segment_example.extract_segments(tokenizer, [begin, *tokenizer.encode(" Text."), begin])


@pytest.mark.parametrize("after_eos", ["text", "timestamp", "control"])
def test_non_eos_content_after_eos_is_rejected(segment_example, tokenizer, after_eos):
    tokens = {
        "text": tokenizer.encode(" Text."),
        "timestamp": [tokenizer.timestamp_begin_token_id],
        "control": [tokenizer.to_token_id("<|fr|>")],
    }
    with pytest.raises(ValueError, match="after EOS"):
        segment_example.extract_segments(tokenizer, [tokenizer.eos_token_ids[0], *tokens[after_eos]])


@pytest.mark.parametrize("value", [True, 50364.0, "50364"])
def test_invalid_token_types_are_rejected(segment_example, tokenizer, value):
    with pytest.raises(TypeError):
        segment_example.extract_segments(tokenizer, [value])


@pytest.mark.parametrize("value", [-1, 51865, 0x80000000])
def test_invalid_token_ids_are_rejected(segment_example, tokenizer, value):
    with pytest.raises(ValueError, match="token"):
        segment_example.extract_segments(tokenizer, [value])


def test_control_tokens_are_not_silently_decoded(segment_example, tokenizer):
    begin = tokenizer.timestamp_begin_token_id
    tokens = [begin, *tokenizer.encode(" Text"), tokenizer.to_token_id("<|fr|>"), begin + 10]
    with pytest.raises(ValueError, match="control"):
        segment_example.extract_segments(tokenizer, tokens)


def test_prompt_must_be_removed(segment_example, tokenizer):
    prompt = tokenizer.encode("<|startoftranscript|><|en|><|transcribe|>")
    begin = tokenizer.timestamp_begin_token_id
    generated = [begin, *tokenizer.encode(" Text."), begin + 10]
    with pytest.raises(ValueError, match="control"):
        segment_example.extract_segments(tokenizer, [*prompt, *generated])
    assert len(segment_example.extract_segments(tokenizer, generated)) == 1


def test_tokenizer_without_timestamp_metadata_is_rejected(segment_example):
    model = Path(__file__).resolve().parents[1] / "models" / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    with pytest.raises(ValueError, match="timestamp token metadata"):
        segment_example.extract_segments(og.Tokenizer(str(model)), [])


def test_example_serializes_empty_segments(whisper_example, tokenizer):
    prompt = tokenizer.encode("<|startoftranscript|><|en|><|transcribe|>")
    output = whisper_example.segment_output(tokenizer, [*prompt, tokenizer.eos_token_ids[0]], prompt, 1, 4)
    assert output == {"batch_index": 1, "beam_index": 4, "time_reference": "audio_window", "segments": []}


def test_example_excludes_prompt_timestamps(whisper_example, tokenizer):
    prompt = tokenizer.encode("<|startoftranscript|><|en|><|transcribe|><|5.00|>")
    begin = tokenizer.timestamp_begin_token_id
    text = tokenizer.encode(" Current window.")
    output = whisper_example.segment_output(tokenizer, [*prompt, begin, *text, begin + 10], prompt, 0, 0)
    assert output["segments"] == [{"text": tokenizer.decode(text), "start_time": 0.0, "stop_time": 0.2}]


def test_example_accepts_native_arrays(whisper_example, tokenizer):
    prompt = tokenizer.encode("<|startoftranscript|><|en|><|transcribe|>")
    begin = tokenizer.timestamp_begin_token_id
    text = tokenizer.encode(" Text.")
    tokens = np.array([*prompt, begin, *text, begin + 10], dtype=np.int32)
    output = whisper_example.segment_output(tokenizer, tokens, prompt, 0, 1)
    assert output["segments"] == [{"text": tokenizer.decode(text), "start_time": 0.0, "stop_time": 0.2}]


def test_example_reports_prompt_mismatch_with_hypothesis(whisper_example, tokenizer):
    with pytest.raises(ValueError, match=r"batch 1, beam 4:.*expected decoder prompt"):
        whisper_example.segment_output(tokenizer, [], tokenizer.encode("<|startoftranscript|>"), 1, 4)


def test_example_reports_incomplete_hypothesis(whisper_example, tokenizer):
    prompt = tokenizer.encode("<|startoftranscript|><|en|><|transcribe|>")
    tokens = [*prompt, tokenizer.timestamp_begin_token_id, *tokenizer.encode(" Incomplete")]
    with pytest.raises(ValueError, match=r"batch 0, beam 2:.*no closing timestamp"):
        whisper_example.segment_output(tokenizer, tokens, prompt, 0, 2)
