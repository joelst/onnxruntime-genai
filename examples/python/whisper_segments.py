# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import operator
from collections.abc import Iterable
from dataclasses import dataclass

import onnxruntime_genai as og


@dataclass(frozen=True)
class WhisperSegment:
    text: str
    start_time: float
    stop_time: float


def extract_segments(tokenizer: og.Tokenizer, tokens: Iterable[int]) -> list[WhisperSegment]:
    """Extract explicitly bounded segments from a completed, prompt-free Whisper suffix.

    This example requires the standard Whisper vocabulary: text IDs precede EOS,
    control IDs follow EOS, and timestamp IDs form the vocabulary suffix. Times
    are relative to one audio window, not absolute acoustic frame intervals.
    """
    if not tokenizer.has_timestamp_tokens:
        raise ValueError("Whisper timestamp token metadata is required.")
    eos_tokens = tokenizer.eos_token_ids
    if len(eos_tokens) != 1:
        raise ValueError("Exactly one Whisper EOS token is required.")
    eos = eos_tokens[0]
    if eos != tokenizer.to_token_id("<|endoftext|>") or not 0 < eos < tokenizer.timestamp_begin_token_id:
        raise ValueError("The standard Whisper vocabulary layout is required.")

    segments = []
    text_tokens = []
    start = None
    last_timestamp = None
    finished = False
    for value in tokens:
        if isinstance(value, bool):
            raise TypeError("Token IDs must be integers, not booleans.")
        token = operator.index(value)
        if not 0 <= token <= 0x7FFFFFFF:
            raise ValueError(f"Invalid token ID: {token}.")
        if token == eos:
            if text_tokens:
                raise ValueError("Text has no closing timestamp before EOS.")
            finished = True
            continue
        if finished:
            raise ValueError("Only repeated EOS tokens are allowed after EOS.")
        if tokenizer.is_timestamp_token(token):
            if last_timestamp is not None and token < last_timestamp:
                raise ValueError("Timestamp boundaries must be monotonic.")
            last_timestamp = token
            if text_tokens:
                if start is None or token <= start:
                    raise ValueError("A text segment must end after its start timestamp.")
                segments.append(
                    WhisperSegment(
                        tokenizer.decode(text_tokens),
                        tokenizer.timestamp_to_seconds(start),
                        tokenizer.timestamp_to_seconds(token),
                    )
                )
                text_tokens = []
                start = None
            else:
                start = token
        elif token >= eos:
            raise ValueError(f"Unexpected control or out-of-vocabulary token: {token}.")
        else:
            if start is None:
                raise ValueError("Text requires an opening timestamp for each segment.")
            text_tokens.append(token)

    if text_tokens:
        raise ValueError("Text has no closing timestamp at the end of the sequence.")
    return segments
