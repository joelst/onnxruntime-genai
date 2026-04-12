# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import functools
import json
import os
import shutil
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--test_models",
        help="Path to the current working directory",
        type=str,
        required=True,
    )


def get_path_for_model(data_path, model_name, precision, device):
    model_path = os.path.join(data_path, model_name, precision, device)
    if not os.path.exists(model_path):
        pytest.skip(f"Model {model_name} not found at {model_path}")
    return model_path


@pytest.fixture
def phi2_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "phi-2",
        "int4",
    )


@pytest.fixture
def phi3_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "phi-3-mini",
        "int4",
    )


@pytest.fixture
def phi4_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "phi-4-mini",
        "int4",
    )


@pytest.fixture
def gemma_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "gemma",
        "int4",
    )


@pytest.fixture
def llama_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "llama",
        "int4",
    )


@pytest.fixture
def qwen_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "qwen-2.5-0.5b",
        "int4",
    )


@pytest.fixture
def path_for_model(request):
    return functools.partial(get_path_for_model, request.config.getoption("--test_models"))


@pytest.fixture
def nemotron_speech_model_path(request):
    """Return the path to a nemotron_speech model directory, or skip if not available."""
    test_data = request.config.getoption("--test_models")
    model_path = os.path.join(test_data, "nemotron-speech-streaming")
    if not os.path.exists(model_path):
        pytest.skip(f"Nemotron speech model not found at {model_path}")
    return model_path


@pytest.fixture
def test_data_path(request):
    return request.config.getoption("--test_models")


_GEMMA4_MODEL_DIR_NAME = "gemma4-kv-sharing-preprocessing"


@pytest.fixture
def gemma4_model_path(test_data_path):
    path = os.fspath(Path(test_data_path) / _GEMMA4_MODEL_DIR_NAME)
    if not os.path.exists(path):
        pytest.skip(f"Gemma 4 test model not found at {path}")
    return path


@pytest.fixture
def gemma4_vlm_path(test_data_path, tmp_path):
    """
    Create a minimal genai_config.json with type='gemma4' (not 'gemma4_text') so that
    the C++ runtime tries to create a MultiModalProcessor for it.
    """
    try:
        import onnxruntime_genai as og  # noqa: F401 — just check it's available
    except ImportError:
        pytest.skip("onnxruntime_genai not installed")

    src = Path(test_data_path) / _GEMMA4_MODEL_DIR_NAME
    if not src.exists():
        pytest.skip(f"Gemma 4 test model not found at {src}")

    dest = tmp_path / "gemma4-vlm"
    shutil.copytree(src, dest)

    config_path = dest / "genai_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["model"]["type"] = "gemma4"
    with open(config_path, "w") as f:
        json.dump(cfg, f)

    return os.fspath(dest)
