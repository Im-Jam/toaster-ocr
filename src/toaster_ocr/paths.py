"""Model cache paths and pinned upstream Hugging Face / LiteParse model revisions."""
from __future__ import annotations

import os
from pathlib import Path

OCRIA_NANO_PACKAGE = "ocria-nano"

MODEL_SPECS = {
    "indra": {
        "repo_id": "datalab-to/indra-ocr",
        "revision": "bd02167fdf2005a19962cc15188f8496a2c87d3b",
        "license": "Modified AI Pubs OpenRAIL-M (weights) / Apache-2.0 (indra-ocr code)",
    },
    "slanet_plus": {
        "repo_id": "PaddlePaddle/SLANet_plus_onnx",
        "revision": "7dbe640e127602bf506815e822c09758de73c482",
        "files": ("inference.onnx", "inference.yml"),
        "license": "Apache-2.0",
    },
    "deepdoc_tsr": {
        "repo_id": "InfiniFlow/deepdoc",
        "revision": "9c7aa2c730d7a242d7f04cf6109a6a239aefc717",
        "files": ("tsr.onnx",),
        "license": "Apache-2.0",
    },
    "pp_doclayout_s": {
        "repo_id": "stefanj0/PP-DocLayout-S-ONNX",
        "revision": "7e851b9e5ad66613d7318a3213ab3336ebe0a514",
        "files": ("pp_doclayout_s.onnx",),
        "license": "Apache-2.0",
    },
    "pix2text_mfr15": {
        "repo_id": "breezedeus/pix2text-mfr-1.5",
        "revision": "1cef9f0bdcd6a4c63df7de1311fb0894593340cc",
        "files": ("encoder_model.onnx", "decoder_model.onnx", "tokenizer.json"),
        "license": "MIT",
    },
}

DOWNLOAD_COMPONENTS = (
    "ocria_nano",
    "pp_doclayout_s",
    "pix2text_mfr15",
    "indra",
    "slanet_plus",
    "deepdoc_tsr",
)


def cache_home() -> Path:
    env = os.environ.get("TOASTER_OCR_HOME")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "toaster-ocr"


def indra_dir() -> Path:
    env = os.environ.get("TOASTER_OCR_INDRA_DIR")
    return Path(env).expanduser() if env else cache_home() / "indra"


def stack_models_dir() -> Path:
    env = os.environ.get("TOASTER_OCR_MODELS_DIR")
    return Path(env).expanduser() if env else cache_home() / "models"
