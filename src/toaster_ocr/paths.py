"""Configurable model directory resolution (no hardcoded home paths)."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_SPECS = {
    "indra": {
        "repo_id": "datalab-to/indra-ocr",
        "revision": "bd02167fdf2005a19962cc15188f8496a2c87d3b",
        "subdir": "indra",
        "env": "TOASTER_OCR_INDRA_DIR",
    },
    "ppocrv6": {
        "repo_id": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        "revision": "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
        "subdir": "ppocrv6",
        "env": "TOASTER_OCR_PPV6_DIR",
    },
    "ppv3": {
        "repo_id": "PaddlePaddle/PP-DocLayoutV3",
        "revision": "main",
        "subdir": "ppv3",
        "env": "TOASTER_OCR_PPV3_DIR",
    },
    "slanet": {
        "repo_id": "dimtri009/SLANet-1M",
        "revision": "a8ea21bf31cebc0f04909ede8517e3029c32f19b",
        "subdir": "slanet",
        "env": "TOASTER_OCR_SLANET_DIR",
    },
}


def cache_home() -> Path:
    env = os.environ.get("TOASTER_OCR_HOME")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "toaster-ocr"


def model_dir(name: str) -> Path:
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model component: {name}")
    spec = MODEL_SPECS[name]
    override = os.environ.get(spec["env"])
    if override:
        return Path(override).expanduser()
    return cache_home() / spec["subdir"]
