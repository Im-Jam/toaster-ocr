"""Download pinned upstream model artifacts via huggingface_hub."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .paths import MODEL_SPECS, model_dir


def download_models(components: Iterable[str] | None = None) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    selected = list(components) if components else list(MODEL_SPECS.keys())
    out: dict[str, Path] = {}
    for name in selected:
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown component {name!r}; expected one of {list(MODEL_SPECS)}")
        spec = MODEL_SPECS[name]
        dest = model_dir(name)
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_dir=str(dest),
        )
        out[name] = dest
    return out
