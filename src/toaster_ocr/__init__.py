"""toaster-ocr: pure-ONNX fast-CPU document OCR."""
from __future__ import annotations

from .download import download_models
from .engine import OCRResult, ToasterOCR
from .fast_stack import FastCPUStack
from .paths import MODEL_SPECS, cache_home, indra_dir, stack_models_dir

__version__ = "0.2.0"

__all__ = [
    "FastCPUStack",
    "MODEL_SPECS",
    "OCRResult",
    "ToasterOCR",
    "__version__",
    "cache_home",
    "download_models",
    "indra_dir",
    "stack_models_dir",
]
