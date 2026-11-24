"""toaster-ocr: fast pure-ONNX CPU document OCR."""
from __future__ import annotations

from .download import download_models
from .engine import OCRResult, ToasterOCR
from .fast_stack import PPDocLayoutSONNX, Pix2TextMFRKVONNX
from .paths import MODEL_SPECS, cache_home, stack_models_dir

__version__ = "0.3.0"

__all__ = [
    "MODEL_SPECS",
    "OCRResult",
    "PPDocLayoutSONNX",
    "Pix2TextMFRKVONNX",
    "ToasterOCR",
    "__version__",
    "cache_home",
    "download_models",
    "stack_models_dir",
]
