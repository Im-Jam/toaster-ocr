"""toaster-ocr: Fast pure-ONNX CPU document OCR (v0.4 System MoE)."""
from __future__ import annotations

from .download import download_models
from .engine import OCRResult, ToasterOCR
from .fast_stack import PPDocLayoutSONNX, Pix2TextMFRKVONNX
from .moe import should_prefer_ocria_table, splice_tables_jaccard

__all__ = [
    "OCRResult",
    "PPDocLayoutSONNX",
    "Pix2TextMFRKVONNX",
    "ToasterOCR",
    "download_models",
    "should_prefer_ocria_table",
    "splice_tables_jaccard",
]
__version__ = "0.4.0"
