"""Clean high-level wrapper around Indra OrtOCR + FastCPUStack (images and PDFs)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np

from .fast_stack import FastCPUStack
from .paths import indra_dir, stack_models_dir


@dataclass
class OCRResult:
    markdown: str
    pages: list[str]
    elapsed_seconds: float
    stats: dict[str, Any] = field(default_factory=dict)


class ToasterOCR:
    """Pure-ONNX CPU document OCR engine."""

    def __init__(
        self,
        *,
        threads: int = 2,
        indra_model_dir: str | Path | None = None,
        models_root: str | Path | None = None,
        furniture: bool = True,
        math: bool = True,
        dpi: float = 200.0,
    ):
        from indra_ocr import document, pipeline

        i_dir = Path(indra_model_dir) if indra_model_dir else indra_dir()
        m_root = Path(models_root) if models_root else stack_models_dir()
        if not (i_dir / "det.onnx").is_file():
            raise FileNotFoundError(
                f"Indra models not found in {i_dir}. Run `toaster-ocr download` first."
            )
        if not (m_root / "slanet-plus.onnx").is_file():
            raise FileNotFoundError(
                f"FastCPUStack models not found in {m_root}. Run `toaster-ocr download` first."
            )

        self.ocr = pipeline.OrtOCR(
            str(i_dir),
            threads=threads,
            furniture=furniture,
            math=math,
            providers=["CPUExecutionProvider"],
        )
        self.fast_stack = FastCPUStack(self.ocr, threads=threads, models_root=m_root)
        self.converter = document.Converter(self.ocr, dpi=dpi, mode="auto", image_ocr=True)

    def read_image(
        self,
        image: str | Path | Image.Image,
        *,
        baseline: bool = False,
        enable_math_mfr: bool = True,
    ) -> OCRResult:
        t0 = time.perf_counter()
        pil = Image.open(image).convert("L") if not isinstance(image, Image.Image) else image.convert("L")
        gray = np.asarray(pil, dtype=np.uint8)
        self.fast_stack.begin_page(enabled=not baseline, enable_math_mfr=enable_math_mfr)
        res = self.ocr(gray)
        md = res.markdown if getattr(res, "markdown", None) is not None else str(res)
        return OCRResult(
            markdown=md,
            pages=[md],
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            stats=dict(self.fast_stack.page_stats),
        )

    def read_pdf(
        self,
        pdf_path: str | Path,
        *,
        baseline: bool = False,
        enable_math_mfr: bool = True,
        max_pages: int = 0,
        password: str | None = None,
    ) -> OCRResult:
        t0 = time.perf_counter()
        pages: list[str] = []
        last_stats: dict[str, Any] = {}
        self.fast_stack.begin_page(enabled=not baseline, enable_math_mfr=enable_math_mfr)
        for _idx, page in self.converter.convert(str(pdf_path), max_pages=max_pages, password=password):
            pages.append(page.markdown if getattr(page, "markdown", None) is not None else str(page))
            last_stats = dict(self.fast_stack.page_stats)
            self.fast_stack.begin_page(enabled=not baseline, enable_math_mfr=enable_math_mfr)
        md = "\n\n---\n\n".join(pages)
        return OCRResult(
            markdown=md,
            pages=pages,
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            stats=last_stats,
        )

    def read(
        self,
        path: str | Path,
        *,
        baseline: bool = False,
        enable_math_mfr: bool = True,
    ) -> OCRResult:
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            return self.read_pdf(p, baseline=baseline, enable_math_mfr=enable_math_mfr)
        return self.read_image(p, baseline=baseline, enable_math_mfr=enable_math_mfr)
