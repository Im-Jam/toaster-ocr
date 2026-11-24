"""High-level `ToasterOCR` engine combining LiteParse (OCRia-nano-1 + LayoutNet) with PP-DocLayout-S + QInt8 KV-MFR."""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .fast_stack import (
    PPDocLayoutSONNX,
    Pix2TextMFRKVONNX,
    filter_formula_boxes,
    formula_conf_for_page,
    is_textlayer_page,
    splice_formulas_into_markdown,
)
from .paths import stack_models_dir


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
        models_root: str | Path | None = None,
        layout_dpi: float = 200.0,
        liteparse_dpi: float = 150.0,
        keep_headers_footers: bool = True,
    ):
        from liteparse import LiteParse

        m_root = Path(models_root) if models_root else stack_models_dir()
        dl_path = m_root / "pp_doclayout_s.onnx"
        mfr_dir = m_root / "pix2text_mfr15"
        if not dl_path.is_file() or not (mfr_dir / "mfr_dec_step.onnx").is_file():
            raise FileNotFoundError(
                f"toaster-ocr models not found in {m_root}. Run `toaster-ocr download` first."
            )

        self.layout_dpi = float(layout_dpi)
        self.liteparse_dpi = float(liteparse_dpi)
        self.keep_headers_footers = bool(keep_headers_footers)
        self.layout = PPDocLayoutSONNX(dl_path, threads=threads)
        self.mfr = Pix2TextMFRKVONNX(mfr_dir, threads=threads)
        self._lp = LiteParse(
            ocr_backend="ocria-nano",
            output_format="markdown",
            extract_blocks=True,
            keep_headers_footers=self.keep_headers_footers,
            extract_links=False,
            dpi=self.liteparse_dpi,
            num_workers=1,
            quiet=True,
        )

    def _enhance_page(
        self,
        pg: Any,
        gray: np.ndarray,
        *,
        scale_x: float,
        scale_y: float,
    ) -> tuple[str, int, int, bool]:
        H, W = gray.shape[:2]
        textlayer = is_textlayer_page(pg.text_items)
        f_conf = formula_conf_for_page(textlayer)
        dets = self.layout.detect(gray, conf_thresh=0.28)
        f_px, f_pdf = filter_formula_boxes(
            dets,
            pg.blocks,
            img_h=H,
            scale_x=scale_x,
            scale_y=scale_y,
            conf_thresh=f_conf,
        )
        if not f_px:
            return pg.markdown, 0, 0, textlayer

        crops = [
            gray[
                max(0, int(np.floor(y0 - 6))) : min(H, int(np.ceil(y1 + 6))),
                max(0, int(np.floor(x0 - 6))) : min(W, int(np.ceil(x1 + 6))),
            ]
            for x0, y0, x1, y1 in f_px
        ]
        latex_list = self.mfr.recognize_batch(crops)
        n_replaced = sum(1 for lx in latex_list if lx and lx.strip())
        md = splice_formulas_into_markdown(
            pg.markdown,
            pg.blocks,
            pg.text_items,
            f_pdf,
            latex_list,
            is_textlayer=textlayer,
        )
        return md, len(f_px), n_replaced, textlayer

    def _parse_file(self, path: Path, *, max_pages: int = 0, password: str | None = None) -> Any:
        if not max_pages and password is None:
            return self._lp.parse(str(path))
        from liteparse import LiteParse

        lp = LiteParse(
            ocr_backend="ocria-nano",
            output_format="markdown",
            extract_blocks=True,
            keep_headers_footers=self.keep_headers_footers,
            extract_links=False,
            dpi=self.liteparse_dpi,
            num_workers=1,
            quiet=True,
            max_pages=max_pages or None,
            password=password,
        )
        return lp.parse(str(path))

    def read_image(
        self,
        image: str | Path | Image.Image,
        *,
        baseline: bool = False,
        enable_math_mfr: bool = True,
    ) -> OCRResult:
        t0 = time.perf_counter()
        tmp_path: Path | None = None
        if isinstance(image, Image.Image):
            pil = image.convert("L")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_path = Path(tf.name)
            image.convert("RGB").save(tmp_path, format="PNG")
            in_path = tmp_path
        else:
            in_path = Path(image)
            pil = Image.open(in_path).convert("L")

        try:
            res = self._parse_file(in_path)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        pg = res.pages[0]
        if baseline or not enable_math_mfr:
            md = pg.markdown
            stats = {"backend": "liteparse+ocria-nano", "pages": 1, "formula_boxes": 0, "formulas_replaced": 0}
        else:
            gray = np.asarray(pil, dtype=np.uint8)
            H, W = gray.shape[:2]
            sx = W / max(1e-6, float(pg.width))
            sy = H / max(1e-6, float(pg.height))
            md, f_boxes, f_rep, textlayer = self._enhance_page(pg, gray, scale_x=sx, scale_y=sy)
            stats = {
                "backend": "liteparse+ocria-nano+mfr",
                "pages": 1,
                "formula_boxes": f_boxes,
                "formulas_replaced": f_rep,
                "textlayer": textlayer,
            }

        return OCRResult(
            markdown=md,
            pages=[md],
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            stats=stats,
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
        import pypdfium2 as pdfium

        t0 = time.perf_counter()
        p = Path(pdf_path)
        res = self._parse_file(p, max_pages=max_pages, password=password)

        if baseline or not enable_math_mfr:
            pages = [pg.markdown for pg in res.pages]
            return OCRResult(
                markdown="\n\n---\n\n".join(pages),
                pages=pages,
                elapsed_seconds=round(time.perf_counter() - t0, 3),
                stats={
                    "backend": "liteparse+ocria-nano",
                    "pages": len(pages),
                    "formula_boxes": 0,
                    "formulas_replaced": 0,
                },
            )

        scale = self.layout_dpi / 72.0
        pages: list[str] = []
        tot_boxes = 0
        tot_replaced = 0
        textlayer_pages = 0
        pdf = pdfium.PdfDocument(str(p), password=password)
        try:
            for idx, pg in enumerate(res.pages):
                bmp = pdf[idx].render(scale=scale)
                gray = np.asarray(bmp.to_pil().convert("L"), dtype=np.uint8)
                md, f_boxes, f_rep, textlayer = self._enhance_page(
                    pg, gray, scale_x=scale, scale_y=scale
                )
                pages.append(md)
                tot_boxes += f_boxes
                tot_replaced += f_rep
                textlayer_pages += int(textlayer)
        finally:
            pdf.close()

        return OCRResult(
            markdown="\n\n---\n\n".join(pages),
            pages=pages,
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            stats={
                "backend": "liteparse+ocria-nano+mfr",
                "pages": len(pages),
                "formula_boxes": tot_boxes,
                "formulas_replaced": tot_replaced,
                "textlayer_pages": textlayer_pages,
            },
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
