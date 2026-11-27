"""High-level `ToasterOCR` v0.4 engine supporting Fast (1-Pass Pre-Route), Balanced (Anatomical System MoE), OCRia-nano, and Indra modes."""
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
from .moe import (
    TABLE_RE,
    ensure_x86_rec_latin_qdq_parity,
    should_prefer_ocria_table,
    splice_tables_jaccard,
)
from .paths import indra_dir, stack_models_dir


@dataclass
class OCRResult:
    markdown: str
    pages: list[str]
    elapsed_seconds: float
    stats: dict[str, Any] = field(default_factory=dict)
    layout_blocks: list[list[dict[str, Any]]] = field(default_factory=list)


class ToasterOCR:
    """Pure-ONNX CPU document OCR engine (v0.4).

    Modes:
    - `fast` (1-Pass Pre-Route): PDFs -> LiteParse (OCRia-nano-1 + LayoutNet) + KV-MFR;
      Images -> Indra v0.2.1 + FastCPUStack when installed (falls back to OCRia-nano-1 if Indra weights are absent).
    - `balanced` (Anatomical System MoE v4): combines OCRia-nano-1 and Indra v0.2.1 with
      contrastive DOM/anatomical table arbitration + Jaccard slot-aligned table replacement
      and born-digital math routing.
    - `ocria`: always run LiteParse (OCRia-nano-1) + PP-DocLayout-S + KV-MFR.
    - `indra`: always run Indra v0.2.1 + SLANet-plus + DeepDoc TSR + PP-DocLayout-S + KV-MFR.
    """

    def __init__(
        self,
        *,
        threads: int = 2,
        mode: str = "fast",
        models_root: str | Path | None = None,
        indra_model_dir: str | Path | None = None,
        layout_dpi: float = 200.0,
        liteparse_dpi: float = 200.0,
        keep_headers_footers: bool = True,
    ):
        from liteparse import LiteParse

        if mode not in ("fast", "balanced", "ocria", "indra"):
            raise ValueError(f"Unsupported mode {mode!r}; choose 'fast', 'balanced', 'ocria', or 'indra'.")

        self.mode = mode
        self.threads = int(threads)
        m_root = Path(models_root) if models_root else stack_models_dir()
        i_dir = Path(indra_model_dir) if indra_model_dir else indra_dir()
        dl_path = m_root / "pp_doclayout_s.onnx"
        mfr_dir = m_root / "pix2text_mfr15"
        if not dl_path.is_file() or not (mfr_dir / "mfr_dec_step.onnx").is_file():
            raise FileNotFoundError(
                f"toaster-ocr models not found in {m_root}. Run `toaster-ocr download` first."
            )

        ensure_x86_rec_latin_qdq_parity()
        self.layout_dpi = float(layout_dpi)
        self.liteparse_dpi = float(liteparse_dpi)
        self.keep_headers_footers = bool(keep_headers_footers)
        self.layout = PPDocLayoutSONNX(dl_path, threads=threads)
        self.mfr = Pix2TextMFRKVONNX(mfr_dir, threads=threads)
        self._lp = LiteParse(
            ocr_backend="ocria-nano",
            output_format="markdown",
            extract_blocks=True,
            include_complexity=True,
            keep_headers_footers=self.keep_headers_footers,
            extract_links=False,
            dpi=self.liteparse_dpi,
            num_workers=1,
            quiet=True,
        )

        # Optional Indra + FastCPUStack backend for `fast` (images), `balanced` (MoE), and `indra` modes
        self.indra_ocr = None
        self.indra_stack = None
        self.indra_conv = None
        if mode in ("fast", "balanced", "indra") and (i_dir / "det.onnx").is_file() and (m_root / "slanet-plus.onnx").is_file():
            try:
                from indra_ocr import document as indra_doc, pipeline as indra_pipe
                from .indra_stack import FastCPUStack

                self.indra_ocr = indra_pipe.OrtOCR(
                    str(i_dir),
                    threads=threads,
                    furniture=keep_headers_footers,
                    math=True,
                    providers=["CPUExecutionProvider"],
                )
                self.indra_stack = FastCPUStack(self.indra_ocr, threads=threads, models_root=m_root)
                self.indra_conv = indra_doc.Converter(self.indra_ocr, dpi=layout_dpi, mode="auto", image_ocr=True)
            except ImportError:
                if mode == "indra":
                    raise

    @staticmethod
    def _layout_blocks(pg: Any) -> list[dict[str, Any]]:
        """Expose LiteParse blocks in the OPB `{label, bbox, text}` shape, normalized to page size."""
        width = max(1e-6, float(pg.width))
        height = max(1e-6, float(pg.height))
        labels = {
            "paragraph": "text",
            "text": "text",
            "heading": "heading",
            "section_header": "heading",
            "list": "list",
            "list_item": "list",
            "caption": "caption",
            "figure_caption": "caption",
            "footnote": "footnote",
            "code": "code",
            "table": "table",
            "merged_table": "table",
            "equation": "equation",
            "formula": "equation",
            "form": "form",
            "page_header": "page_furniture",
            "page_footer": "page_furniture",
            "header": "page_furniture",
            "footer": "page_furniture",
            "page_number": "page_furniture",
            "page_furniture": "page_furniture",
            "grid_fallback": "text",
            "rule": "text",
            "figure": "figure",
            "image": "figure",
        }
        out: list[dict[str, Any]] = []
        for block in getattr(pg, "blocks", None) or []:
            box = getattr(block, "bbox", None)
            if box is None:
                continue
            x0 = min(1.0, max(0.0, float(box.x) / width))
            y0 = min(1.0, max(0.0, float(box.y) / height))
            x1 = min(1.0, max(0.0, float(box.x + box.width) / width))
            y1 = min(1.0, max(0.0, float(box.y + box.height) / height))
            text = getattr(block, "text", None)
            if text is None and (getattr(block, "header", None) or getattr(block, "rows", None)):
                parts = []
                if getattr(block, "header", None):
                    parts.extend(c.text for c in block.header if getattr(c, "text", ""))
                if getattr(block, "rows", None):
                    parts.extend(
                        cell.text
                        for row in block.rows
                        for cell in row
                        if getattr(cell, "text", "")
                    )
                text = " ".join(parts)
            if text is None and getattr(block, "lines", None):
                text = " ".join(block.lines)
            out.append({
                "label": labels.get(getattr(block, "kind", ""), "text"),
                "bbox": [x0, y0, x1, y1],
                "text": " ".join(str(text or "").split()),
            })
        return out

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
            include_complexity=True,
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
        # In `fast`, `balanced`, and `indra` modes, images route to Indra v0.2.1 when available
        if self.mode in ("fast", "balanced", "indra") and self.indra_ocr is not None and self.indra_stack is not None:
            pil = Image.open(image).convert("L") if not isinstance(image, Image.Image) else image.convert("L")
            gray = np.asarray(pil, dtype=np.uint8)
            self.indra_stack.begin_page(enabled=not baseline, enable_math_mfr=enable_math_mfr)
            res_ind = self.indra_ocr(gray)
            md_ind = res_ind.markdown if getattr(res_ind, "markdown", None) is not None else str(res_ind)
            return OCRResult(
                markdown=md_ind,
                pages=[md_ind],
                elapsed_seconds=round(time.perf_counter() - t0, 3),
                stats={"backend": "indra-v0.2.1", "mode": self.mode, **dict(self.indra_stack.page_stats)},
            )

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
            layout_blocks=[self._layout_blocks(pg)],
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

        if self.mode == "indra" and self.indra_conv is not None and self.indra_stack is not None:
            pages_ind: list[str] = []
            self.indra_stack.begin_page(enabled=not baseline, enable_math_mfr=enable_math_mfr)
            for _idx, page in self.indra_conv.convert(str(p), max_pages=max_pages, password=password):
                pages_ind.append(page.markdown if getattr(page, "markdown", None) is not None else str(page))
                self.indra_stack.begin_page(enabled=not baseline, enable_math_mfr=enable_math_mfr)
            return OCRResult(
                markdown="\n\n---\n\n".join(pages_ind),
                pages=pages_ind,
                elapsed_seconds=round(time.perf_counter() - t0, 3),
                stats={"backend": "indra-v0.2.1", "mode": "indra", "pages": len(pages_ind)},
            )

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
                layout_blocks=[self._layout_blocks(pg) for pg in res.pages],
            )

        scale = self.layout_dpi / 72.0
        pages = []
        tot_boxes = 0
        tot_replaced = 0
        textlayer_pages = 0
        pdf = pdfium.PdfDocument(str(p), password=password)
        try:
            for idx, pg in enumerate(res.pages):
                bmp = pdf[idx].render(scale=scale)
                gray = np.asarray(bmp.to_pil().convert("L"), dtype=np.uint8)
                ocr_md, f_boxes, f_rep, textlayer = self._enhance_page(
                    pg, gray, scale_x=scale, scale_y=scale
                )
                # In `balanced` (System MoE v4) mode, arbitrate tables & non-math scans with Indra when installed
                if self.mode == "balanced" and self.indra_conv is not None and self.indra_stack is not None:
                    use_ocria_math = bool(textlayer and f_boxes > 0)
                    has_table = "<table" in (ocr_md or "").lower()
                    if not use_ocria_math or has_table:
                        self.indra_stack.begin_page(enabled=True, enable_math_mfr=(not use_ocria_math))
                        pg_ind = self.indra_conv.pdf_page(str(p), pdf, idx)
                        ind_md = (pg_ind.markdown if getattr(pg_ind, "markdown", None) is not None else pg_ind.text) or ""
                        comp = getattr(pg, "complexity", None)
                        is_garbled = bool(getattr(comp, "is_garbled", False)) if comp is not None else False
                        none_cells = 0
                        tot_cells = 0
                        for b in getattr(pg, "blocks", None) or []:
                            if getattr(b, "kind", "") in ("table", "merged_table"):
                                for c in list(getattr(b, "header", None) or []) + [c for r in (getattr(b, "rows", None) or []) for c in r]:
                                    if getattr(c, "text", "").strip():
                                        tot_cells += 1
                                        if getattr(c, "bbox", None) is None:
                                            none_cells += 1
                        none_frac = none_cells / max(1, tot_cells)
                        base_md = ocr_md if use_ocria_math else ind_md
                        if TABLE_RE.search(ind_md) and TABLE_RE.search(ocr_md):
                            pick_ocr = should_prefer_ocria_table(
                                ind_md,
                                ocr_md,
                                lp_is_garbled=is_garbled,
                                ocr_none_bbox_frac=none_frac,
                            )
                            if (not use_ocria_math) and pick_ocr:
                                spliced = splice_tables_jaccard(base_md, ocr_md)
                                if spliced is not None:
                                    base_md = spliced
                            elif use_ocria_math and (not pick_ocr):
                                spliced = splice_tables_jaccard(base_md, ind_md)
                                if spliced is not None:
                                    base_md = spliced
                        ocr_md = base_md

                pages.append(ocr_md)
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
                "backend": "system-moe-v4" if (self.mode == "balanced" and self.indra_conv is not None) else "liteparse+ocria-nano+mfr",
                "mode": self.mode,
                "pages": len(pages),
                "formula_boxes": tot_boxes,
                "formulas_replaced": tot_replaced,
                "textlayer_pages": textlayer_pages,
            },
            layout_blocks=[self._layout_blocks(pg) for pg in res.pages],
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
