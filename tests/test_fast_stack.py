"""Unit tests for LiteParse+MFR splicing, formula gating, and HTTP server."""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
import unittest
import urllib.request
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toaster_ocr.engine import ToasterOCR
from toaster_ocr.fast_stack import (
    filter_formula_boxes,
    formula_conf_for_page,
    is_textlayer_page,
    splice_formulas_into_markdown,
)
from toaster_ocr.moe import (
    is_valid_display_latex,
    should_prefer_ocria_table,
    splice_tables_jaccard,
)
from toaster_ocr.server import make_handler


@dataclass
class _BBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class _Block:
    kind: str
    bbox: _BBox
    text: str = ""
    text_item_indices: list[int] | None = None


@dataclass
class _TextItem:
    text: str
    font_name: str = "Times"


class TestFastStack(unittest.TestCase):
    def test_textlayer_and_formula_threshold(self) -> None:
        pdf_tis = [_TextItem("a", "F1") for _ in range(5)]
        scan_tis = [_TextItem("a", "OCR") for _ in range(5)]
        short_tis = [_TextItem("a", "F1"), _TextItem("b", "F1")]
        self.assertTrue(is_textlayer_page(pdf_tis))
        self.assertFalse(is_textlayer_page(scan_tis))
        self.assertFalse(is_textlayer_page(short_tis))
        self.assertEqual(formula_conf_for_page(True), 0.35)
        self.assertEqual(formula_conf_for_page(False), 0.28)

    def test_filter_formula_boxes_skips_tables(self) -> None:
        table_blk = _Block("merged_table", _BBox(50, 50, 200, 100))
        dets = [
            ("formula", 0.90, [120.0, 120.0, 320.0, 180.0]),  # inside table at 2x scale -> skipped
            ("formula", 0.90, [120.0, 400.0, 320.0, 460.0]),  # outside table -> kept
            ("formula", 0.20, [120.0, 500.0, 320.0, 560.0]),  # below conf -> skipped
        ]
        f_px, f_pdf = filter_formula_boxes(
            dets,
            [table_blk],
            img_h=1000,
            scale_x=2.0,
            scale_y=2.0,
            conf_thresh=0.28,
        )
        self.assertEqual(len(f_px), 1)
        self.assertEqual(f_pdf[0], [60.0, 200.0, 160.0, 230.0])

    def test_splice_formulas_preserves_tables_and_inline_math(self) -> None:
        # 1. On scanned pages, standalone garbled OCR block is replaced by $$...$$
        scan_blk = _Block("paragraph", _BBox(20, 100, 180, 30), text="6. a and 5 a.")
        tbl_blk = _Block("merged_table", _BBox(20, 200, 180, 100), text="")
        md_scan = "<table><tr><th>A</th><td>1</td></tr></table>\n\n6. a and 5 a."
        out_scan = splice_formulas_into_markdown(
            md_scan,
            [tbl_blk, scan_blk],
            [],
            [[18.0, 98.0, 202.0, 132.0]],
            [r"\frac{6a}{5}"],
            is_textlayer=False,
        )
        self.assertIn("<table><tr><th>A</th><td>1</td></tr></table>", out_scan)
        self.assertIn("$$\n\\frac{6a}{5}\n$$", out_scan)
        self.assertNotIn("6. a and 5 a.", out_scan)

        # 2. On born-digital PDF text-layer pages, inline $math$ in prose is preserved
        pdf_blk = _Block("paragraph", _BBox(20, 100, 180, 40), text="Let $x > 0$ satisfy V = sum")
        md_pdf = "Let $x > 0$ satisfy V = sum"
        out_pdf = splice_formulas_into_markdown(
            md_pdf,
            [pdf_blk],
            [],
            [[20.0, 100.0, 200.0, 140.0]],
            [r"V = \bigoplus_i W_i"],
            is_textlayer=True,
        )
        self.assertIn("Let $x > 0$ satisfy", out_pdf)
        self.assertIn("$$\nV = \\bigoplus_i W_i\n$$", out_pdf)

    def test_layout_blocks_mapping(self) -> None:
        @dataclass
        class _Cell:
            text: str

        @dataclass
        class _TblBlock:
            kind: str
            bbox: _BBox
            text: str | None = None
            header: list[_Cell] | None = None
            rows: list[list[_Cell]] | None = None

        @dataclass
        class _Page:
            width: float
            height: float
            blocks: list[Any]

        pg = _Page(
            width=200.0,
            height=400.0,
            blocks=[
                _TblBlock("merged_table", _BBox(20, 40, 100, 80), header=[_Cell("H1")], rows=[[_Cell("C1")]]),
                _TblBlock("page_header", _BBox(0, 0, 200, 20), text="Header"),
                _TblBlock("page_footer", _BBox(0, 380, 200, 20), text="Page 1"),
                _TblBlock("heading", _BBox(20, 20, 100, 20), text="Title"),
                _TblBlock("list_item", _BBox(20, 140, 100, 20), text="Item"),
            ],
        )
        got = ToasterOCR._layout_blocks(pg)
        self.assertEqual([b["label"] for b in got], ["table", "page_furniture", "page_furniture", "heading", "list"])
        self.assertEqual(got[0]["text"], "H1 C1")
        self.assertEqual(got[0]["bbox"], [0.1, 0.1, 0.6, 0.3])

    def test_server_endpoints(self) -> None:
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.markdown = "# Hello OCR"
        mock_result.pages = ["# Hello OCR"]
        mock_result.elapsed_seconds = 0.12
        mock_result.stats = {"formula_boxes": 0}
        mock_engine.read_image.return_value = mock_result

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(mock_engine, api_key="secret-k"))
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                self.assertEqual(r.status, 200)
                self.assertEqual(json.loads(r.read())["status"], "ok")

            buf = io.BytesIO()
            Image.new("RGB", (32, 32), color=(255, 255, 255)).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            body = json.dumps({
                "model": "toaster-ocr",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
            }).encode("utf-8")

            req_unauth = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_unauth, timeout=5)
            self.assertEqual(ctx.exception.code, 401)

            req_auth = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer secret-k"},
            )
            with urllib.request.urlopen(req_auth, timeout=5) as r:
                resp = json.loads(r.read())
            self.assertEqual(resp["choices"][0]["message"]["content"], "# Hello OCR")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_moe_table_splicer_and_arbiter(self) -> None:
        base_md = (
            "Intro\n\n<table><tr><th>Age</th><th>$I V P$</th></tr><tr><td>Mean<br>SD</td><td>10.2<br>3.1</td></tr></table>\n\n"
            "Middle\n\n<table><tr><th>Other</th><th>Val</th></tr><tr><td>X</td><td>9</td></tr></table>"
        )
        donor_md = (
            "Intro\n\n<table><tr><th>Age</th><th>IVP</th></tr><tr><td>Mean</td><td>10.2</td></tr><tr><td>SD</td><td>3.1</td></tr></table>"
        )
        spliced = splice_tables_jaccard(base_md, donor_md)
        self.assertIsNotNone(spliced)
        assert spliced is not None
        # Must preserve both base slots (max(2, 1) == 2 tables) and replace the matching first slot
        self.assertEqual(spliced.count("<table"), 2)
        self.assertIn("<td>SD</td><td>3.1</td>", spliced)
        self.assertIn("<td>X</td><td>9</td>", spliced)
        self.assertTrue(should_prefer_ocria_table(base_md, donor_md))
        self.assertFalse(should_prefer_ocria_table(base_md, donor_md, lp_is_garbled=True))

    def test_display_latex_verifier(self) -> None:
        self.assertTrue(is_valid_display_latex(r"$$\delta = \frac{\rho_1 + \rho_2}{\omega_2}$$"))
        self.assertFalse(is_valid_display_latex(r"$$x^{2 + y{2 = 2ax$$"))
        self.assertFalse(is_valid_display_latex("$$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~$$"))


if __name__ == "__main__":
    unittest.main()
