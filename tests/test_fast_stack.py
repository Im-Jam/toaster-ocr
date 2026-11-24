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
from unittest.mock import MagicMock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toaster_ocr.fast_stack import (
    filter_formula_boxes,
    formula_conf_for_page,
    is_textlayer_page,
    splice_formulas_into_markdown,
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


if __name__ == "__main__":
    unittest.main()
