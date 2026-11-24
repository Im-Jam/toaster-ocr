"""Unit tests for FastCPUStack table matching, textlayer trust fix, and HTTP server."""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toaster_ocr.fast_stack import (
    _promote_first_row_to_th,
    formula_conf_for_rec,
    install_textlayer_trust_fix,
    match_and_build_slanet_html,
)
from toaster_ocr.server import make_handler


class TestFastStack(unittest.TestCase):
    def test_promote_first_row_to_th(self) -> None:
        html = "<table><tr><td>Col A</td><td>Col B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        out = _promote_first_row_to_th(html)
        self.assertIn("<th>Col A</th><th>Col B</th>", out)
        self.assertIn("<td>1</td><td>2</td>", out)
        # Idempotent if th already exists
        self.assertEqual(_promote_first_row_to_th(out), out)

    def test_match_and_build_slanet_html_conserves_all_fragments(self) -> None:
        pred_struct = [
            "<table>", "<tbody>",
            "<tr>", "<td></td>", "<td></td>", "</tr>",
            "<tr>", "<td></td>", "<td></td>", "</tr>",
            "</tbody>", "</table>",
        ]
        cell_bboxes = [
            [0, 0, 50, 20],
            [50, 0, 100, 20],
            [0, 20, 50, 40],
            [50, 20, 100, 40],
        ]
        boxes = np.array(
            [
                [2, 2, 48, 18],
                [52, 2, 98, 18],
                [2, 22, 48, 38],
                [52, 22, 98, 38],
            ],
            dtype=np.float32,
        )
        texts = ["Header A", "Header B", "Val 1", "Val 2"]
        html_out, iou_rate, horiz_merge_rate = match_and_build_slanet_html(pred_struct, cell_bboxes, boxes, texts)
        for t in texts:
            self.assertIn(t, html_out)
        self.assertEqual(iou_rate, 1.0)
        self.assertEqual(horiz_merge_rate, 0.0)
        self.assertIn("<th>Header A</th>", html_out)

    def test_formula_conf_for_rec(self) -> None:
        textlayer_rec = [("a", 1.0, []), ("b", 1.0, []), ("c", 1.0, []), ("d", 1.0, [])]
        scan_rec = [("a", 0.85, []), ("b", 0.91, []), ("c", 0.78, []), ("d", 0.92, [])]
        short_rec = [("a", 1.0, []), ("b", 1.0, [])]
        self.assertEqual(formula_conf_for_rec(textlayer_rec), 0.35)
        self.assertEqual(formula_conf_for_rec(scan_rec), 0.28)
        self.assertEqual(formula_conf_for_rec(short_rec), 0.28)

    def test_install_textlayer_trust_fix(self) -> None:
        from indra_ocr import textlayer

        install_textlayer_trust_fix()
        self.assertEqual(textlayer.MAX_BAD_CHAR_FRAC, 0.05)
        self.assertEqual(textlayer.MAX_NO_INK_LINE_FRAC, 0.30)
        self.assertEqual(textlayer.DARK, 230)
        # PUA checkbox U+E001 must NOT be counted as bad char; U+FFFD must be
        self.assertEqual(textlayer.bad_char_count("Hello \ue001 world"), 0)
        self.assertEqual(textlayer.bad_char_count("Hello \ufffd world"), 1)

    def test_server_endpoints(self) -> None:
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.markdown = "# Hello OCR"
        mock_result.pages = ["# Hello OCR"]
        mock_result.elapsed_seconds = 0.12
        mock_result.stats = {"tables_replaced": 0}
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

            # Unauthorized check
            req_unauth = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_unauth, timeout=5)
            self.assertEqual(ctx.exception.code, 401)

            # Authorized check
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
