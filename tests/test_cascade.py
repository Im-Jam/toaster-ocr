import tempfile
import unittest
from pathlib import Path
from PIL import Image

from toaster_ocr.indra_engine import DEFAULT_PPV6_CONF_THRESHOLD, should_escalate_line
from toaster_ocr.pipeline import ToasterOCR


class _FakeIndra:
    def __init__(self, has_table: bool):
        self.has_table = has_table

    def run_image(self, image_path, upright_out=None, **kwargs):
        Path(upright_out).write_bytes(Path(image_path).read_bytes())
        return {
            "vertical": False,
            "image_path": str(upright_out),
            "width": 16,
            "height": 16,
            "markdown": "<table><tr><td>A</td></tr></table>" if self.has_table else "Plain prose",
            "parts": [
                {
                    "id": "part_0000",
                    "label": "Table" if self.has_table else "Text",
                    "bbox": [0, 0, 16, 16],
                    "markdown": "<table><tr><td>A</td></tr></table>" if self.has_table else "Plain prose",
                    "lines": [],
                    "cells": [{"id": "c0", "line_id": 0, "text": "A", "box": [1, 1, 10, 10]}],
                }
            ],
            "inference_seconds": 0.1,
            "ppv6_seconds": 0.0,
            "ppv6_escalated": 0,
            "ppv6_candidates": 1,
        }


class TestConfidenceCascade(unittest.TestCase):
    def test_skips_math_and_rtl_and_empty(self):
        self.assertFalse(should_escalate_line("", 0.1, 0.96))
        self.assertFalse(should_escalate_line("$x^2 + y^2$", 0.2, 0.96))
        self.assertFalse(should_escalate_line("مرحبا", 0.2, 0.96))
        self.assertFalse(should_escalate_line("$x$", 0.1, 1.0))

    def test_threshold_routing(self):
        self.assertEqual(DEFAULT_PPV6_CONF_THRESHOLD, 0.96)
        self.assertTrue(should_escalate_line("noisy line", 0.85, 0.96))
        self.assertTrue(should_escalate_line("borderline", 0.96, 0.96))
        self.assertFalse(should_escalate_line("clean line", 0.985, 0.96))
        self.assertTrue(should_escalate_line("clean line", 0.999, 1.0))

    def test_table_gate_skips_layout_when_no_table(self):
        ocr = ToasterOCR()
        ocr._indra = _FakeIndra(has_table=False)
        calls = {"ppv3": 0, "slanet": 0}
        ocr._detect_ppv3 = lambda *a: calls.__setitem__("ppv3", calls["ppv3"] + 1) or {"regions": [], "inference_seconds": 0.1}
        ocr._repair_slanet = lambda *a: calls.__setitem__("slanet", calls["slanet"] + 1) or {"markdown": ""}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.png"
            Image.new("RGB", (16, 16), "white").save(p)
            res = ocr.run_image(p, mode="pp-v6")
            self.assertTrue(res["table_layout_skipped"])
            self.assertEqual(calls, {"ppv3": 0, "slanet": 0})
            res_always = ocr.run_image(p, mode="pp-v6", table_gate=False)
            self.assertFalse(res_always["table_layout_skipped"])
            self.assertEqual(calls["ppv3"], 1)


if __name__ == "__main__":
    unittest.main()
