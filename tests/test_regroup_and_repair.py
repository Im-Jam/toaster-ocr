import unittest

from toaster_ocr.ppv3_layout import clean_math_markdown, regroup
from toaster_ocr.slanet_tables import assign_and_render, is_sparse_grid_explosion, padding_for


class TestRegroupAndRepair(unittest.TestCase):
    def test_regroup_tables_only_preserves_all_atoms(self):
        native = {
            "width": 1000,
            "height": 1000,
            "parts": [
                {
                    "id": "part_0000",
                    "label": "Text",
                    "bbox": [50, 50, 400, 90],
                    "markdown": "Header note",
                    "line_ids": [0],
                    "lines": [{"id": 0, "text": "Header note", "box": [50, 50, 400, 90]}],
                    "cells": [{"id": "line_0_cell_0", "line_id": 0, "text": "Header note", "box": [50, 50, 400, 90]}],
                },
                {
                    "id": "part_0001",
                    "label": "Text",
                    "bbox": [100, 200, 600, 350],
                    "markdown": "ColA\nColB",
                    "line_ids": [1, 2],
                    "lines": [
                        {"id": 1, "text": "ColA", "box": [110, 210, 250, 240]},
                        {"id": 2, "text": "ColB", "box": [350, 210, 500, 240]},
                    ],
                    "cells": [
                        {"id": "line_1_cell_0", "line_id": 1, "text": "ColA", "box": [110, 210, 250, 240]},
                        {"id": "line_2_cell_0", "line_id": 2, "text": "ColB", "box": [350, 210, 500, 240]},
                    ],
                },
            ],
        }
        regions = [{"box_xyxy": [90.0, 190.0, 620.0, 360.0], "raw_label": "table", "score": 0.95, "order_seq": 1}]
        out, diag = regroup(native, regions, mode="tables_only")
        self.assertIsNone(diag["fallback"])
        self.assertEqual(diag["reconstructed_tables"], 1)
        self.assertEqual([p["label"] for p in out["parts"]], ["Text", "Table"])
        self.assertIn("ColA", out["markdown"])
        self.assertIn("ColB", out["markdown"])
        self.assertIn("Header note", out["markdown"])

    def test_regroup_strips_furniture_when_enabled(self):
        native = {
            "width": 1000,
            "height": 1000,
            "parts": [
                {
                    "id": "p0",
                    "label": "Text",
                    "bbox": [50, 20, 400, 230],
                    "markdown": "Running Header\nBody text stays",
                    "line_ids": [1, 2],
                    "lines": [
                        {"id": 1, "text": "Running Header", "box": [50, 20, 400, 40]},
                        {"id": 2, "text": "Body text stays", "box": [50, 200, 400, 230]},
                    ],
                    "cells": [
                        {"id": "c0", "line_id": 1, "text": "Running Header", "box": [50, 20, 400, 40]},
                        {"id": "c1", "line_id": 2, "text": "Body text stays", "box": [50, 200, 400, 230]},
                    ],
                },
            ],
            "markdown": "Running Header\nBody text stays",
        }
        regions = [{"box_xyxy": [40.0, 10.0, 450.0, 50.0], "score": 0.9, "label_id": 13, "raw_label": "header", "order_seq": 0}]
        reg, diag = regroup(native, regions, mode="tables_only", strip_furniture=True)
        self.assertEqual(diag["stripped_furniture_atoms"], 1)
        self.assertNotIn("Running Header", reg["markdown"])
        self.assertIn("Body text stays", reg["markdown"])

    def test_clean_math_markdown_fixes_punctuation_and_parity(self):
        raw = "## Diagram $p\nWe have $x = 1 ,$ and\n$$a + b = c .$$\nAlso $u ( x ) =$ $v ( x ) .$"
        cleaned = clean_math_markdown(raw)
        self.assertEqual(
            cleaned,
            "## Diagram p\nWe have $x = 1$, and\n$$a + b = c$$.\nAlso $u ( x ) = v ( x )$.",
        )

    def test_padding_and_nearest_missing_and_sparse_gate(self):
        frags = [
            {"text": "A", "bbox": [10, 10, 40, 30]},
            {"text": "B", "bbox": [105, 10, 140, 30]},
        ]
        self.assertGreaterEqual(padding_for((200, 100), frags), 2)
        cells = [
            {"row_nums": [0], "column_nums": [0], "bbox": [5, 5, 80, 40], "header": True},
            {"row_nums": [0], "column_nums": [1], "bbox": [110, 5, 180, 40], "header": True},
        ]
        md, missing = assign_and_render(cells, frags, 200, 50, rescue_nearest_missing=True)
        self.assertEqual(missing, [])
        self.assertIn("<th>A</th><th>B</th>", md)

        native_part = {
            "markdown": "<table><tr><td>A</td><td>B</td></tr></table>",
            "cells": [{"text": "A"}, {"text": "B"}],
        }
        exploded = "<table>" + "".join("<tr>" + "<td></td>" * 5 + "</tr>" for _ in range(4)) + "</table>"
        self.assertTrue(is_sparse_grid_explosion(native_part, exploded))


if __name__ == "__main__":
    unittest.main()
