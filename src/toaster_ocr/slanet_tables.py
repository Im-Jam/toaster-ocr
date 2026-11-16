"""SLANet-1M table structure predictor with context padding, sparse-grid gate, and nearest-missing rescue."""
from __future__ import annotations

import copy
import html
import math
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from .paths import model_dir


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _overlap_frac(a, b) -> float:
    return _intersection(a, b) / max(_area(a), 1e-9)


def _center(b) -> tuple[float, float]:
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


def validate_grid(cells: list[dict[str, Any]]) -> tuple[int, int]:
    if not cells:
        raise ValueError("empty_structure")
    occupied = set()
    for c in cells:
        rs = sorted(c["row_nums"])
        cs = sorted(c["column_nums"])
        if (
            not rs
            or not cs
            or rs != list(range(rs[0], rs[-1] + 1))
            or cs != list(range(cs[0], cs[-1] + 1))
            or min(rs + cs) < 0
            or _area(c["bbox"]) <= 0
        ):
            raise ValueError("invalid_cell_geometry_or_span")
        for r in rs:
            for col in cs:
                if (r, col) in occupied:
                    raise ValueError("overlapping_spans")
                occupied.add((r, col))
    nr = max(r for r, _ in occupied) + 1
    nc = max(c for _, c in occupied) + 1
    if nr * nc > 10000 or len(occupied) != nr * nc:
        raise ValueError("incomplete_grid")
    return nr, nc


def cells_html(cells: list[dict[str, Any]]) -> str:
    nr, _ = validate_grid(cells)
    rows = []
    for row in range(nr):
        items = []
        for c in sorted(cells, key=lambda c: min(c["column_nums"])):
            if min(c["row_nums"]) != row:
                continue
            tag = "th" if c.get("header") else "td"
            attrs = ""
            if len(c["row_nums"]) > 1:
                attrs += f' rowspan="{len(c["row_nums"])}"'
            if len(c["column_nums"]) > 1:
                attrs += f' colspan="{len(c["column_nums"])}"'
            items.append(f"<{tag}{attrs}>{html.escape(c.get('text', ''), quote=True)}</{tag}>")
        rows.append("<tr>" + "".join(items) + "</tr>")
    return "<table>\n" + "\n".join(rows) + "\n</table>"


class _StructureParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells: list[dict[str, Any]] = []
        self.row = -1
        self.col = 0
        self.used: set[tuple[int, int]] = set()
        self.open_cell = False
        self.in_row = False
        self.header = False

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self.header = True
        if tag == "tr":
            if self.in_row:
                raise ValueError("nested_row")
            self.row += 1
            self.col = 0
            self.in_row = True
        if tag == "td":
            if not self.in_row or self.open_cell:
                raise ValueError("invalid_cell_nesting")
            while (self.row, self.col) in self.used:
                self.col += 1
            a = dict(attrs)
            rs = int(a.get("rowspan", "1"))
            cs = int(a.get("colspan", "1"))
            if not 1 <= rs <= 100 or not 1 <= cs <= 100:
                raise ValueError("invalid_span")
            rows = list(range(self.row, self.row + rs))
            cols = list(range(self.col, self.col + cs))
            for r in rows:
                for c in cols:
                    if (r, c) in self.used:
                        raise ValueError("overlapping_spans")
                    self.used.add((r, c))
            self.cells.append({"row_nums": rows, "column_nums": cols, "header": self.header})
            self.col += cs
            self.open_cell = True

    def handle_endtag(self, tag):
        if tag == "td":
            if not self.open_cell:
                raise ValueError("unmatched_cell_end")
            self.open_cell = False
        if tag == "tr":
            if self.open_cell or not self.in_row:
                raise ValueError("unclosed_cell_or_row")
            self.in_row = False
        if tag == "thead":
            self.header = False


class _GridHTMLFeatures(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self.current: list[dict[str, Any]] | None = None
        self.cell: dict[str, Any] | None = None
        self.depth = 0
        self.feed(text or "")

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
        if self.depth != 1:
            return
        if tag == "tr":
            self.current = []
            self.rows.append(self.current)
        if tag in ("td", "th"):
            a = dict(attrs)
            try:
                rs = max(1, int(a.get("rowspan", 1)))
            except (ValueError, TypeError):
                rs = 1
            try:
                cs = max(1, int(a.get("colspan", 1)))
            except (ValueError, TypeError):
                cs = 1
            self.cell = {"rowspan": rs, "colspan": cs, "text": ""}
            if self.current is None:
                self.current = []
                self.rows.append(self.current)
            self.current.append(self.cell)

    def handle_data(self, data):
        if self.depth == 1 and self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.cell = None
        if tag == "table":
            self.depth = max(0, self.depth - 1)

    def features(self) -> dict[str, Any]:
        cs = [c for r in self.rows for c in r]
        return {
            "rows": len(self.rows),
            "columns": max((sum(c["colspan"] for c in r) for r in self.rows), default=0),
            "cells": len(cs),
            "empty": sum(not c["text"].strip() for c in cs) / max(1, len(cs)),
        }


def padding_for(crop_size: tuple[float, float], fragments: list[dict[str, Any]]) -> int:
    """Context padding: 3% of min crop side, capped at half the median glyph height (min 2px)."""
    w, h = crop_size
    heights = sorted(max(1.0, float(f["bbox"][3]) - float(f["bbox"][1])) for f in fragments)
    med = heights[(len(heights) - 1) // 2] if heights else 8.0
    p = max(2, round(0.03 * min(w, h)))
    return min(p, max(2, round(0.5 * med)))


def assign_and_render(
    cells: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
    crop_w: int,
    crop_h: int,
    *,
    rescue_nearest_missing: bool = True,
) -> tuple[str, list[int]]:
    """Assign native fragments to SLANet cells with optional conservative nearest-missing rescue."""
    nr, nc = validate_grid(cells)
    cells = copy.deepcopy(cells)
    assignments: list[list[int]] = [[] for _ in cells]
    missing: list[int] = []
    for i, f in enumerate(fragments):
        scores = [_overlap_frac(f["bbox"], c["bbox"]) for c in cells]
        best = max(range(len(cells)), key=scores.__getitem__) if scores else None
        if best is not None and scores[best] >= 0.10:
            assignments[best].append(i)
            continue
        if rescue_nearest_missing and cells:
            fx, fy = _center(f["bbox"])
            cand = []
            for n, c in enumerate(cells):
                cx, cy = _center(c["bbox"])
                dx = abs(fx - cx) / max(crop_w / nc, 1e-9)
                dy = abs(fy - cy) / max(crop_h / nr, 1e-9)
                cand.append((max(dx, dy), dx * dx + dy * dy, n))
            dist, _, n = min(cand)
            if dist <= 0.50:
                assignments[n].append(i)
                continue
        missing.append(i)

    fraction = (len(fragments) - len(missing)) / len(fragments)
    if fraction < 0.50:
        raise ValueError("assignment_below_half")
    for c, ids in zip(cells, assignments):
        ids.sort(key=lambda i: (fragments[i]["bbox"][1], fragments[i]["bbox"][0], i))
        c["fragment_indices"] = ids
        c["text"] = " ".join(fragments[i]["text"] for i in ids)
    table = cells_html(cells)
    appendix = "\n\n" + "\n\n".join(html.escape(fragments[i]["text"], quote=True) for i in missing) if missing else ""
    return table + appendix, missing


def is_sparse_grid_explosion(native_part: dict[str, Any], predicted_markdown: str) -> bool:
    """Reject SLANet outputs that explode into mostly-empty (>50% empty) oversized grids."""
    n_frags = max(1, len([x for x in native_part.get("cells", []) if x.get("text", "").strip()]))
    nf = _GridHTMLFeatures(native_part.get("markdown", "")).features()
    sf = _GridHTMLFeatures(predicted_markdown).features()
    return sf["cells"] > 2 * n_frags and sf["empty"] > 0.5 and sf["cells"] > nf["cells"]


class SLANetTables:
    """SLANet-1M table structure predictor with context padding and sparse-grid fallback."""

    def __init__(self, slanet_dir: Path | None = None, threads: int = 2):
        import paddle.inference as pi

        base = Path(slanet_dir) if slanet_dir else model_dir("slanet")
        pdmodel = base / "inference.pdmodel"
        pdiparams = base / "inference.pdiparams"
        yml = base / "inference.yml"
        if not (pdmodel.exists() and pdiparams.exists() and yml.exists()):
            raise FileNotFoundError(
                f"SLANet models missing in {base}. Run 'toaster-ocr download slanet' or set TOASTER_OCR_SLANET_DIR."
            )
        t0 = time.monotonic()
        cfg = pi.Config(str(pdmodel), str(pdiparams))
        cfg.disable_gpu()
        cfg.set_cpu_math_library_num_threads(threads)
        cfg.switch_ir_optim(False)
        cfg.disable_glog_info()
        self.predictor = pi.create_predictor(cfg)
        yml_cfg = yaml.safe_load(yml.read_text())
        chars = list(yml_cfg["PostProcess"]["character_dict"])
        if "<td></td>" not in chars:
            chars.append("<td></td>")
        if "<td>" in chars:
            chars.remove("<td>")
        self.chars = ["sos"] + chars + ["eos"]
        self.load_seconds = time.monotonic() - t0

    def _predict_cells(self, crop: Image.Image) -> list[dict[str, Any]]:
        w, h = crop.size
        ratio = 488.0 / max(w, h)
        rw, rh = max(1, int(w * ratio)), max(1, int(h * ratio))
        a = cv2.resize(np.asarray(crop)[:, :, ::-1], (rw, rh), interpolation=cv2.INTER_LINEAR).astype("float32") / 255.0
        a = (a - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array([0.229, 0.224, 0.225], dtype="float32")
        inp = np.zeros((488, 488, 3), dtype="float32")
        inp[: a.shape[0], : a.shape[1]] = a
        handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        arr = inp.transpose(2, 0, 1)[None]
        handle.reshape(arr.shape)
        handle.copy_from_cpu(arr)
        self.predictor.run()
        outs = [self.predictor.get_output_handle(name).copy_to_cpu() for name in self.predictor.get_output_names()]
        loc = next(x for x in outs if x.shape[-1] == 4)
        probs = next(x for x in outs if x.shape[-1] == len(self.chars))
        idx = probs[0].argmax(-1)
        scores = probs[0].max(-1)
        tokens, boxes, confs, ended = [], [], [], False
        for n, k in enumerate(idx):
            ch = self.chars[int(k)]
            if n > 0 and ch == "eos":
                ended = True
                break
            if ch in ("sos", "eos"):
                continue
            tokens.append(ch)
            confs.append(float(scores[n]))
            if ch in ("<td>", "<td", "<td></td>"):
                boxes.append((loc[0, n] * np.array([w, h, w, h])).tolist())
        if not ended or (sum(confs) / max(1, len(confs))) < 0.50:
            raise ValueError("slanet_low_confidence_or_no_eos")
        parser = _StructureParser()
        parser.feed("".join(tokens))
        parser.close()
        if parser.open_cell or parser.in_row or len(parser.cells) != len(boxes):
            raise ValueError("slanet_invalid_structure")
        for c, b in zip(parser.cells, boxes):
            c["bbox"] = b
        return parser.cells

    def repair_tables(self, image_path: str | Path, page_dict: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        out = copy.deepcopy(page_dict)
        w, h = int(out["width"]), int(out["height"])
        image = None
        attempted = 0
        replaced = 0
        for part in out.get("parts", []):
            if part.get("label") != "Table":
                continue
            attempted += 1
            try:
                if image is None:
                    image = Image.open(image_path).convert("RGB")
                b = part["bbox"]
                frags_raw = []
                for f in part.get("cells", []):
                    if not f.get("text", "").strip():
                        continue
                    fb = f.get("box", f.get("bbox"))
                    frags_raw.append({"text": f["text"], "bbox": list(map(float, fb))})
                if not frags_raw:
                    continue
                pad = padding_for((max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])), frags_raw)
                x0 = max(0, math.floor(b[0] - pad))
                y0 = max(0, math.floor(b[1] - pad))
                x1 = min(w, math.ceil(b[2] + pad))
                y1 = min(h, math.ceil(b[3] + pad))
                if x1 <= x0 or y1 <= y0:
                    continue
                crop_frags = [
                    {"text": f["text"], "bbox": [f["bbox"][0] - x0, f["bbox"][1] - y0, f["bbox"][2] - x0, f["bbox"][3] - y0]}
                    for f in frags_raw
                ]
                cells = self._predict_cells(image.crop((x0, y0, x1, y1)))
                repaired_md, _ = assign_and_render(cells, crop_frags, x1 - x0, y1 - y0, rescue_nearest_missing=True)
                if is_sparse_grid_explosion(part, repaired_md):
                    continue
                part["markdown"] = repaired_md
                replaced += 1
            except Exception:
                continue
        out["markdown"] = "\n\n".join(p["markdown"] for p in out.get("parts", []) if p.get("markdown"))
        out["tables_attempted"] = attempted
        out["tables_replaced"] = replaced
        out["slanet_seconds"] = time.monotonic() - start
        return out
