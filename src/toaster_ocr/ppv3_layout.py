"""PP-DocLayoutV3 detector and exact-once native table cell regrouping."""
from __future__ import annotations

import copy
import html
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from indra_ocr import layout_post

from .paths import model_dir


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb > inter else 0.0


def _area_overlap(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    return inter / aa if aa else 0.0


def _valid_box(box) -> bool:
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(x, (int, float)) and math.isfinite(x) for x in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _table_html(atoms) -> str:
    boxes = np.asarray([a["box"] for a in atoms], dtype=np.float32).reshape(-1, 4)
    return layout_post.table_html(boxes, [a["text"] for a in atoms])


def _atoms(page):
    atoms = []
    for pi, part in enumerate(page.get("parts", [])):
        for ci, cell in enumerate(part.get("cells") or []):
            box = cell.get("box")
            if not _valid_box(box) or not isinstance(cell.get("text"), str):
                raise ValueError(f"invalid native atom at part {pi}, cell {ci}")
            atoms.append(
                {
                    "key": f"{pi}:{ci}",
                    "part_index": pi,
                    "cell_index": ci,
                    "source_part_id": part.get("id"),
                    "source_cell_id": cell.get("id"),
                    "text": cell["text"],
                    "box": list(map(float, box)),
                    "line_id": cell.get("line_id"),
                }
            )
    return atoms


def regroup(native: dict[str, Any], regions: list[dict[str, Any]], mode: str = "tables_only", min_ioa: float = 0.5):
    """Regroup native Indra cell atoms into PP-DocLayoutV3 table regions without dropping text."""
    if mode not in ("tables_only", "layout_first", "native_geometry"):
        raise ValueError("mode must be tables_only, layout_first, or native_geometry")
    page = copy.deepcopy(native)
    diagnostics: dict[str, Any] = {
        "mode": mode,
        "fallback": None,
        "table_regions": 0,
        "reconstructed_tables": 0,
    }
    try:
        try:
            atoms = _atoms(native)
        except ValueError as exc:
            diagnostics["fallback"] = f"opaque_native_parts: {exc}"
            return copy.deepcopy(native), diagnostics
        if mode == "native_geometry":
            for part in page.get("parts", []):
                if part.get("label") == "Table" and part.get("cells"):
                    part["markdown"] = _table_html(
                        [
                            {"text": c["text"], "box": c["box"]}
                            for c in part["cells"]
                            if isinstance(c.get("text"), str) and _valid_box(c.get("box"))
                        ]
                    )
            page["markdown"] = "\n\n".join(p.get("markdown", "") for p in page.get("parts", []) if p.get("markdown"))
            return page, diagnostics

        width, height = native.get("width"), native.get("height")
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)) or width <= 0 or height <= 0:
            raise ValueError("missing page dimensions")
        regs = []
        for ri, region in enumerate(regions or []):
            box = region.get("box_xyxy")
            if not _valid_box(box) or box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                raise ValueError(f"invalid/out-of-page region box {ri}")
            regs.append((ri, region, list(map(float, box))))
        table_regs = [(i, r, b) for i, r, b in regs if r.get("raw_label") == "table"]
        diagnostics["table_regions"] = len(table_regs)
        if not regs:
            diagnostics["fallback"] = "no_regions"
            return page, diagnostics

        assigned = defaultdict(list)
        for atom in atoms:
            ranked = sorted(((_area_overlap(atom["box"], b), i, r) for i, r, b in regs), key=lambda x: (-x[0], x[1]))
            if ranked and ranked[0][0] >= min_ioa:
                assigned[ranked[0][1]].append(atom)

        source_parts = native.get("parts", [])
        consumed = set()
        rendered = []
        for ri, region, box in table_regs:
            selected = assigned.get(ri, [])
            if not selected:
                continue
            consumed |= {a["key"] for a in selected}
            line_ids = list(dict.fromkeys(a["line_id"] for a in selected if a["line_id"] is not None))
            cells = [
                {
                    "id": a["source_cell_id"],
                    "line_id": a["line_id"],
                    "text": a["text"],
                    "box": a["box"],
                    "source_atom_key": a["key"],
                }
                for a in sorted(selected, key=lambda a: (a["box"][1], a["box"][0], a["key"]))
            ]
            lines = []
            for a in selected:
                if a["line_id"] is not None:
                    found = next(
                        (ln for p in source_parts for ln in p.get("lines", []) if ln.get("id") == a["line_id"]), None
                    )
                    if found is not None and all(ln.get("id") != found.get("id") for ln in lines):
                        lines.append(copy.deepcopy(found))
            p = {
                "id": f"ppv3_table_{ri:04d}",
                "label": "Table",
                "bbox": box,
                "line_ids": line_ids,
                "lines": lines,
                "cells": cells,
                "markdown": _table_html(selected),
            }
            rendered.append((region.get("order_seq", 10**9), ri, p))
        diagnostics["reconstructed_tables"] = len(rendered)
        if not rendered and mode == "tables_only":
            diagnostics["fallback"] = "no_table_atoms_assigned"
            return page, diagnostics

        rebuilt = []
        for pi, part in enumerate(source_parts):
            part_atoms = [a for a in atoms if a["part_index"] == pi]
            taken = [a for a in part_atoms if a["key"] in consumed]
            if not taken:
                untouched = copy.deepcopy(part)
                for a in part_atoms:
                    untouched["cells"][a["cell_index"]]["source_atom_key"] = a["key"]
                rebuilt.append(((pi, 0, 0), untouched))
                continue
            remaining = [a for a in part_atoms if a["key"] not in consumed]
            if remaining:
                residual = copy.deepcopy(part)
                residual["cells"] = []
                for a in remaining:
                    c = copy.deepcopy(part.get("cells", [])[a["cell_index"]])
                    c["source_atom_key"] = a["key"]
                    residual["cells"].append(c)
                residual["line_ids"] = list(dict.fromkeys(a["line_id"] for a in remaining if a["line_id"] is not None))
                residual["lines"] = [
                    copy.deepcopy(ln) for ln in part.get("lines", []) if ln.get("id") in residual["line_ids"]
                ]
                if part.get("label") == "Table":
                    residual["markdown"] = _table_html(remaining)
                else:
                    residual["markdown"] = "\n".join(html.escape(a["text"], quote=False) for a in remaining)
                residual["bbox"] = [
                    min(a["box"][0] for a in remaining),
                    min(a["box"][1] for a in remaining),
                    max(a["box"][2] for a in remaining),
                    max(a["box"][3] for a in remaining),
                ]
                rebuilt.append(((pi, 0, 0), residual))

        for _, ri, part in rendered:
            source_positions = [a["part_index"] for a in assigned[ri] if a["key"] in consumed]
            anchor = min(source_positions) if source_positions else len(source_parts)
            rebuilt.append(((anchor, 1, ri), part))

        if mode == "layout_first":
            ordered = []
            for pos, part in rebuilt:
                if part["id"].startswith("ppv3_table_"):
                    ri = int(part["id"].rsplit("_", 1)[-1])
                    seq = next((r.get("order_seq", 10**9) for i, r, _ in regs if i == ri), 10**9)
                    ordered.append((float(seq), pos, part))
                else:
                    bbox = part.get("bbox")
                    candidates = [(_iou(bbox, b), r.get("order_seq", 10**9), i) for i, r, b in regs if _valid_box(bbox)]
                    seq = max(candidates, key=lambda x: (x[0], -x[1], -x[2]))[1] if candidates else 10**9
                    ordered.append((float(seq), pos, part))
            rebuilt_parts = [p for _, _, p in sorted(ordered, key=lambda x: (x[0], x[1]))]
        else:
            rebuilt_parts = [p for _, p in sorted(rebuilt, key=lambda x: x[0])]
        page["parts"] = rebuilt_parts
        page["markdown"] = "\n\n".join(p.get("markdown", "") for p in rebuilt_parts if p.get("markdown"))
        return page, diagnostics
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        diagnostics["fallback"] = f"irreconcilable: {exc}"
        return copy.deepcopy(native), diagnostics


class PPDocLayoutV3Detector:
    """CPU detector wrapper for PaddlePaddle/PP-DocLayoutV3 via HuggingFace transformers."""

    def __init__(self, ppv3_dir: Path | None = None, threads: int = 2):
        import torch
        from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessor

        base = Path(ppv3_dir) if ppv3_dir else model_dir("ppv3")
        if not (base / "config.json").exists():
            raise FileNotFoundError(
                f"PP-DocLayoutV3 models missing in {base}. Run 'toaster-ocr download ppv3' or set TOASTER_OCR_PPV3_DIR."
            )
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.torch = torch
        t0 = time.monotonic()
        self.processor = PPDocLayoutV3ImageProcessor.from_pretrained(base, local_files_only=True)
        self.model = PPDocLayoutV3ForObjectDetection.from_pretrained(base, local_files_only=True).eval().to("cpu")
        self.load_seconds = time.monotonic() - t0

    def detect(self, image_path: str | Path, width: int, height: int) -> dict[str, Any]:
        t0 = time.monotonic()
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        if image.size != (width, height):
            raise ValueError(f"Dimension mismatch: image={image.size} vs expected={(width, height)}")
        batch = self.processor(images=image, return_tensors="pt")
        with self.torch.inference_mode():
            outputs = self.model(**batch)
        pred = self.processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=[(height, width)])[0]
        regions = []
        for i, (box, score, label, rank) in enumerate(
            zip(pred["boxes"], pred["scores"], pred["labels"], pred["order_seq"])
        ):
            b = [max(0.0, min(float(width if k % 2 == 0 else height), v)) for k, v in enumerate(box.tolist())]
            if not _valid_box(b):
                continue
            regions.append(
                {
                    "box_xyxy": b,
                    "score": float(score),
                    "label_id": int(label),
                    "raw_label": self.model.config.id2label[int(label)],
                    "order_seq": int(rank),
                    "service_order": i,
                }
            )
        return {"regions": regions, "inference_seconds": time.monotonic() - t0}


class PaddleDocLayoutDetector:
    """CPU detector wrapper for PaddlePaddle/PP-DocLayout-{S,M,L} via paddle.inference."""

    def __init__(self, variant: str = "s", model_path: Path | None = None, threads: int = 2):
        import cv2
        import paddle.inference as pi
        import yaml

        v = variant.lower()
        if v not in ("s", "m", "l"):
            raise ValueError(f"Unknown PP-DocLayout variant {variant!r}; expected s, m, or l")
        key = f"doclayout-{v}"
        base = Path(model_path) if model_path else model_dir(key)
        if not (base / "inference.json").exists():
            raise FileNotFoundError(
                f"PP-DocLayout-{v.upper()} missing in {base}. Run 'toaster-ocr download {key}'."
            )
        cv2.setNumThreads(1)
        self.cv2 = cv2
        yml = yaml.safe_load((base / "inference.yml").read_text())
        self.labels = yml["label_list"]
        self.th, self.tw = yml["Preprocess"][0]["target_size"]
        norm = yml["Preprocess"][1]
        self.mean = np.array(norm.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.std = np.array(norm.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)

        t0 = time.monotonic()
        cfg = pi.Config(str(base / "inference.json"), str(base / "inference.pdiparams"))
        cfg.disable_gpu()
        cfg.set_cpu_math_library_num_threads(threads)
        cfg.switch_ir_optim(False)
        cfg.disable_glog_info()
        self.pred = pi.create_predictor(cfg)
        self.load_seconds = time.monotonic() - t0

    def detect(self, image_path: str | Path, width: int, height: int) -> dict[str, Any]:
        t0 = time.monotonic()
        bgr = self.cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        rgb = bgr[:, :, ::-1]
        h_img, w_img = rgb.shape[:2]
        if (w_img, h_img) != (width, height):
            raise ValueError(f"Dimension mismatch: image={(w_img, h_img)} vs expected={(width, height)}")
        rs = (
            self.cv2.resize(rgb, (self.tw, self.th), interpolation=self.cv2.INTER_CUBIC).astype(np.float32) / 255.0
            - self.mean
        ) / self.std
        inp = rs.transpose(2, 0, 1)[None]
        sf = np.array([[self.th / float(h_img), self.tw / float(w_img)]], dtype=np.float32)
        for name in self.pred.get_input_names():
            h = self.pred.get_input_handle(name)
            if name == "image":
                h.reshape(inp.shape)
                h.copy_from_cpu(inp)
            elif name == "scale_factor":
                h.reshape(sf.shape)
                h.copy_from_cpu(sf)
            elif name == "im_shape":
                ims = np.array([[float(self.th), float(self.tw)]], dtype=np.float32)
                h.reshape(ims.shape)
                h.copy_from_cpu(ims)
        self.pred.run()
        boxes = self.pred.get_output_handle(self.pred.get_output_names()[0]).copy_to_cpu()
        regions = []
        for i, b in enumerate(boxes):
            cid = int(b[0])
            score = float(b[1])
            if cid < 0 or score < 0.5:
                continue
            box = [
                max(0.0, min(float(w_img), float(b[2]))),
                max(0.0, min(float(h_img), float(b[3]))),
                max(0.0, min(float(w_img), float(b[4]))),
                max(0.0, min(float(h_img), float(b[5]))),
            ]
            if not _valid_box(box):
                continue
            regions.append(
                {
                    "box_xyxy": box,
                    "score": score,
                    "label_id": cid,
                    "raw_label": self.labels[cid],
                    "order_seq": i,
                    "service_order": i,
                }
            )
        return {"regions": regions, "inference_seconds": time.monotonic() - t0}
