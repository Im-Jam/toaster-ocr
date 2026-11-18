"""Indra OrtOCR engine with optional confidence-switched PP-OCRv6_small_rec_onnx."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image
from indra_ocr import layout_post, pipeline
from indra_ocr.order_post import BLOCK_CLASSES
from indra_ocr.rtl import has_rtl

from .paths import model_dir

DEFAULT_PPV6_CONF_THRESHOLD = 0.96
DEFAULT_PPV6_MIN_CHARS = 5


def _box(b) -> list[float]:
    return [float(v) for v in b[:4]]


def _union(b) -> list[float]:
    return [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())]


def _crop_box(bgr: np.ndarray, box) -> np.ndarray | None:
    h_img, w_img = bgr.shape[:2]
    x0, y0, x1, y1 = map(float, box[:4])
    if x1 <= x0 or y1 <= y0:
        return None
    lh = max(1.0, y1 - y0)
    cx0, cx1 = int(math.floor(x0 - 0.3 * lh)), int(math.ceil(x1 + 0.3 * lh))
    cy0, cy1 = int(math.floor(y0 - 0.2 * lh)), int(math.ceil(y1 + 0.2 * lh))
    crop = np.full((max(1, cy1 - cy0), max(1, cx1 - cx0), 3), 255, dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, cx0), max(0, cy0), min(w_img, cx1), min(h_img, cy1)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - cy0 : sy1 - cy0, sx0 - cx0 : sx1 - cx0] = bgr[sy0:sy1, sx0:sx1]
    return crop


def should_escalate_line(text: str, conf: float, threshold: float) -> bool:
    """Return True if a recognized line/cell should be re-read by PP-OCRv6."""
    if not text or "$" in text or has_rtl(text):
        return False
    if threshold >= 1.0:
        return True
    if len(text.strip()) <= DEFAULT_PPV6_MIN_CHARS:
        return False
    return conf <= threshold


class PPV6Recognizer:
    """ONNX Runtime CTC recognizer for PP-OCRv6_small_rec_onnx."""

    def __init__(self, ppv6_dir: Path | None = None, threads: int = 2):
        base = Path(ppv6_dir) if ppv6_dir else model_dir("ppocrv6")
        model_path = base / "inference.onnx"
        yml_path = base / "inference.yml"
        if not model_path.exists() or not yml_path.exists():
            raise FileNotFoundError(
                f"PP-OCRv6 files missing in {base}. Run 'toaster-ocr download ppocrv6' or set TOASTER_OCR_PPV6_DIR."
            )
        cfg = yaml.safe_load(yml_path.read_text())
        self.chars = ["blank"] + list(cfg["PostProcess"]["character_dict"]) + [" "]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    @staticmethod
    def _prep(img_bgr: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        img_h, max_img_w = 48, 3200
        img_w = min(max_img_w, int(img_h * max_wh_ratio))
        h, w = img_bgr.shape[:2]
        ratio = w / float(max(1, h))
        resized_w = min(img_w, max(1, int(math.ceil(img_h * ratio))))
        rs = cv2.resize(img_bgr, (resized_w, img_h)).astype("float32")
        rs = (rs.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        pad = np.zeros((3, img_h, img_w), dtype=np.float32)
        pad[:, :, :resized_w] = rs
        return pad

    def recognize_crops(self, crops: list[np.ndarray], batch_size: int = 16) -> list[str]:
        if not crops:
            return []
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(1, crops[i].shape[0]))
        results = [""] * len(crops)
        for off in range(0, len(order), batch_size):
            chunk = order[off : off + batch_size]
            max_wh = max(320.0 / 48.0, max(crops[i].shape[1] / max(1, crops[i].shape[0]) for i in chunk))
            batch = np.stack([self._prep(crops[i], max_wh) for i in chunk], axis=0)
            probs = self.sess.run(None, {self.in_name: batch})[0]
            idx = probs.argmax(axis=-1)
            for b, orig_i in enumerate(chunk):
                sel = np.ones(len(idx[b]), dtype=bool)
                sel[1:] = idx[b][1:] != idx[b][:-1]
                sel &= idx[b] != 0
                results[orig_i] = "".join(self.chars[t] for t in idx[b][sel])
        return results


class _Capture:
    def __init__(self, ocr):
        self.ocr = ocr
        self.parts: list[dict[str, Any]] = []
        self.delegate_block = layout_post._block_part
        self.delegate_md = pipeline.page_markdown
        self.use_ppv6 = False
        self.ppv6_conf_threshold = DEFAULT_PPV6_CONF_THRESHOLD
        self.ppv6: PPV6Recognizer | None = None
        self.current_bgr_turned: np.ndarray | None = None
        self.last_rec_boxes_confs: list[tuple[list[float], float]] = []
        self.ppv6_seconds = 0.0
        self.ppv6_escalated = 0
        self.ppv6_candidates = 0

    def block(self, parts, blk, boxes, texts, cls_logits, keep_furniture, cells, tab):
        n = len(parts)
        self.delegate_block(parts, blk, boxes, texts, cls_logits, keep_furniture, cells, tab)
        if len(parts) > n:
            cls = int(cls_logits[blk].sum(0).argmax())
            lines = [{"id": int(i), "text": texts[i], "box": _box(boxes[i]), "conf": None} for i in blk]
            cs = []
            for i in blk:
                for k, (b, t) in enumerate((cells or {}).get(i) or [(boxes[i, :4], texts[i])]):
                    cs.append({"id": f"line_{i}_cell_{k}", "line_id": int(i), "text": t, "box": _box(b)})
            self.parts.append(
                {
                    "id": f"part_{len(self.parts):04d}",
                    "markdown": parts[-1],
                    "label": BLOCK_CLASSES[cls],
                    "bbox": _union(boxes[blk, :4]),
                    "line_ids": [int(i) for i in blk],
                    "lines": lines,
                    "cells": cs,
                }
            )

    def _line_conf(self, box) -> float:
        if not self.last_rec_boxes_confs:
            return 0.0
        x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        best_iou, best_c = -1.0, None
        for rb, rc in self.last_rec_boxes_confs:
            if rb[0] - 2.0 <= cx <= rb[2] + 2.0 and rb[1] - 2.0 <= cy <= rb[3] + 2.0:
                best_c = rc if best_c is None else min(best_c, rc)
            ix0, iy0 = max(x0, rb[0]), max(y0, rb[1])
            ix1, iy1 = min(x1, rb[2]), min(y1, rb[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if inter > 0.0:
                area = max(1e-3, (x1 - x0) * (y1 - y0) + (rb[2] - rb[0]) * (rb[3] - rb[1]) - inter)
                iou = inter / area
                if iou > best_iou:
                    best_iou = iou
                    if best_c is None:
                        best_c = rc
        return float(best_c) if best_c is not None else 0.0

    def _apply_ppv6(self, boxes, texts, cells):
        if not self.use_ppv6 or self.ppv6 is None or self.current_bgr_turned is None or self.ocr._page_rtl:
            return texts, cells
        t0 = time.monotonic()
        bgr = self.current_bgr_turned
        texts = list(texts)
        thr = self.ppv6_conf_threshold
        crop_tasks = []
        candidates = 0
        for i, (b, tx) in enumerate(zip(boxes, texts)):
            if not tx or "$" in tx or has_rtl(tx):
                continue
            candidates += 1
            if not should_escalate_line(tx, self._line_conf(b), thr):
                continue
            c = _crop_box(bgr, b)
            if c is not None:
                crop_tasks.append(("line", i, None, c))
        new_cells = None
        if cells is not None:
            new_cells = {k: list(v) for k, v in cells.items()}
            for i, clist in new_cells.items():
                for k, (cb, ctx) in enumerate(clist):
                    if not ctx or "$" in ctx or has_rtl(ctx):
                        continue
                    candidates += 1
                    if not should_escalate_line(ctx, self._line_conf(cb), thr):
                        continue
                    c = _crop_box(bgr, cb)
                    if c is not None:
                        crop_tasks.append(("cell", i, k, c))
        if crop_tasks:
            preds = self.ppv6.recognize_crops([x[3] for x in crop_tasks])
            for (kind, idx, subidx, _), pred_tx in zip(crop_tasks, preds):
                if not pred_tx:
                    continue
                if kind == "line":
                    texts[idx] = pred_tx
                elif new_cells is not None:
                    cb, _ = new_cells[idx][subidx]
                    new_cells[idx][subidx] = (cb, pred_tx)
        self.ppv6_candidates = candidates
        self.ppv6_escalated = len(crop_tasks)
        self.ppv6_seconds = time.monotonic() - t0
        return texts, new_cells if new_cells is not None else cells

    def markdown(self, boxes, texts, order, cls_logits, same, keep_furniture=False, cells=None, tab=None, pictures=None):
        self.parts = []
        texts, cells = self._apply_ppv6(boxes, texts, cells)
        md = self.delegate_md(boxes, texts, order, cls_logits, same, keep_furniture, cells, tab, pictures)
        if pictures is not None and len(pictures):
            slots: dict[int, list[dict[str, Any]]] = {}
            for k, p in enumerate(np.asarray(pictures)[:, :4]):
                x0, y0, x1, y1 = p
                cy = (y0 + y1) / 2
                at = next(
                    (
                        j
                        for j, pt in enumerate(self.parts)
                        if (pt["bbox"][1] + pt["bbox"][3]) / 2 > cy and min(x1, pt["bbox"][2]) > max(x0, pt["bbox"][0])
                    ),
                    len(self.parts),
                )
                slots.setdefault(at, []).append(
                    {
                        "id": f"picture_{k:04d}",
                        "markdown": f"![](picture-{k})",
                        "label": "Picture",
                        "bbox": _box(p),
                        "line_ids": [],
                        "lines": [],
                        "cells": [],
                    }
                )
            merged = []
            for j in range(len(self.parts) + 1):
                merged.extend(slots.get(j, []))
                if j < len(self.parts):
                    merged.append(self.parts[j])
            self.parts = merged
        return md

    def install(self):
        layout_post._block_part = self.block
        pipeline.page_markdown = self.markdown

    def close(self):
        layout_post._block_part = self.delegate_block
        pipeline.page_markdown = self.delegate_md

    def map_upright(self, page, wh):
        if page.vertical or not page.skew:
            return
        for part in self.parts:
            part["bbox"] = pipeline.unrotate_box(part["bbox"], page.skew, wh)
            for ln in part["lines"]:
                ln["box"] = pipeline.unrotate_box(ln["box"], page.skew, wh)
            for c in part["cells"]:
                c["box"] = pipeline.unrotate_box(c["box"], page.skew, wh)


class IndraEngine:
    """Persistent Indra OrtOCR wrapper with optional confidence-switched PP-OCRv6."""

    def __init__(
        self,
        indra_dir: Path | None = None,
        ppv6_dir: Path | None = None,
        threads: int = 2,
    ):
        cv2.setNumThreads(1)
        m_dir = Path(indra_dir) if indra_dir else model_dir("indra")
        if not (m_dir / "det.onnx").exists():
            raise FileNotFoundError(
                f"Indra models missing in {m_dir}. Run 'toaster-ocr download indra' or set TOASTER_OCR_INDRA_DIR."
            )
        self.threads = threads
        self.ppv6_dir = Path(ppv6_dir) if ppv6_dir else model_dir("ppocrv6")
        t0 = time.monotonic()
        self.ocr = pipeline.OrtOCR(
            str(m_dir), threads=threads, furniture=True, math=True, providers=["CPUExecutionProvider"]
        )
        self._math_reader = self.ocr.math
        self.capture = _Capture(self.ocr)
        self.capture.install()
        self._orig_turn = self.ocr._turn
        self._orig_recognize = self.ocr.recognize
        self._raw_bgr: np.ndarray | None = None

        def wrapped_turn(gray, k, a, small=None):
            out_gray, out_k, out_skew = self._orig_turn(gray, k, a, small)
            if self.capture.use_ppv6 and self._raw_bgr is not None:
                bgr = self._raw_bgr
                if out_k == pipeline.VERTICAL:
                    bgr = np.ascontiguousarray(np.rot90(bgr, 1))
                elif out_k:
                    bgr = np.ascontiguousarray(np.rot90(bgr, -out_k))
                if abs(out_skew) >= pipeline.DESKEW_MIN:
                    bh, bw = bgr.shape[:2]
                    mat = cv2.getRotationMatrix2D((bw / 2, bh / 2), out_skew, 1.0)
                    bgr = cv2.warpAffine(
                        bgr, mat, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                    )
                self.capture.current_bgr_turned = bgr
            else:
                self.capture.current_bgr_turned = None
            return out_gray, out_k, out_skew

        def wrapped_recognize(gray, boxes, vertical=False):
            rec = self._orig_recognize(gray, boxes, vertical=vertical)
            if self.capture.use_ppv6:
                self.capture.last_rec_boxes_confs = [([float(v) for v in b[:4]], float(r[1])) for b, r in zip(boxes, rec)]
            else:
                self.capture.last_rec_boxes_confs = []
            return rec

        self.ocr._turn = wrapped_turn
        self.ocr.recognize = wrapped_recognize
        self.load_seconds = time.monotonic() - t0

    def run_image(
        self,
        image_path: str | Path,
        upright_out: str | Path | None = None,
        *,
        math: bool = True,
        ppv6: bool = False,
        ppv6_threshold: float = DEFAULT_PPV6_CONF_THRESHOLD,
    ) -> dict[str, Any]:
        self.ocr.math = self._math_reader if math else None
        if ppv6 and self.capture.ppv6 is None:
            self.capture.ppv6 = PPV6Recognizer(self.ppv6_dir, threads=self.threads)
        self.capture.use_ppv6 = ppv6
        self.capture.ppv6_conf_threshold = float(ppv6_threshold)
        self.capture.ppv6_seconds = 0.0
        self.capture.ppv6_escalated = 0
        self.capture.ppv6_candidates = 0
        self.capture.last_rec_boxes_confs = []
        self.capture.parts = []

        t0 = time.monotonic()
        rgb = Image.open(image_path).convert("RGB")
        ow, oh = rgb.size
        rgb_arr = np.asarray(rgb)
        self._raw_bgr = rgb_arr[:, :, ::-1].copy() if ppv6 else None
        try:
            gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
            page = self.ocr(gray)
        finally:
            self._raw_bgr = None
            self.capture.current_bgr_turned = None
        elapsed = time.monotonic() - t0
        if page.rotation:
            rgb = rgb.rotate(page.rotation, expand=True)
        w, h = rgb.size
        self.capture.map_upright(page, (w, h))
        out_img_path = None
        if upright_out is not None:
            up_p = Path(upright_out)
            up_p.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(up_p, format="PNG")
            out_img_path = str(up_p.resolve())
        md = (page.markdown if page.markdown is not None else page.text) or ""
        parts = [] if page.vertical else self.capture.parts
        if not page.vertical and not parts and md:
            parts = [
                {
                    "id": "part_0000",
                    "markdown": md,
                    "label": "Text",
                    "bbox": [0.0, 0.0, float(w), float(h)],
                    "line_ids": list(range(len(page.lines))),
                    "lines": [
                        {"id": i, "text": ln.text, "box": ln.box, "conf": ln.conf} for i, ln in enumerate(page.lines)
                    ],
                    "cells": [],
                }
            ]
        return {
            "success": True,
            "vertical": page.vertical,
            "image_path": out_img_path,
            "width": w,
            "height": h,
            "original_width": ow,
            "original_height": oh,
            "rotation": page.rotation,
            "skew": page.skew,
            "markdown": md,
            "parts": parts,
            "timings": page.timings,
            "inference_seconds": elapsed,
            "ppv6_seconds": self.capture.ppv6_seconds,
            "ppv6_escalated": self.capture.ppv6_escalated,
            "ppv6_candidates": self.capture.ppv6_candidates,
        }

    def close(self):
        self.ocr._turn = self._orig_turn
        self.ocr.recognize = self._orig_recognize
        self.capture.close()
        if self.ocr.pool:
            self.ocr.pool.shutdown()
