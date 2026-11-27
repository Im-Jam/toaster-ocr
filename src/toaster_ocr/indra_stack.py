"""Pure-ONNX Fast-CPU OCR Stack for Indra (verified on full 2,937-page Omni Parse Bench: +4.50 pts / +908 net tests).

All 4 upgrades run in-process via ONNX Runtime (zero PaddlePaddle, zero PyTorch, zero network access):
1. Guarded SLANet-plus (`models/slanet-plus.onnx`, 7.4 MB): table structure + 100%-conserving <br> matcher + sparse-grid fallback.
2. PP-DocLayout-S (`models/pp_doclayout_s.onnx`, 4.7 MB, ~73 ms/page on 2 ARM64 cores) + KV-cached pix2text-mfr-1.5
   (`models/pix2text_mfr15/{encoder_model,mfr_dec_init,mfr_dec_step}.onnx`, ~319 ms/eq): display formula detection & LaTeX.
3. Page-adaptive `lm.npz` beam search (`LM_GATE -> 0.985, LM_BEAM -> 16` only on pages with median line conf < 0.91).
4. PDF `textlayer.py` PUA/ink trust fix (`MAX_BAD_CHAR_FRAC=0.05`, ignoring PUA `Co` checkboxes, `MAX_NO_INK_LINE_FRAC=0.30`).
"""
from __future__ import annotations

import html as _html
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer
import yaml

from .paths import stack_models_dir

MODELS_ROOT = stack_models_dir()

DOCLAYOUT_S_LABELS = (
    "paragraph_title", "image", "text", "number", "abstract", "content",
    "figure_title", "formula", "table", "table_title", "reference", "doc_title",
    "footnote", "header", "algorithm", "footer", "seal", "chart_title", "chart",
    "formula_number", "header_image", "footer_image", "aside_text",
)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ort_session(path: Path, threads: int = 2) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(threads))
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def formula_conf_for_rec(rec: Sequence[Any]) -> float:
    """Use conservative 0.35 formula threshold on born-digital PDF text-layer pages, 0.28 on scans."""
    if len(rec) < 4:
        return 0.28
    n_ocr = sum(1 for item in rec if len(item) >= 2 and 0.0 < float(item[1]) < 0.9999)
    return 0.35 if (n_ocr / len(rec)) < 0.35 else 0.28


def install_textlayer_trust_fix() -> None:
    """Fix Indra's overly aggressive PDF text-layer rejection on born-digital forms & documents."""
    from indra_ocr import textlayer

    textlayer.MAX_BAD_CHAR_FRAC = 0.05
    textlayer.MAX_NO_INK_LINE_FRAC = 0.30
    textlayer.DARK = 230

    def _bad_char_count_lenient(text: str) -> int:
        n = 0
        for c in text:
            cat = unicodedata.category(c)
            if c == "\ufffd" or cat in ("Cs", "Cn") or (cat == "Cc" and c != "\t"):
                n += 1
        return n

    textlayer.bad_char_count = _bad_char_count_lenient


class PPDocLayoutSONNX:
    """Pure-ONNX 4.7 MB PP-DocLayout-S detector (~73 ms/page on 2 ARM64 CPU cores)."""

    def __init__(self, model_path: Path | None = None, threads: int = 2):
        path = model_path or (MODELS_ROOT / "pp_doclayout_s.onnx")
        self.sess = _ort_session(path, threads=threads)

    def detect(self, gray: np.ndarray, conf_thresh: float = 0.45) -> list[tuple[str, float, list[float]]]:
        h, w = gray.shape[:2]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray.ndim == 2 else gray
        rs = (cv2.resize(rgb, (480, 480), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        inp = rs.transpose(2, 0, 1)[None]
        sf = np.array([[480.0 / float(h), 480.0 / float(w)]], dtype=np.float32)
        boxes_out = self.sess.run(None, {"image": inp, "scale_factor": sf})[0]
        res: list[tuple[str, float, list[float]]] = []
        for b in boxes_out:
            cid, sc = int(b[0]), float(b[1])
            if 0 <= cid < len(DOCLAYOUT_S_LABELS) and sc >= conf_thresh:
                x0 = max(0.0, min(float(w), float(b[2])))
                y0 = max(0.0, min(float(h), float(b[3])))
                x1 = max(0.0, min(float(w), float(b[4])))
                y1 = max(0.0, min(float(h), float(b[5])))
                if x1 > x0 and y1 > y0:
                    res.append((DOCLAYOUT_S_LABELS[cid], sc, [x0, y0, x1, y1]))
        res.sort(key=lambda x: -x[1])
        kept: list[tuple[str, float, list[float]]] = []
        for lbl, sc, bx in res:
            x0, y0, x1, y1 = bx
            area = max(1.0, (x1 - x0) * (y1 - y0))
            dup = False
            for klbl, _, kbx in kept:
                if klbl != lbl:
                    continue
                kx0, ky0, kx1, ky1 = kbx
                ix = max(0.0, min(x1, kx1) - max(x0, kx0))
                iy = max(0.0, min(y1, ky1) - max(y0, ky0))
                inter = ix * iy
                karea = max(1.0, (kx1 - kx0) * (ky1 - ky0))
                if inter / min(area, karea) >= 0.50:
                    dup = True
                    break
            if not dup:
                kept.append((lbl, sc, bx))
        return kept


class SLANetPlusONNX:
    """Pure-ONNX 7.4 MB SLANet-plus table structure predictor (zero RapidTable/PaddlePaddle runtime)."""

    def __init__(self, model_path: Path | None = None, threads: int = 2):
        path = model_path or (MODELS_ROOT / "slanet-plus.onnx")
        self.sess = _ort_session(path, threads=threads)
        self.in_name = self.sess.get_inputs()[0].name
        meta_map = self.sess.get_modelmeta().custom_metadata_map
        if "character" in meta_map:
            meta_chars = meta_map["character"].splitlines()
        else:
            yml_path = next((p for p in (path.with_suffix(".yml"), path.parent / "inference.yml") if p.is_file()), None)
            if yml_path is None:
                raise FileNotFoundError(f"Missing SLANet character metadata in {path} and no .yml sidecar found")
            cfg = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
            meta_chars = list(cfg["PostProcess"]["character_dict"])
        self.chars = ["sos"] + meta_chars + ["eos"]

    def predict(self, crop_bgr: np.ndarray) -> tuple[list[str], list[list[float]]]:
        h, w = crop_bgr.shape[:2]
        max_side = float(max(h, w))
        ratio = 488.0 / max_side
        rw, rh = max(1, int(w * ratio)), max(1, int(h * ratio))
        rs = cv2.resize(crop_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        rs = (rs - _IMAGENET_MEAN) / _IMAGENET_STD
        pad = np.zeros((488, 488, 3), dtype=np.float32)
        pad[:rh, :rw, :] = rs
        inp = pad.transpose(2, 0, 1)[None, :, :, :]
        outs = self.sess.run(None, {self.in_name: inp})
        loc = next(x for x in outs if x.shape[-1] in (4, 8))
        probs = next(x for x in outs if x.shape[-1] == len(self.chars))
        ids = probs[0].argmax(-1)
        tokens: list[str] = []
        bboxes: list[list[float]] = []
        for n, k in enumerate(ids):
            ch = self.chars[int(k)]
            if n > 0 and ch == "eos":
                break
            if ch in ("sos", "eos"):
                continue
            tokens.append(ch)
            if ch in ("<td>", "<td", "<td></td>"):
                raw = loc[0, n] * max_side
                bboxes.append(raw.tolist())
        return tokens, bboxes


def _promote_first_row_to_th(html_str: str) -> str:
    if "<th" in html_str.lower():
        return html_str
    m_thead = re.search(r"(<thead\b[^>]*>)(.*?)(</thead>)", html_str, flags=re.I | re.S)
    if m_thead:
        inner = re.sub(r"<(/?)td\b", r"<\1th", m_thead.group(2), flags=re.I)
        return html_str[: m_thead.start()] + m_thead.group(1) + inner + m_thead.group(3) + html_str[m_thead.end() :]
    m_tr = re.search(r"(<tr\b[^>]*>)(.*?)(</tr>)", html_str, flags=re.I | re.S)
    if m_tr:
        inner = re.sub(r"<(/?)td\b", r"<\1th", m_tr.group(2), flags=re.I)
        return html_str[: m_tr.start()] + m_tr.group(1) + inner + m_tr.group(3) + html_str[m_tr.end() :]
    return html_str


def match_and_build_slanet_html(
    pred_struct: list[str],
    cell_bboxes: list[list[float]],
    boxes: np.ndarray,
    texts: list[str],
) -> tuple[str, float, float]:
    """100%-fragment-conserving matcher from Indra cell boxes/texts into SLANet-plus structure."""
    n_cells = len(cell_bboxes)
    if n_cells == 0:
        return "", 0.0, 1.0

    c_rects = np.zeros((n_cells, 4), dtype=np.float32)
    for j, pb in enumerate(cell_bboxes):
        if len(pb) == 8:
            c_rects[j] = [
                float(np.min(pb[0::2])),
                float(np.min(pb[1::2])),
                float(np.max(pb[0::2])),
                float(np.max(pb[1::2])),
            ]
        else:
            c_rects[j] = [float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3])]

    matched: dict[int, list[int]] = {}
    iou_hits = 0
    for i, gt_box in enumerate(boxes):
        gx0, gy0, gx1, gy1 = float(gt_box[0]), float(gt_box[1]), float(gt_box[2]), float(gt_box[3])
        g_area = max(1e-6, (gx1 - gx0) * (gy1 - gy0))

        best_key = None
        best_j = 0
        best_iou = 0.0
        for j in range(n_cells):
            cx0, cy0, cx1, cy1 = c_rects[j]
            ix = max(0.0, min(gx1, cx1) - max(gx0, cx0))
            iy = max(0.0, min(gy1, cy1) - max(gy0, cy0))
            inter = ix * iy
            c_area = max(1e-6, (cx1 - cx0) * (cy1 - cy0))
            iou = inter / max(1e-6, g_area + c_area - inter)
            ov_frac = inter / g_area
            d1 = abs(cx0 - gx0) + abs(cy0 - gy0)
            d2 = abs(cx1 - gx1) + abs(cy1 - gy1)
            dist = d1 + d2 + min(d1, d2)
            key = (1.0 - max(iou, 0.5 * ov_frac), dist)
            if best_key is None or key < best_key:
                best_key = key
                best_j = j
                best_iou = max(iou, ov_frac)
        if best_iou > 0.05:
            iou_hits += 1
        matched.setdefault(best_j, []).append(i)

    for idxs in matched.values():
        idxs.sort(key=lambda idx: (float(boxes[idx][1]), float(boxes[idx][0]), idx))

    horiz_merges = 0
    for idxs in matched.values():
        if len(idxs) < 2:
            continue
        b_sub = np.asarray([boxes[idx][:4] for idx in idxs], dtype=np.float32)
        for a in range(len(idxs)):
            merged_flag = False
            for b in range(a + 1, len(idxs)):
                ha = max(1.0, b_sub[a, 3] - b_sub[a, 1])
                hb = max(1.0, b_sub[b, 3] - b_sub[b, 1])
                vy = min(b_sub[a, 3], b_sub[b, 3]) - max(b_sub[a, 1], b_sub[b, 1])
                vx = min(b_sub[a, 2], b_sub[b, 2]) - max(b_sub[a, 0], b_sub[b, 0])
                if vy > 0.4 * min(ha, hb) and vx < -2.0:
                    horiz_merges += 1
                    merged_flag = True
                    break
            if merged_flag:
                break

    end_html: list[str] = ["<table>"]
    td_index = 0
    in_thead = False
    for tag in pred_struct:
        if tag in ("<html>", "</html>", "<body>", "</body>", "<table>", "</table>"):
            continue
        if tag == "<thead>":
            in_thead = True
            end_html.append(tag)
            continue
        if tag == "</thead>":
            in_thead = False
            end_html.append(tag)
            continue
        if "</td>" not in tag:
            if in_thead and tag.startswith("<td"):
                tag = "<th" + tag[3:]
            end_html.append(tag)
            continue

        open_tag = "<th>" if in_thead else "<td>"
        close_tag = "</th>" if in_thead else "</td>"
        if tag == "<td></td>":
            end_html.append(open_tag)

        if td_index in matched:
            parts = [
                _html.escape(str(texts[idx]).strip(), quote=False)
                for idx in matched[td_index]
                if str(texts[idx]).strip()
            ]
            end_html.append("<br>".join(parts))

        if tag == "<td></td>":
            end_html.append(close_tag)
        else:
            end_html.append(close_tag if in_thead else tag)
        td_index += 1

    end_html.append("</table>")
    html_out = _promote_first_row_to_th("".join(end_html))
    iou_rate = iou_hits / max(1, len(boxes))
    horiz_merge_rate = horiz_merges / max(1, len(matched))
    return html_out, iou_rate, horiz_merge_rate


DEEPDOC_TSR_LABELS = (
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
)


def _dedup_1d_intervals(
    intervals: list[tuple[float, float, float]],
    overlap_thr: float = 0.50,
) -> list[tuple[float, float]]:
    ordered = sorted(intervals, key=lambda z: -z[2])
    kept: list[tuple[float, float]] = []
    for a, b, _sc in ordered:
        if b <= a + 1.0:
            continue
        dup = False
        for ka, kb in kept:
            ov = max(0.0, min(b, kb) - max(a, ka))
            if ov / max(1.0, min(b - a, kb - ka)) >= overlap_thr:
                dup = True
                break
        if not dup:
            kept.append((a, b))
    kept.sort(key=lambda z: 0.5 * (z[0] + z[1]))
    return kept


def _align_tsr_boxes(dets: dict[str, list[tuple[float, float, float, float, float]]]) -> None:
    row_keys = ("table row", "table column header", "table projected row header")
    left_vals, right_vals = [], []
    for k in row_keys:
        for bx in dets.get(k, []):
            left_vals.append(bx[0])
            right_vals.append(bx[2])
    if left_vals:
        left = float(np.mean(left_vals)) if len(left_vals) > 4 else float(min(left_vals))
        right = float(np.mean(right_vals)) if len(right_vals) > 4 else float(max(right_vals))
        for k in row_keys:
            dets[k] = [
                (min(bx[0], left), bx[1], max(bx[2], right), bx[3], bx[4])
                for bx in dets.get(k, [])
            ]
    cols = dets.get("table column", [])
    if cols:
        top_vals = [bx[1] for bx in cols]
        bot_vals = [bx[3] for bx in cols]
        top = float(np.median(top_vals)) if len(top_vals) > 4 else float(min(top_vals))
        bot = float(np.median(bot_vals)) if len(bot_vals) > 4 else float(max(bot_vals))
        dets["table column"] = [
            (bx[0], min(bx[1], top), bx[2], max(bx[3], bot), bx[4])
            for bx in cols
        ]


class DeepDocTSRONNX:
    """Pure-ONNX 12.2 MB InfiniFlow/deepdoc PubTables-1M TSR detector (~171 ms on 2 ARM64 cores)."""

    def __init__(self, model_path: Path | None = None, threads: int = 2):
        path = model_path or (MODELS_ROOT / "deepdoc_tsr.onnx")
        self.sess = _ort_session(path, threads=threads)
        self.in_name = self.sess.get_inputs()[0].name

    def detect(
        self,
        crop_bgr: np.ndarray,
        *,
        score_thr: float = 0.25,
        nms_iou: float = 0.20,
    ) -> dict[str, list[tuple[float, float, float, float, float]]]:
        H, W = crop_bgr.shape[:2]
        rs = cv2.resize(crop_bgr, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        inp = rs.transpose(2, 0, 1)[None, :, :, :]
        out = self.sess.run(None, {self.in_name: inp})[0][0]
        cls_scores = out[4 : 4 + len(DEEPDOC_TSR_LABELS), :]
        best_cls = cls_scores.argmax(axis=0)
        best_sc = cls_scores.max(axis=0)
        keep_mask = best_sc > score_thr
        by_lbl: dict[str, list[tuple[float, float, float, float, float]]] = {l: [] for l in DEEPDOC_TSR_LABELS}
        if not keep_mask.any():
            return by_lbl

        idxs = np.nonzero(keep_mask)[0]
        sfx, sfy = float(W) / 640.0, float(H) / 640.0
        cx = out[0, idxs] * sfx
        cy = out[1, idxs] * sfy
        hw = out[2, idxs] * sfx * 0.5
        hh = out[3, idxs] * sfy * 0.5
        x0 = np.clip(cx - hw, 0.0, float(W))
        y0 = np.clip(cy - hh, 0.0, float(H))
        x1 = np.clip(cx + hw, 0.0, float(W))
        y1 = np.clip(cy + hh, 0.0, float(H))
        clss = best_cls[idxs]
        scs = best_sc[idxs]

        for c_i, lbl in enumerate(DEEPDOC_TSR_LABELS):
            m = clss == c_i
            if not m.any():
                continue
            bx = np.stack([x0[m], y0[m], x1[m], y1[m], scs[m]], axis=1)
            bx = bx[np.argsort(-bx[:, 4])]
            kept: list[tuple[float, float, float, float, float]] = []
            for row in bx:
                ax0, ay0, ax1, ay1, asc = [float(v) for v in row]
                a_area = max(1e-3, (ax1 - ax0) * (ay1 - ay0))
                sup = False
                for kx0, ky0, kx1, ky1, _ in kept:
                    ix = max(0.0, min(ax1, kx1) - max(ax0, kx0))
                    iy = max(0.0, min(ay1, ky1) - max(ay0, ky0))
                    inter = ix * iy
                    k_area = max(1e-3, (kx1 - kx0) * (ky1 - ky0))
                    if inter / max(1e-3, a_area + k_area - inter) > nms_iou:
                        sup = True
                        break
                if not sup:
                    kept.append((ax0, ay0, ax1, ay1, asc))
            by_lbl[lbl] = kept
        return by_lbl

    def build_html(
        self,
        crop_bgr: np.ndarray,
        crop_boxes: np.ndarray,
        texts: list[str],
    ) -> tuple[str | None, float, float, int, int, float]:
        dets = self.detect(crop_bgr, score_thr=0.25, nms_iou=0.20)
        _align_tsr_boxes(dets)
        raw_rows = dets.get("table row", [])
        raw_cols = dets.get("table column", [])
        col_headers = dets.get("table column header", [])
        proj_headers = dets.get("table projected row header", [])
        span_cells = dets.get("table spanning cell", [])
        mean_row_sc = float(np.mean([b[4] for b in raw_rows])) if raw_rows else 0.0

        rows = _dedup_1d_intervals([(y0, y1, sc) for (_, y0, _, y1, sc) in raw_rows], overlap_thr=0.50)
        cols = _dedup_1d_intervals([(x0, x1, sc) for (x0, _, x1, _, sc) in raw_cols], overlap_thr=0.50)

        for _px0, py0, _px1, py1, _psc in proj_headers:
            covered = any(
                max(0.0, min(py1, ry1) - max(py0, ry0)) / max(1.0, min(py1 - py0, ry1 - ry0)) >= 0.45
                for ry0, ry1 in rows
            )
            if not covered and py1 > py0 + 2.0:
                rows.append((py0, py1))
        rows.sort(key=lambda z: 0.5 * (z[0] + z[1]))

        R, C = len(rows), len(cols)
        if R < 2 or C < 2:
            return None, 0.0, 1.0, 0, 0, mean_row_sc

        header_rows: set[int] = set()
        for r_idx, (ry0, ry1) in enumerate(rows):
            rh = max(1.0, ry1 - ry0)
            for _hx0, hy0, _hx1, hy1, _hsc in col_headers:
                if max(0.0, min(ry1, hy1) - max(ry0, hy0)) / rh >= 0.45:
                    header_rows.add(r_idx)
                    break
        if header_rows:
            max_h = -1
            for r_idx in range(R):
                if r_idx in header_rows:
                    max_h = r_idx
                else:
                    break
            header_rows = set(range(max_h + 1)) if max_h >= 0 else set()
        if not header_rows:
            header_rows = {0}

        proj_rows: set[int] = set()
        for r_idx, (ry0, ry1) in enumerate(rows):
            if r_idx in header_rows:
                continue
            rh = max(1.0, ry1 - ry0)
            for _px0, py0, _px1, py1, psc in proj_headers:
                if psc < 0.35:
                    continue
                if max(0.0, min(ry1, py1) - max(ry0, py0)) / rh >= 0.50:
                    proj_rows.add(r_idx)
                    break

        spans: dict[tuple[int, int], tuple[int, int]] = {}
        owner: dict[tuple[int, int], tuple[int, int]] = {}
        for r_idx in sorted(proj_rows):
            spans[(r_idx, 0)] = (1, C)
            for c_idx in range(C):
                owner[(r_idx, c_idx)] = (r_idx, 0)

        for sx0, sy0, sx1, sy1, ssc in sorted(span_cells, key=lambda z: -z[4]):
            if ssc < 0.35:
                continue
            r_hits = [
                ri for ri, (ry0, ry1) in enumerate(rows)
                if max(0.0, min(sy1, ry1) - max(sy0, ry0)) / max(1.0, ry1 - ry0) >= 0.45
            ]
            c_hits = [
                ci for ci, (cx0, cx1) in enumerate(cols)
                if max(0.0, min(sx1, cx1) - max(sx0, cx0)) / max(1.0, cx1 - cx0) >= 0.45
            ]
            if not r_hits or not c_hits:
                continue
            r0, r1 = min(r_hits), max(r_hits)
            c0, c1 = min(c_hits), max(c_hits)
            rs, cs = r1 - r0 + 1, c1 - c0 + 1
            if rs == 1 and cs == 1:
                continue
            in_h = [r in header_rows for r in range(r0, r1 + 1)]
            if any(in_h) and not all(in_h):
                continue
            if any((rr, cc) in owner for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)):
                continue
            spans[(r0, c0)] = (rs, cs)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    owner[(rr, cc)] = (r0, c0)

        for rr in range(R):
            for cc in range(C):
                if (rr, cc) not in owner:
                    spans[(rr, cc)] = (1, 1)
                    owner[(rr, cc)] = (rr, cc)

        cell_frags: dict[tuple[int, int], list[int]] = {}
        hits = 0
        for i, gb in enumerate(crop_boxes):
            gx0, gy0, gx1, gy1 = [float(v) for v in gb[:4]]
            gcx, gcy = 0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1)
            gh, gw = max(1.0, gy1 - gy0), max(1.0, gx1 - gx0)

            best_r, best_rov, best_rd = 0, -1.0, 1e9
            for ri, (ry0, ry1) in enumerate(rows):
                ov = max(0.0, min(gy1, ry1) - max(gy0, ry0)) / gh
                d = abs(gcy - 0.5 * (ry0 + ry1))
                if ov > best_rov + 1e-4 or (abs(ov - best_rov) <= 1e-4 and d < best_rd):
                    best_r, best_rov, best_rd = ri, ov, d

            best_c, best_cov, best_cd = 0, -1.0, 1e9
            for ci, (cx0, cx1) in enumerate(cols):
                ov = max(0.0, min(gx1, cx1) - max(gx0, cx0)) / gw
                d = abs(gcx - 0.5 * (cx0 + cx1))
                if ov > best_cov + 1e-4 or (abs(ov - best_cov) <= 1e-4 and d < best_cd):
                    best_c, best_cov, best_cd = ci, ov, d

            if best_rov > 0.1 and best_cov > 0.1:
                hits += 1
            anc = owner[(best_r, best_c)]
            cell_frags.setdefault(anc, []).append(i)

        horiz_merges = 0
        for anc, idxs in cell_frags.items():
            idxs.sort(key=lambda idx: (float(crop_boxes[idx][1]), float(crop_boxes[idx][0]), idx))
            if len(idxs) >= 2 and spans[anc][1] == 1:
                b_sub = np.asarray([crop_boxes[idx][:4] for idx in idxs], dtype=np.float32)
                found = False
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        ha = max(1.0, b_sub[a, 3] - b_sub[a, 1])
                        hb = max(1.0, b_sub[b, 3] - b_sub[b, 1])
                        vy = min(b_sub[a, 3], b_sub[b, 3]) - max(b_sub[a, 1], b_sub[b, 1])
                        vx = min(b_sub[a, 2], b_sub[b, 2]) - max(b_sub[a, 0], b_sub[b, 0])
                        if vy > 0.4 * min(ha, hb) and vx < -4.0:
                            horiz_merges += 1
                            found = True
                            break
                    if found:
                        break

        out_html: list[str] = ["<table>"]
        has_thead = len(header_rows) > 0 and len(header_rows) < R
        if has_thead:
            out_html.append("<thead>")
        in_tbody = False
        for rr in range(R):
            if has_thead and rr not in header_rows and not in_tbody:
                out_html.append("</thead><tbody>")
                in_tbody = True
            out_html.append("<tr>")
            for cc in range(C):
                if owner[(rr, cc)] != (rr, cc):
                    continue
                rs, cs = spans[(rr, cc)]
                tag = "th" if (rr in header_rows or rr in proj_rows) else "td"
                attrs = ""
                if rs > 1:
                    attrs += f' rowspan="{rs}"'
                if cs > 1:
                    attrs += f' colspan="{cs}"'
                idxs = cell_frags.get((rr, cc), [])
                parts = [_html.escape(str(texts[k]).strip(), quote=False) for k in idxs if str(texts[k]).strip()]
                out_html.append(f"<{tag}{attrs}>{'<br>'.join(parts)}</{tag}>")
            out_html.append("</tr>")
        if in_tbody:
            out_html.append("</tbody>")
        elif has_thead:
            out_html.append("</thead>")
        out_html.append("</table>")

        html_str = "".join(out_html)
        iou_rate = hits / max(1, len(crop_boxes))
        horiz_rate = horiz_merges / max(1, len(cell_frags))
        cells_all = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", html_str, flags=re.I | re.S)
        n_cells = len(cells_all)
        n_empty = sum(1 for c in cells_all if not c.strip())
        n_nonempty = sum(1 for x in texts if str(x).strip())
        n_spans = len(re.findall(r"(?:colspan|rowspan)=", html_str, flags=re.I))
        if (
            iou_rate < 0.50
            or horiz_rate > 0.18
            or (n_cells > 2.4 * max(1, n_nonempty) and (n_empty / max(1, n_cells)) > 0.55)
        ):
            return None, iou_rate, horiz_rate, len(proj_rows), n_spans, mean_row_sc
        return html_str, iou_rate, horiz_rate, len(proj_rows), n_spans, mean_row_sc


class Pix2TextMFRKVONNX:
    """KV-cached ONNX pix2text-mfr-1.5 (~319 ms/eq on 2 ARM64 CPU cores, 9.6x faster than uncached)."""

    def __init__(self, model_dir: Path | None = None, threads: int = 2):
        d = model_dir or (MODELS_ROOT / "pix2text_mfr15")
        self.enc = _ort_session(d / "encoder_model.onnx", threads=threads)
        self.dec_init = _ort_session(d / "mfr_dec_init.onnx", threads=threads)
        self.dec_step = _ort_session(d / "mfr_dec_step.onnx", threads=1)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))

    def recognize_batch(
        self,
        crops_gray: list[np.ndarray],
        *,
        max_new_tokens: int = 256,
        min_mfr_conf: float = 0.65,
    ) -> list[str | None]:
        if not crops_gray:
            return []
        B = len(crops_gray)
        batch_px = np.zeros((B, 3, 384, 384), dtype=np.float32)
        for i, cg in enumerate(crops_gray):
            rgb = cv2.cvtColor(cg, cv2.COLOR_GRAY2RGB) if cg.ndim == 2 else cg
            pil_img = Image.fromarray(rgb).resize((384, 384), resample=Image.Resampling.BICUBIC)
            arr = (np.asarray(pil_img, dtype=np.float32) / 255.0 - 0.5) / 0.5
            batch_px[i] = arr.transpose(2, 0, 1)

        enc_out = self.enc.run(None, {"pixel_values": batch_px})[0]
        ck_all, cv_all = self.dec_init.run(None, {"encoder_hidden_states": enc_out})
        bos_id, eos_id = 1, 2
        out_texts: list[str | None] = []
        step_cap = min(max_new_tokens, 256)
        for i in range(B):
            ck_i = ck_all[:, i : i + 1]
            cv_i = cv_all[:, i : i + 1]
            sk_i = np.zeros((6, 1, 8, 32, 0), dtype=np.float32)
            sv_i = np.zeros((6, 1, 8, 0, 32), dtype=np.float32)
            tok = np.array([bos_id], dtype=np.int64)
            seq: list[int] = []
            probs: list[float] = []
            hit_eos = False
            for t in range(step_cap):
                logits, sk_i, sv_i = self.dec_step.run(
                    None,
                    {
                        "token": tok,
                        "pos": np.array(t, dtype=np.int64),
                        "self_k": sk_i,
                        "self_v": sv_i,
                        "cross_k": ck_i,
                        "cross_v": cv_i,
                    },
                )
                row = logits[0]
                nt = int(row.argmax())
                e = np.exp(row - row[nt])
                p_top = float(1.0 / e.sum())
                probs.append(p_top)
                if nt == eos_id:
                    hit_eos = True
                    break
                if nt not in (0, 1, 2):
                    seq.append(nt)
                if len(probs) >= 8 and float(np.mean(probs)) < (min_mfr_conf - 0.20):
                    break
                if len(seq) >= 16 and len(set(seq[-12:])) <= 2:
                    break
                tok = np.array([nt], dtype=np.int64)

            mean_conf = float(np.mean(probs)) if probs else 0.0
            if not hit_eos or not seq or mean_conf < min_mfr_conf:
                out_texts.append(None)
                continue
            txt = self.tok.decode(seq, skip_special_tokens=True).strip()
            if txt.count("~") > 12 and txt.count("~") > 0.25 * len(txt):
                out_texts.append(None)
                continue
            for l_del, r_del in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
                if txt.startswith(l_del) and txt.endswith(r_del) and len(txt) > len(l_del) + len(r_del):
                    txt = txt[len(l_del) : -len(r_del)].strip()
                    break
            out_texts.append(txt if len(txt) >= 1 else None)
        return out_texts


class FastCPUStack:
    """Installs and coordinates all 4 verified pure-ONNX upgrades on an Indra OrtOCR / Converter."""

    def __init__(self, ocr: Any, threads: int = 2, models_root: Path | None = None):
        self.ocr = ocr
        self.models_root = models_root or MODELS_ROOT
        self.enabled = False
        self.enable_math_mfr = True
        self.page_stats: dict[str, Any] = {}
        self._reset_page_stats()

        install_textlayer_trust_fix()
        self.layout = PPDocLayoutSONNX(self.models_root / "pp_doclayout_s.onnx", threads=threads)
        self.slanet = SLANetPlusONNX(self.models_root / "slanet-plus.onnx", threads=threads)
        tsr_path = self.models_root / "deepdoc_tsr.onnx"
        self.deepdoc_tsr = DeepDocTSRONNX(tsr_path, threads=threads) if tsr_path.is_file() else None
        self.mfr = Pix2TextMFRKVONNX(self.models_root / "pix2text_mfr15", threads=threads)

        self._install_hooks()

    def _reset_page_stats(self) -> None:
        self.page_stats = {
            "tables_attempted": 0,
            "tables_replaced": 0,
            "tables_tsr_replaced": 0,
            "tables_sparse_fallback": 0,
            "tables_error_fallback": 0,
            "formula_boxes_detected": 0,
            "formula_boxes_replaced": 0,
            "formula_boxes_inserted": 0,
            "formula_boxes_guarded": 0,
            "lm_adapt_boosted": False,
            "regions": [],
        }

    def begin_page(self, *, enabled: bool = True, enable_math_mfr: bool = True) -> None:
        self.enabled = enabled
        self.enable_math_mfr = enable_math_mfr
        self.ocr._current_gray = None
        self.ocr._last_layout_det = None
        self._reset_page_stats()

    def _install_hooks(self) -> None:
        from indra_ocr import layout_post, pipeline as indra_pipeline

        ocr = self.ocr
        orig_detect = ocr.detect
        orig_detect_all = ocr.detect_all
        orig_recognize = ocr.recognize
        orig_assemble = ocr.assemble
        orig_table_html = layout_post.table_html
        base_id = id(ocr.rec)
        base_orig_tup = getattr(ocr, "lms", {}).get(base_id)

        def _wrapped_detect(gray: np.ndarray):
            ocr._current_gray = gray
            ocr._last_layout_det = None
            return orig_detect(gray)

        def _wrapped_detect_all(gray: np.ndarray):
            ocr._current_gray = gray
            ocr._last_layout_det = None
            return orig_detect_all(gray)

        def _recognize_adaptive(gray: np.ndarray, boxes: np.ndarray, batch: int = 64, vertical: bool = False):
            ocr._current_gray = gray
            if not self.enabled or base_orig_tup is None or vertical or len(boxes) < 4:
                return orig_recognize(gray, boxes, batch=batch, vertical=vertical)
            ocr.lms[base_id] = base_orig_tup
            indra_pipeline.LM_BEAM = 8
            indra_pipeline.LM_CAND_FLOOR = 3e-3
            indra_pipeline.MIN_CONF = 0.50
            out = orig_recognize(gray, boxes, batch=batch, vertical=vertical)
            confs = [c for (t, c, _) in out if c > 0.0 and "$" not in (t or "") and c < 0.9999]
            if len(confs) >= 4 and float(np.median(confs)) < 0.91:
                self.page_stats["lm_adapt_boosted"] = True
                lm_obj, alpha, beta, _gate = base_orig_tup
                ocr.lms[base_id] = (lm_obj, alpha, beta, 0.985)
                indra_pipeline.LM_BEAM = 16
                indra_pipeline.LM_CAND_FLOOR = 1e-3
                try:
                    out = orig_recognize(gray, boxes, batch=batch, vertical=vertical)
                finally:
                    ocr.lms[base_id] = base_orig_tup
                    indra_pipeline.LM_BEAM = 8
                    indra_pipeline.LM_CAND_FLOOR = 3e-3
            return out

        def _assemble_with_mfr(wh, boxes, rec, t, vertical=False, pictures=None):
            gray = getattr(ocr, "_current_gray", None)
            if self.enabled and self.enable_math_mfr and gray is not None and not vertical:
                try:
                    dets = ocr._last_layout_det
                    if dets is None:
                        dets = self.layout.detect(gray, conf_thresh=0.28)
                        ocr._last_layout_det = dets
                        self.page_stats["regions"] = [
                            {"raw_label": lbl, "score": round(sc, 4), "bbox": [round(v, 1) for v in bx]}
                            for lbl, sc, bx in dets
                        ]
                    H, W = gray.shape[:2]
                    formula_conf = formula_conf_for_rec(rec)
                    self.page_stats["formula_conf_thresh"] = formula_conf
                    f_boxes: list[list[float]] = []
                    for lbl, sc, bx in dets:
                        if lbl == "formula" and sc >= formula_conf:
                            x0, y0, x1, y1 = bx
                            bw, bh = x1 - x0, y1 - y0
                            if bw >= 16 and bh >= 12 and bh <= 0.35 * H:
                                f_boxes.append([x0, y0, x1, y1])
                            elif bw >= 16 and bh > 0.35 * H:
                                self.page_stats["formula_boxes_guarded"] += 1

                    if f_boxes:
                        self.page_stats["formula_boxes_detected"] += len(f_boxes)
                        crops: list[np.ndarray] = []
                        padded_boxes: list[list[float]] = []
                        for x0, y0, x1, y1 in f_boxes:
                            px0 = max(0, int(np.floor(x0 - 6)))
                            py0 = max(0, int(np.floor(y0 - 6)))
                            px1 = min(W, int(np.ceil(x1 + 6)))
                            py1 = min(H, int(np.ceil(y1 + 6)))
                            crops.append(gray[py0:py1, px0:px1])
                            padded_boxes.append([float(px0), float(py0), float(px1), float(py1)])

                        latexes = self.mfr.recognize_batch(crops, max_new_tokens=256, min_mfr_conf=0.65)
                        n_lines = len(boxes)
                        replacements: dict[int, tuple[list[float], Any]] = {}
                        inserts: list[tuple[list[float], Any]] = []
                        drop_idx: set[int] = set()
                        b_arr = np.asarray(boxes[:, :4], dtype=np.float32) if n_lines else np.zeros((0, 4), np.float32)

                        for fb, ltx in zip(padded_boxes, latexes):
                            if not ltx or len(ltx) < 2:
                                self.page_stats["formula_boxes_guarded"] += 1
                                continue
                            fx0, fy0, fx1, fy1 = fb
                            covered = []
                            partial_line_overlap = False
                            for j in range(n_lines):
                                if j in drop_idx:
                                    continue
                                lx0, ly0, lx1, ly1 = b_arr[j]
                                cx, cy = 0.5 * (lx0 + lx1), 0.5 * (ly0 + ly1)
                                lw = max(1.0, lx1 - lx0)
                                lh = max(1.0, ly1 - ly0)
                                ov_x = max(0.0, min(lx1, fx1) - max(lx0, fx0))
                                ov_y = max(0.0, min(ly1, fy1) - max(ly0, fy0))
                                if fx0 <= cx <= fx1 and fy0 <= cy <= fy1 and (ov_x / lw) >= 0.55:
                                    covered.append(j)
                                elif ov_y >= 0.4 * lh and ov_x >= 0.15 * lw:
                                    partial_line_overlap = True
                            math_str = f"${ltx}$"
                            xc = 0.5 * (fx0 + fx1)
                            if covered:
                                first_j = min(covered)
                                first_toks = rec[first_j][2]
                                if first_toks and len(first_toks[0]) >= 3:
                                    seq_id = min(tk[2] for tk in first_toks if len(tk) >= 3)
                                    new_rec = (math_str, 1.0, [(math_str, xc, seq_id)])
                                else:
                                    new_rec = (math_str, 1.0, [(math_str, xc)])
                                drop_idx.update(covered)
                                replacements[first_j] = ([fx0, fy0, fx1, fy1], new_rec)
                                self.page_stats["formula_boxes_replaced"] += 1
                            elif not partial_line_overlap:
                                inserts.append(([fx0, fy0, fx1, fy1], (math_str, 1.0, [(math_str, xc)])))
                                self.page_stats["formula_boxes_inserted"] += 1
                            else:
                                self.page_stats["formula_boxes_guarded"] += 1

                        if replacements or inserts:
                            new_b: list[list[float]] = []
                            new_r: list[Any] = []
                            for j in range(n_lines):
                                if j in replacements:
                                    bx_r, rc_r = replacements[j]
                                    new_b.append(bx_r)
                                    new_r.append(rc_r)
                                elif j not in drop_idx:
                                    new_b.append(boxes[j, :4].tolist())
                                    new_r.append(rec[j])
                            for bx_i, rc_i in inserts:
                                new_b.append(bx_i)
                                new_r.append(rc_i)
                            boxes = np.array(new_b, dtype=np.float32).reshape(-1, 4)
                            rec = new_r
                except Exception:
                    pass
            return orig_assemble(wh, boxes, rec, t, vertical=vertical, pictures=pictures)

        def _slanet_table_html(boxes: np.ndarray, texts: list[str]) -> str:
            if not self.enabled:
                return orig_table_html(boxes, texts)
            rows = layout_post._rows(boxes)
            span_x = boxes[:, 2].max() - boxes[:, 0].min()
            if (
                len(rows) > 2
                and len(rows[0]) == 1
                and max(len(r) for r in rows) > 1
                and boxes[rows[0][0], 2] - boxes[rows[0][0], 0] > 0.6 * span_x
            ):
                keep = [k for k in range(len(texts)) if k != rows[0][0]]
                return (
                    texts[rows[0][0]]
                    + "\n\n"
                    + _slanet_table_html(boxes[keep], [texts[k] for k in keep])
                )

            native_html = orig_table_html(boxes, texts)
            gray = getattr(ocr, "_current_gray", None)
            if gray is None or getattr(ocr, "_page_rtl", False) or len(texts) < 2:
                return native_html

            self.page_stats["tables_attempted"] += 1
            try:
                H, W = gray.shape[:2]
                bx0, by0 = float(boxes[:, 0].min()), float(boxes[:, 1].min())
                bx1, by1 = float(boxes[:, 2].max()), float(boxes[:, 3].max())
                tw, th = max(1.0, bx1 - bx0), max(1.0, by1 - by0)
                med_lh = float(np.median(np.maximum(1.0, boxes[:, 3] - boxes[:, 1])))
                pad = max(4, min(round(0.03 * min(tw, th)), round(0.5 * med_lh)))
                cx0, cy0 = max(0, int(np.floor(bx0 - pad))), max(0, int(np.floor(by0 - pad)))
                cx1, cy1 = min(W, int(np.ceil(bx1 + pad))), min(H, int(np.ceil(by1 + pad)))
                if cx1 - cx0 < 24 or cy1 - cy0 < 24:
                    return native_html

                crop_bgr = cv2.cvtColor(gray[cy0:cy1, cx0:cx1], cv2.COLOR_GRAY2BGR)
                crop_boxes = np.zeros((len(boxes), 4), dtype=np.float32)
                crop_boxes[:, 0] = boxes[:, 0] - cx0
                crop_boxes[:, 1] = boxes[:, 1] - cy0
                crop_boxes[:, 2] = boxes[:, 2] - cx0
                crop_boxes[:, 3] = boxes[:, 3] - cy0

                str_texts = [str(x) for x in texts]
                slanet_ok, slanet_html, slanet_iou, slanet_hm = False, None, 0.0, 1.0
                pred_struct, cell_bboxes = self.slanet.predict(crop_bgr)
                if pred_struct and cell_bboxes:
                    pred_html, iou_rate, horiz_merge_rate = match_and_build_slanet_html(
                        pred_struct, cell_bboxes, crop_boxes, str_texts
                    )
                    if pred_html and "<table" in pred_html.lower():
                        cells_all = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", pred_html, flags=re.I | re.S)
                        n_cells = len(cells_all)
                        n_empty = sum(1 for c in cells_all if not c.strip())
                        n_nonempty_frags = sum(1 for x in str_texts if x.strip())
                        if not (
                            iou_rate < 0.45
                            or horiz_merge_rate > 0.15
                            or (n_cells > 2.2 * max(1, n_nonempty_frags) and (n_empty / max(1, n_cells)) > 0.50)
                        ):
                            slanet_ok, slanet_html, slanet_iou, slanet_hm = True, pred_html, iou_rate, horiz_merge_rate

                if self.deepdoc_tsr is not None:
                    tsr_html, tsr_iou, tsr_hm, n_proj, n_spans, mean_row_sc = self.deepdoc_tsr.build_html(
                        crop_bgr, crop_boxes, str_texts
                    )
                    if tsr_html is not None and (
                        not slanet_ok
                        or (n_proj >= 1 and tsr_hm <= slanet_hm + 0.05)
                        or (n_spans >= 3 and mean_row_sc >= 0.70 and tsr_hm <= slanet_hm + 0.05)
                        or (tsr_iou > slanet_iou + 0.10 and tsr_hm <= slanet_hm)
                    ):
                        self.page_stats["tables_tsr_replaced"] += 1
                        self.page_stats["tables_replaced"] += 1
                        return tsr_html

                if slanet_ok and slanet_html is not None:
                    self.page_stats["tables_replaced"] += 1
                    return slanet_html

                self.page_stats["tables_sparse_fallback"] += 1
                return native_html
            except Exception:
                self.page_stats["tables_error_fallback"] += 1
                return native_html

        ocr.detect = _wrapped_detect
        ocr.detect_all = _wrapped_detect_all
        ocr.recognize = _recognize_adaptive
        ocr.assemble = _assemble_with_mfr
        layout_post.table_html = _slanet_table_html
