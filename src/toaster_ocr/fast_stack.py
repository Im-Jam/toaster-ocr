"""Pure-ONNX CPU math detection & KV-cached MFR splicing on top of LiteParse (OCRia-nano-1)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

from .paths import stack_models_dir

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


def is_textlayer_page(text_items: Sequence[Any]) -> bool:
    """Return True when most LiteParse TextItems came from a native PDF text layer."""
    n = len(text_items)
    if n < 4:
        return False
    n_ocr = sum(1 for ti in text_items if getattr(ti, "font_name", "") == "OCR")
    return (n_ocr / n) < 0.35


def formula_conf_for_page(is_textlayer: bool) -> float:
    """Conservative PP-DocLayout-S formula threshold: 0.35 on PDF text layers, 0.28 on scans."""
    return 0.35 if is_textlayer else 0.28


class PPDocLayoutSONNX:
    """Pure-ONNX 4.7 MB PP-DocLayout-S detector (~73 ms/page on 2 ARM64 CPU cores)."""

    def __init__(self, model_path: Path | None = None, threads: int = 2):
        path = model_path or (stack_models_dir() / "pp_doclayout_s.onnx")
        self.sess = _ort_session(Path(path), threads=threads)

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


class Pix2TextMFRKVONNX:
    """KV-cached pure-ONNX `pix2text-mfr-1.5` (34.3 MB QInt8, ~200-320 ms/eq on 2 ARM64 cores)."""

    def __init__(self, model_dir: Path | None = None, threads: int = 2):
        d = Path(model_dir or (stack_models_dir() / "pix2text_mfr15"))
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


def box_xyxy(bx: Any) -> list[float]:
    return [float(bx.x), float(bx.y), float(bx.x + bx.width), float(bx.y + bx.height)]


def overlap_frac(a: list[float], b: list[float]) -> tuple[float, float]:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    ba = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / aa, inter / ba


def block_text(b: Any, text_items: Sequence[Any]) -> str:
    if getattr(b, "text", None):
        return str(b.text)
    if getattr(b, "rows", None):
        parts = []
        for row in b.rows:
            for cell in row:
                t = getattr(cell, "text", "") or ""
                if t:
                    parts.append(t)
        return " ".join(parts)
    idxs = getattr(b, "text_item_indices", None) or []
    return " ".join(text_items[i].text for i in idxs if 0 <= i < len(text_items))


def filter_formula_boxes(
    dets: Sequence[tuple[str, float, list[float]]],
    blocks: Sequence[Any],
    *,
    img_h: int,
    scale_x: float,
    scale_y: float,
    conf_thresh: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Keep valid display-formula boxes outside LayoutNet tables."""
    table_boxes_pdf = [
        box_xyxy(b.bbox)
        for b in blocks
        if getattr(b, "kind", None) in ("table", "merged_table") and getattr(b, "bbox", None) is not None
    ]
    f_px: list[list[float]] = []
    f_pdf: list[list[float]] = []
    for lbl, sc, bx in dets:
        if lbl != "formula" or sc < conf_thresh:
            continue
        x0, y0, x1, y1 = bx
        bw, bh = x1 - x0, y1 - y0
        if bw < 16.0 or bh < 12.0 or bh > 0.35 * img_h:
            continue
        fb_pdf = [x0 / scale_x, y0 / scale_y, x1 / scale_x, y1 / scale_y]
        if any(overlap_frac(fb_pdf, tb)[0] >= 0.35 for tb in table_boxes_pdf):
            continue
        f_px.append([x0, y0, x1, y1])
        f_pdf.append(fb_pdf)
    return f_px, f_pdf


def splice_formulas_into_markdown(
    md: str,
    blocks: Sequence[Any],
    text_items: Sequence[Any],
    f_boxes_pdf: Sequence[list[float]],
    latex_list: Sequence[str | None],
    *,
    is_textlayer: bool,
) -> str:
    """Surgically splice LaTeX equations into LiteParse Markdown without disturbing tables or inline math."""
    out_md = md
    appended: list[str] = []

    for fb, raw_lx in zip(f_boxes_pdf, latex_list):
        lx = (raw_lx or "").strip()
        if not lx:
            continue
        eq_block = f"\n\n$$\n{lx}\n$$\n\n"
        best_b = None
        best_score = 0.0
        for b in blocks:
            if getattr(b, "kind", None) in ("table", "merged_table", "rule", "figure") or getattr(b, "bbox", None) is None:
                continue
            fa, ffb = overlap_frac(box_xyxy(b.bbox), fb)
            if fa >= 0.50 or ffb >= 0.65:
                sc = max(fa, ffb)
                if sc > best_score:
                    best_score = sc
                    best_b = b

        if best_b is not None:
            b_txt = block_text(best_b, text_items).strip()
            fa_b, _ = overlap_frac(box_xyxy(best_b.bbox), fb)
            is_standalone = (fa_b >= 0.55) and (len(re.findall(r"[A-Za-z]{4,}", b_txt)) <= 3)
            if b_txt and b_txt in out_md:
                if is_standalone and not is_textlayer:
                    out_md = out_md.replace(b_txt, eq_block, 1)
                else:
                    out_md = out_md.replace(b_txt, b_txt + eq_block, 1)
                continue
        appended.append(eq_block)

    if appended:
        out_md = out_md.rstrip() + "\n\n" + "\n".join(x.strip() for x in appended)
    return re.sub(r"\n{3,}", "\n\n", out_md).strip()
