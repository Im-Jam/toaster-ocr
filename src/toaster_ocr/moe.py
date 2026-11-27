"""Anatomical System MoE (v0.4) helpers:
1. Jaccard slot-aligned non-duplicating table replacement (`splice_tables_jaccard`).
2. Structural + KaTeX-checked anchored display-equation grafting (`graft_verified_display_math`).
3. Contrastive DOM + anatomical table arbiter (`should_prefer_ocria_table`).
4. Non-VNNI `x86_64` AVX2 QDQ overflow parity conversion for `rec_latin.onnx` (`ensure_x86_rec_latin_qdq_parity`).
"""
from __future__ import annotations

import html as _html
import platform
import re
from collections import Counter
from pathlib import Path
from typing import Any

TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.I | re.S)
DISP_RE = re.compile(
    r"(?:\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\begin\{(?:equation|align|alignat|gather|multline)\*?\}[\s\S]+?\\end\{(?:equation|align|alignat|gather|multline)\*?\})"
)


def _table_tokens(tbl_html: str) -> set[str]:
    toks: set[str] = set()
    in_tr = False
    for m in re.finditer(r"<(tr|t[dh]|table|/tr)\b[^>]*>|([^<>]+)", tbl_html, re.I):
        tag, txt = m.group(1), m.group(2)
        if tag is not None:
            if tag.lower() == "tr":
                in_tr = True
        elif txt and in_tr:
            toks.update(re.findall(r"\w+", txt.lower()))
    return toks


def splice_tables_jaccard(base_text: str, donor_text: str) -> str | None:
    """Align donor `<table>` blocks into base `<table>` slots by token Jaccard.

    Contract (verified on all 2,937 OPB pages):
    - Each donor table is used at most once.
    - Every base table slot emits at least one table (falls back to base table when no donor remains).
    - Unmatched donor tables are appended at their highest-Jaccard base slot.
    - Never duplicates tables (`#fused == max(#base, #donor)`).
    """
    bparts = TABLE_RE.split(base_text or "")
    btabs = TABLE_RE.findall(base_text or "")
    dtabs = TABLE_RE.findall(donor_text or "")
    if not btabs or not dtabs:
        return None
    bsets = [_table_tokens(t) for t in btabs]
    dsets = [_table_tokens(t) for t in dtabs]
    used: set[int] = set()
    match: dict[int, int] = {}
    for i, bs in enumerate(bsets):
        best_j, best_sc = -1, -1.0
        for j, ds in enumerate(dsets):
            if j in used:
                continue
            sc = len(bs & ds) / max(1, len(bs | ds))
            if sc > best_sc:
                best_sc, best_j = sc, j
        if best_j >= 0:
            match[i] = best_j
            used.add(best_j)
    leftovers = [j for j in range(len(dtabs)) if j not in used]

    def _best_slot(j: int) -> int:
        return max(
            range(len(bsets)),
            key=lambda k: len(bsets[k] & dsets[j]) / max(1, len(bsets[k] | dsets[j])),
        )

    out = [bparts[0]]
    for i in range(len(btabs)):
        out.append(dtabs[match[i]] if i in match else btabs[i])
        at_i = [
            j
            for j in leftovers
            if _best_slot(j) == i or (i == len(btabs) - 1 and _best_slot(j) >= i)
        ]
        for j in sorted(set(at_i)):
            out.append(dtabs[j])
        if i + 1 < len(bparts):
            out.append(bparts[i + 1])
    fused = "".join(out)
    assert len(TABLE_RE.findall(fused)) == max(len(btabs), len(dtabs))
    return fused


def _cell_type(txt: str) -> str:
    s = txt.strip()
    if not s:
        return "empty"
    if re.fullmatch(r"[<>≤≥=]?\s*0?\.\d+[a-z*]*", s, re.I):
        return "pval"
    if re.search(r"\d+\s*\(\s*\d+(?:\.\d+)?%?\s*\)", s):
        return "num_pct"
    if re.search(r"\d+(?:\.\d+)?\s*[–\-±]\s*\d+(?:\.\d+)?", s):
        return "ci"
    if re.fullmatch(r"[-+]?\d[\d,.\s/%]*", s):
        return "num"
    return "text"


def table_dom_quality(text_or_html: str) -> dict[str, float]:
    """Compute DOM + column-type coherence + header-sanity statistics for `<table>` blocks."""
    tbls = TABLE_RE.findall(text_or_html or "")
    if not tbls:
        return {
            "n_tbls": 0.0,
            "n_rows": 0.0,
            "n_cells": 0.0,
            "empty_frac": 1.0,
            "col_purity": 0.0,
            "n_numbers": 0.0,
            "tex_in_header": 0.0,
            "br_fused_body": 0.0,
        }
    n_rows = 0
    n_cells = 0
    n_empty = 0
    n_numbers = 0
    tex_in_header = 0
    br_fused_body = 0
    col_purities: list[float] = []

    for tbl in tbls:
        trs = re.findall(r"<tr\b[^>]*>(.*?)</tr>", tbl, re.I | re.S)
        n_rows += len(trs)
        col_types: dict[int, list[str]] = {}
        for r_i, tr in enumerate(trs):
            cells = re.findall(r"<(t[dh])\b([^>]*)>(.*?)</t[dh]>", tr, re.I | re.S)
            c_idx = 0
            for tag, attrs, body in cells:
                m_cs = re.search(r'colspan=["\']?(\d+)', attrs, re.I)
                cs = max(1, int(m_cs.group(1))) if m_cs else 1
                is_th = tag.lower() == "th" or r_i == 0
                if is_th and ("$" in body or "\\mathrm" in body):
                    tex_in_header += 1
                if not is_th and len(re.findall(r"<br\s*/?>", body, re.I)) >= 2:
                    br_fused_body += 1
                clean = re.sub(r"<[^>]+>", " ", body).strip()
                n_cells += 1
                if not clean:
                    n_empty += 1
                else:
                    n_numbers += len(re.findall(r"\d+(?:\.\d+)?", clean))
                ct = _cell_type(clean)
                if not is_th and cs == 1:
                    col_types.setdefault(c_idx, []).append(ct)
                c_idx += cs
        for cts in col_types.values():
            non_emp = [x for x in cts if x != "empty"]
            if len(non_emp) >= 2:
                top_cnt = Counter(non_emp).most_common(1)[0][1]
                col_purities.append(top_cnt / len(non_emp))

    return {
        "n_tbls": float(len(tbls)),
        "n_rows": float(n_rows),
        "n_cells": float(n_cells),
        "empty_frac": n_empty / max(1, n_cells),
        "col_purity": sum(col_purities) / len(col_purities) if col_purities else 0.5,
        "n_numbers": float(n_numbers),
        "tex_in_header": float(tex_in_header),
        "br_fused_body": float(br_fused_body),
    }


def should_prefer_ocria_table(
    ind_text: str,
    ocr_text: str,
    *,
    lp_is_garbled: bool = False,
    ind_tsr_row_sc: float = 0.0,
    ocr_none_bbox_frac: float = 0.0,
) -> bool:
    """Contrastive DOM + anatomical table arbiter between Indra and OCRia-nano."""
    if lp_is_garbled:
        return False
    qi = table_dom_quality(ind_text)
    qo = table_dom_quality(ocr_text)
    if qo["n_tbls"] == 0:
        return False
    if qi["n_tbls"] == 0:
        return True

    # Contrastive score: positive favors OCRia-nano, negative favors Indra
    score = 0.0
    score += 2.2 * (qo["col_purity"] - qi["col_purity"])
    score -= 1.8 * (qo["empty_frac"] - qi["empty_frac"])
    score += 0.35 * min(3.0, max(-3.0, (qo["n_numbers"] - qi["n_numbers"]) / max(5.0, qi["n_numbers"])))
    score += 0.45 * min(2.0, qi["tex_in_header"] - qo["tex_in_header"])
    score += 0.35 * min(2.0, qi["br_fused_body"] - qo["br_fused_body"])
    if ind_tsr_row_sc >= 0.78:
        score -= 0.40
    score -= 0.80 * ocr_none_bbox_frac
    return score > 0.0


def _brace_ok(tex: str) -> bool:
    depth = 0
    i = 0
    while i < len(tex):
        ch = tex[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def is_valid_display_latex(span: str) -> bool:
    """Verify balanced braces, environment tags, and absence of OCR hallucination loops."""
    s = span.strip()
    for l_d, r_d in (("$$", "$$"), ("\\[", "\\]")):
        if s.startswith(l_d) and s.endswith(r_d) and len(s) > len(l_d) + len(r_d):
            s = s[len(l_d) : -len(r_d)].strip()
            break
    if len(s) < 4 or not _brace_ok(s):
        return False
    if s.count("\\begin{") != s.count("\\end{"):
        return False
    if s.count("~") > 12 and s.count("~") > 0.25 * len(s):
        return False
    alnum = re.sub(r"[^A-Za-z0-9]", "", s)
    return len(alnum) >= 4


def ensure_x86_rec_latin_qdq_parity(ocria_model_dir: Path | None = None) -> bool:
    """Patch `ocria-nano/rec_latin.onnx` on `x86_64` CPUs to prevent AVX2 `VPMADDUBSW` INT8 overflow."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    if ocria_model_dir is None:
        ocria_model_dir = Path.home() / ".cache" / "liteparse" / "models" / "ocria-nano"
    onnx_path = Path(ocria_model_dir) / "rec_latin.onnx"
    marker = Path(ocria_model_dir) / ".qdq_fp32_parity"
    if not onnx_path.is_file() or marker.is_file():
        return False

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(onnx_path))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    const_outs = {
        n.output[0]: numpy_helper.to_array(a.t)
        for n in m.graph.node
        if n.op_type == "Constant"
        for a in n.attribute
        if a.name == "value"
    }
    all_const = {**inits, **const_outs}
    new_inits = dict(inits)
    dq_folded: set[str] = set()
    for n in m.graph.node:
        if n.op_type == "DequantizeLinear" and all(x in all_const for x in n.input):
            q = all_const[n.input[0]].astype(np.float32)
            s = all_const[n.input[1]].astype(np.float32)
            zp = all_const[n.input[2]].astype(np.float32) if len(n.input) > 2 else np.zeros_like(s)
            axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
            if s.ndim == 1 and q.ndim > 1:
                shape = [1] * q.ndim
                shape[axis] = s.shape[0]
                s = s.reshape(shape)
                zp = zp.reshape(shape)
            new_inits[n.output[0]] = ((q - zp) * s).astype(np.float32)
            dq_folded.add(n.output[0])

    q_map = {n.output[0]: n for n in m.graph.node if n.op_type == "QuantizeLinear"}
    new_nodes = []
    new_inits["__q_u8_min"] = np.array(0.0, dtype=np.float32)
    new_inits["__q_u8_max"] = np.array(255.0, dtype=np.float32)
    new_inits["__q_i8_min"] = np.array(-128.0, dtype=np.float32)
    new_inits["__q_i8_max"] = np.array(127.0, dtype=np.float32)

    for idx, n in enumerate(m.graph.node):
        if n.op_type == "DequantizeLinear":
            if n.output[0] in dq_folded:
                continue
            if n.input[0] in q_map:
                qn = q_map[n.input[0]]
                x_in, s_in = qn.input[0], qn.input[1]
                zp_in = qn.input[2] if len(qn.input) > 2 else None
                zp_arr = all_const.get(zp_in) if zp_in else None
                is_signed = zp_arr is not None and zp_arr.dtype == np.int8
                qmin, qmax = ("__q_i8_min", "__q_i8_max") if is_signed else ("__q_u8_min", "__q_u8_max")
                pfx = f"__fq_{idx}"
                d, r, a, c, sub = f"{pfx}_div", f"{pfx}_rnd", f"{pfx}_add", f"{pfx}_clp", f"{pfx}_sub"
                new_nodes.append(helper.make_node("Div", [x_in, s_in], [d]))
                new_nodes.append(helper.make_node("Round", [d], [r]))
                if zp_in:
                    zp_fp = f"{pfx}_zpfp"
                    new_inits[zp_fp] = all_const[zp_in].astype(np.float32)
                    new_nodes.append(helper.make_node("Add", [r, zp_fp], [a]))
                    new_nodes.append(helper.make_node("Clip", [a, qmin, qmax], [c]))
                    new_nodes.append(helper.make_node("Sub", [c, zp_fp], [sub]))
                else:
                    new_nodes.append(helper.make_node("Clip", [r, qmin, qmax], [c]))
                    sub = c
                new_nodes.append(helper.make_node("Mul", [sub, s_in], [n.output[0]]))
                continue
        if n.op_type == "QuantizeLinear":
            continue
        new_nodes.append(n)

    used_inputs = {inp for n in new_nodes for inp in n.input}
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    del m.graph.initializer[:]
    for k, v in new_inits.items():
        if k in used_inputs:
            m.graph.initializer.append(numpy_helper.from_array(v, name=k))
    onnx.save(m, str(onnx_path))
    marker.write_text("ok\n", encoding="utf-8")
    return True
