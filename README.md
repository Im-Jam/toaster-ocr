# toaster-ocr

**SOTA OCR efficiency on toaster specs (CPU-only, no GPU, no giant VLM).**

`toaster-ocr` composes small specialist document-OCR models on CPU using a **two-stage cascade**: run fast native Indra first (~1.9 s/page), and only escalate the lines and pages that actually need extra help:

1. **Text recognition switch (`conf <= 0.96`):** Indra already outputs per-line CTC confidence for free. Easy lines (`conf > 0.96`) keep Indra; uncertain non-math/non-RTL lines and table cells route to `PP-OCRv6_small_rec_onnx` (21.2 MB ONNX).
2. **Table layout switch (`has_indra_table`):** Prose/math pages with zero Indra-detected tables skip `PP-DocLayoutV3` (~3.81 s/page) + `SLANet-1M` (~1.14 s/page). Pages with tables run `PP-DocLayoutV3` table regrouping + `SLANet-1M` structure repair (**+19–20 TEDS**). Pass `--always-layout` if you want `PP-DocLayoutV3` on every page regardless.

---

## Why a simple threshold instead of a learned adapter?

`PP-OCRv6_small_rec_onnx` (5.29M params) fixes real English/Chinese misreads compared to Indra's built-in recognizer, but running it on every line adds ~12–15 s/page on dense documents on CPU. On a 100-page development set (`DEV100`, 8,114 lines), we compared an 8-feature page-grouped out-of-fold logistic classifier against pure `conf <= 0.96`: pure `conf <= 0.96` beat the 8-feature classifier (`0.1511` vs `0.1513` page-avg edit distance) with zero extra model parameters.

Both benchmarks below are **completely disjoint held-out evaluations** (`0%` page overlap with `DEV100`).

---

## Benchmarks (2 ARM64 CPU cores, no GPU)

Measured on CPUs `0,1` (`taskset -c 0,1 nice -n 10`, `2` intra-op threads, serial execution, `math=True` across all arms). Raw numbers: [`benchmarks/summary.json`](benchmarks/summary.json).

### 1. Held-out OmniDocBench (`NEW100` — 100 unseen pages: 60% table / 40% control, 50 EN / 50 ZH)

| Pipeline | Escalated lines | Layout run on | Mean / page (median) | Text ED page avg ↓ | Text ED whole ↓ | Order ED ↓ | Table TEDS ↑ | Formula norm ED ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`indra`** (`math=True`) | `0.0%` | `0%` | **1.91 s** (`1.63 s`) | `0.1329` | `0.1700` | `0.2978` | `0.4308` | `0.7498` |
| **`indra-pp`** (`+ PPv3 + SLANet`) | `0.0%` | `100%` | **7.00 s** (`6.43 s`) | `0.1317` | `0.1687` | `0.2722` | `0.6333` | `0.7469` |
| **`pp-v6` (2-switch default)** | **17.79%** | **72%** | **8.84 s** (`7.94 s`) | **`0.1209`** | **`0.1645`** | **`0.2716`** | **`0.6211`** | **`0.7311`** |
| **`pp-v6 --always-layout`** | **17.79%** | `100%` | **9.91 s** (`8.38 s`) | **`0.1209`** | **`0.1645`** | **`0.2716`** | **`0.6367`** | **`0.7311`** |
| **`pp-v6-full`** (`100%` `PP-OCRv6`) | `99.97%` | `100%` | **22.39 s** (`19.37 s`) | `0.1198` | `0.1644` | `0.2726` | `0.6369` | `0.7311` |

*(Note: `NEW100` was deliberately oversampled to 60% table pages, so 72/100 pages still trigger `PPv3+SLANet`. On those 28 skipped pages, `PP-DocLayoutV3` caught 2 edge-case tables — a tiny 2×2 exam stamp box and a formula-heavy uncertainty table — explaining the `0.6211` vs `0.6367` TEDS difference while keeping `0.0000` loss on text, order, and math.)*

### 2. olmOCR-bench Stratified Pilot (`OLM24` — 24 pages, 188 official unit tests)

| Pipeline | Escalated | Layout run on | Mean / page (median) | Overall Pass | `long_tiny_text` | `multi_column` | `table_tests` | `headers_footers` | `arxiv_math` | `old_scans` | `old_scans_math` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`indra`** | `0%` | `0%` | `2.25 s` (`1.70 s`) | `34.57%` (`65/188`) | `88.9%` | `84.6%` | `72.0%` | `31.3%` | `20.8%` | `4.2%` | `1.7%` |
| **`indra-pp`** | `0%` | `100%` | `6.68 s` (`5.95 s`) | **`35.64%`** (`67/188`) | `88.9%` | **`92.3%`** | **`76.0%`** | `31.3%` | `20.8%` | `4.2%` | `1.7%` |
| **`pp-v6` (2-switch default)** | **31.67%** | **37.5%** | **8.12 s** (**`5.69 s`**) | **`35.11%`** (`66/188`) | **`92.6%`** | **`92.3%`** | **`76.0%`** | `25.0%` | `20.8%` | `0.0%` | `1.7%` |
| **`pp-v6 --always-layout`** | **31.67%** | `100%` | `10.67 s` (`7.77 s`) | **`35.11%`** (`66/188`) | **`92.6%`** | **`92.3%`** | **`76.0%`** | `25.0%` | `20.8%` | `0.0%` | `1.7%` |
| **`pp-v6-full`** (`100%`) | `100.0%` | `100%` | `20.23 s` (`16.40 s`) | **`35.11%`** (`66/188`) | `88.9%` | **`92.3%`** | **`76.0%`** | `31.3%` | `20.8%` | `0.0%` | `1.7%` |

- On `OLM24` (where 62.5% of pages are non-table prose/math/scans), the **2-switch default** skips `PPv3+SLANet` on `15/24` pages with **0 missed tables (`66/188 == 66/188`)**, dropping median latency from **`16.40 s` (`pp-v6-full`) → `5.69 s`** (**2.88× faster median**, even faster than `indra-pp`!).
- *Per-stage breakdown across all 124 pages:* `PP-DocLayoutV3` ~3.81 s/page mean, `SLANet` ~1.14 s/page mean (~0.97 s median on table pages), `PP-OCRv6` stage `2.70 s` vs `15.24 s` on `NEW100` (**5.65× faster**).

---

## Quickstart

### 1. Install
```bash
git clone https://github.com/Im-Jam/toaster-ocr.git
cd toaster-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```
*(If `paddlepaddle` / `torch` must live in separate venvs on your platform, install base `pip install -e .` and point `TOASTER_OCR_PPV3_PYTHON` / `TOASTER_OCR_SLANET_PYTHON` at those Python executables.)*

### 2. Download pinned upstream models
No model weights are bundled in this repo. Fetch pinned revisions into `~/.cache/toaster-ocr` (override with `TOASTER_OCR_HOME`):
```bash
toaster-ocr download
```

### 3. Run OCR on an image
```bash
# Default 2-switch mode (conf<=0.96 PP-OCRv6 + table-gated PPv3/SLANet)
toaster-ocr run page.png --mode pp-v6

# Also run PP-DocLayoutV3+SLANet on every page even if Indra sees no Table
toaster-ocr run page.png --mode pp-v6 --always-layout

# Fast native Indra only (~1.9s/page)
toaster-ocr run page.png --mode indra

# Force PP-OCRv6 on 100% of non-math lines
toaster-ocr run page.png --mode pp-v6-full --json
```

### 4. Start OpenAI-compatible HTTP server
Exposes `indra-ocr-local`, `indra-pp-local`, `pp-v6-local`, and `pp-v6-full-local` on `/v1/models`, `/v1/chat/completions` (base64 `image_url`), and `/v1/ocr`:
```bash
TOASTER_OCR_API_KEY="your-local-key" toaster-ocr serve --host 127.0.0.1 --port 8080
```

---

## Honest Limits

- **Not a VLM replacement for 19th-century cursive or heavy arXiv LaTeX:** `old_scans` (`0/24`) and `old_scans_math` (`1/59`) fail on **both** Indra and `pp-v6-full` (`100%`) because those pages are cursive handwriting outside printed-line CTC recognition.
- **High-confidence stylized text:** If Indra misreads a bold/stylized word with `conf > 0.96`, the confidence switch keeps Indra unless you raise `--ppv6-threshold 1.0`.
- **Scoring notes:** OmniDocBench `NEW100` omits CDM; `OLM24` runs unmodified `allenai/olmocr` `tests.py` for `present`/`absent`/`order`/`table` and KaTeX 0.16.22 MathML symbol equivalence (`omni_parse_bench.equations`) for `math`.
- **Platform tested:** Benchmarks were measured on Linux ARM64 CPU; x86/macOS/Windows work in standard Python environments when dependencies are installed, but exact CPU timings will vary by hardware.

## Licensing

Newly authored code in this repository is licensed under the **Apache License 2.0** (`LICENSE`). Upstream models and datasets carry separate terms (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — notably `datalab-to/indra-ocr` model weights use modified **AI Pubs OpenRAIL-M** terms and OmniDocBench is research-only).
