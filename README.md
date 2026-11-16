# toaster-ocr

Fast, CPU-only document OCR for low-power hardware (no GPU, no large VLM).

`toaster-ocr` chains small specialist models together with two simple runtime switches so you can pick the speed/accuracy balance you want:

1. **`1-switch` (`--mode indra-pp`, fast English & general documents):** Run Indra (`math=True`) first (~1.9 s/page on 2 ARM64 CPU cores), and only run `PP-DocLayoutV3` + `SLANet-1M` on pages where Indra detects a table. Prose/math pages stay fast (~1.9 s), while table pages get full HTML structure repair (+19 TEDS).
2. **`2-switch` (`--mode pp-v6`, Chinese & dense mixed documents):** Same table layout switch **plus** a line-confidence switch (`conf <= 0.96`). Easy lines keep Indra; uncertain non-math/non-RTL lines and table cells get re-read by `PP-OCRv6_small_rec_onnx` (21.2 MB ONNX), cutting Chinese text edit distance by ~1.9 percentage points without paying brute-force cost on every line.

Pass `--always-layout` on either mode if you want `PP-DocLayoutV3` on every page regardless.

![Text quality and table TEDS vs CPU latency on NEW100](benchmarks/pareto.png)

---

## Why a confidence threshold instead of a learned adapter?

`PP-OCRv6_small_rec_onnx` (5.29M params) fixes real English and Chinese misreads compared to Indra's built-in recognizer, but running it on every line adds ~12 to 15 s/page on dense documents on CPU. On a 100-page development set (`DEV100`, 8,114 lines), we compared an 8-feature page-grouped out-of-fold logistic classifier against pure `conf <= 0.96`: pure `conf <= 0.96` beat the 8-feature classifier (`0.1511` vs `0.1513` page-average edit distance) with zero extra model parameters.

Both benchmarks below are **completely disjoint held-out evaluations** (0% page overlap with `DEV100`).

---

## Benchmarks (2 ARM64 CPU cores, no GPU)

Measured on CPUs `0,1` (`taskset -c 0,1 nice -n 10`, 2 intra-op threads, serial execution, `math=True` across all configurations). Raw metrics JSON: [`benchmarks/summary.json`](benchmarks/summary.json).

### Metric definitions
- **Text ED (Edit Distance, lower is better):** Normalized character edit distance on matched text blocks against ground truth (`0.0000` is a perfect match). In the chart above, `Text Score = 1 - Text ED` (higher is better).
- **Order ED (Reading-order Edit Distance, lower is better):** Normalized edit distance on block reading order.
- **Table TEDS (Tree Edit Distance-based Similarity, higher is better):** Standard HTML table structure + cell content score (`0` to `1`, where `1.000` is an exact HTML tree and cell match).
- **Formula ED (Display formula Edit Distance, lower is better):** Normalized edit distance on matched LaTeX display equations (`n = 67` matched formulas).

### 1. Held-out OmniDocBench (`NEW100`: 100 unseen pages, 60% table / 40% control, 50 EN / 50 ZH)

| Configuration | Lines sent to `PP-OCRv6` | Layout run on | Mean / page (median) | Text ED page avg (lower is better) | Text ED whole (lower is better) | Order ED (lower is better) | Table TEDS (higher is better) | Formula ED (lower is better) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`indra`** (`math=True`) | `0.0%` | `0%` | **1.91 s** (`1.63 s`) | `0.1329` | `0.1700` | `0.2978` | `0.4308` | `0.7498` |
| **`indra + v6 switch`** (no layout/SLANet) | `17.79%` | `0%` | **4.82 s** (`3.76 s`) | `0.1221` | `0.1659` | `0.2975` | `0.4339` | `0.7311` |
| **`1-switch`** (`indra-pp`, tables only) | `0.0%` | **72%** | **5.94 s** (`6.24 s`) | `0.1317` | `0.1687` | `0.2722` | **`0.6177`** | `0.7469` |
| **`indra-pp --always-layout`** | `0.0%` | `100%` | **7.00 s** (`6.43 s`) | `0.1317` | `0.1687` | `0.2722` | `0.6333` | `0.7469` |
| **`2-switch`** (`pp-v6`, `conf<=0.96` + table gate) | **17.79%** | **72%** | **8.84 s** (`7.94 s`) | **`0.1209`** | **`0.1645`** | **`0.2716`** | **`0.6211`** | **`0.7311`** |
| **`v6 switch + always layout`** (`pp-v6 --always-layout`) | **17.79%** | `100%` | **9.91 s** (`8.38 s`) | **`0.1209`** | **`0.1645`** | **`0.2716`** | **`0.6367`** | **`0.7311`** |
| **`indra + v6 full`** (no layout/SLANet) | `99.97%` | `0%` | **17.23 s** (`14.45 s`) | `0.1211` | `0.1659` | `0.2980` | `0.4334` | `0.7311` |
| **`pp-v6-full`** (`100%` `PP-OCRv6` + layout) | `99.97%` | `100%` | **22.39 s** (`19.37 s`) | `0.1198` | `0.1644` | `0.2726` | `0.6369` | `0.7311` |

#### English vs. Simplified Chinese breakdown (`NEW100` Text ED, lower is better)
- **English (`n = 43`):** `1-switch` (`0.1349`) vs `2-switch` (`0.1332`), only `-0.0017` difference. For English-only documents, `1-switch` is ~2.9 s/page faster on `NEW100` (~4.0 s/page faster on `OLM24`) with nearly identical text quality.
- **Simplified Chinese (`n = 48`):** `1-switch` (`0.1287`) vs `2-switch` (`0.1098`), `-0.0189` (1.89 percentage points better). That is where `PP-OCRv6` earns its time.

*(Note: `NEW100` was deliberately oversampled to 60% table pages, so 72/100 pages still trigger `PPv3 + SLANet`. On the 28 skipped pages, `PP-DocLayoutV3` caught 2 edge-case tables: a tiny 2x2 exam stamp box and a formula-heavy uncertainty table. Skipping them explains `0.6177` vs `0.6333` / `0.6211` vs `0.6367` TEDS with `0.0000` change in text, order, or math.)*

### 2. olmOCR-bench Stratified Pilot (`OLM24`: 24 pages, 188 official unit tests)

**Pass rate (higher is better):** percentage of deterministic unit tests passed across all 7 categories (`long_tiny_text`, `multi_column`, `table_tests`, `headers_footers`, `arxiv_math`, `old_scans`, `old_scans_math`).

| Configuration | Escalated lines | Layout run on | Mean / page (median) | Overall pass rate (higher is better) | `long_tiny_text` | `multi_column` | `table_tests` | `headers_footers` | `arxiv_math` | `old_scans` | `old_scans_math` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`indra`** | `0%` | `0%` | `2.25 s` (`1.70 s`) | `34.57%` (`65/188`) | `88.9%` | `84.6%` | `72.0%` | `31.3%` | `20.8%` | `4.2%` | `1.7%` |
| **`1-switch`** (`indra-pp`, tables only) | `0%` | **37.5%** | **4.12 s** (**`3.31 s`**) | **`35.64%`** (`67/188`) | `88.9%` | **`92.3%`** | **`76.0%`** | `31.3%` | `20.8%` | `4.2%` | `1.7%` |
| **`indra-pp --always-layout`** | `0%` | `100%` | `6.68 s` (`5.95 s`) | **`35.64%`** (`67/188`) | `88.9%` | **`92.3%`** | **`76.0%`** | `31.3%` | `20.8%` | `4.2%` | `1.7%` |
| **`2-switch`** (`pp-v6`) | **31.67%** | **37.5%** | `8.12 s` (`5.69 s`) | **`35.11%`** (`66/188`) | **`92.6%`** | **`92.3%`** | **`76.0%`** | `25.0%` | `20.8%` | `0.0%` | `1.7%` |
| **`v6 switch + always layout`** | **31.67%** | `100%` | `10.67 s` (`7.77 s`) | **`35.11%`** (`66/188`) | **`92.6%`** | **`92.3%`** | **`76.0%`** | `25.0%` | `20.8%` | `0.0%` | `1.7%` |
| **`pp-v6-full`** (`100%`) | `100.0%` | `100%` | `20.23 s` (`16.40 s`) | **`35.11%`** (`66/188`) | `88.9%` | **`92.3%`** | **`76.0%`** | `31.3%` | `20.8%` | `0.0%` | `1.7%` |

- On `OLM24` (where 62.5% of pages are non-table prose, math, or scans), **`1-switch`** reaches the highest score (**`35.64%`**, 0 missed tables) at **`4.12 s` mean (`3.31 s` median)**: **2.56 s/page faster** than always running layout, and **4.9x faster** than `pp-v6-full` (`20.23 s`).
- *Per-stage breakdown across all 124 pages:* `PP-DocLayoutV3` ~3.81 s/page mean, `SLANet` ~1.14 s/page mean (~0.97 s median on table pages), `PP-OCRv6` stage `2.70 s` vs `15.24 s` on `NEW100` (**5.65x faster**).

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
# 1-switch mode: fast English/general OCR (table-gated PPv3 + SLANet, no PP-OCRv6)
toaster-ocr run page.png --mode indra-pp

# 2-switch mode: add conf<=0.96 PP-OCRv6 escalation (recommended for Chinese/mixed docs)
toaster-ocr run page.png --mode pp-v6

# Also run PP-DocLayoutV3 + SLANet on every page even if Indra sees no Table
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
- **High-confidence stylized text:** If Indra misreads a bold or stylized word with `conf > 0.96`, the confidence switch keeps Indra unless you raise `--ppv6-threshold 1.0`.
- **Scoring notes:** OmniDocBench `NEW100` omits CDM; `OLM24` runs unmodified `allenai/olmocr` `tests.py` for `present`/`absent`/`order`/`table` and KaTeX 0.16.22 MathML symbol equivalence (`omni_parse_bench.equations`) for `math`.
- **Platform tested:** Benchmarks were measured on Linux ARM64 CPU; x86/macOS/Windows work in standard Python environments when dependencies are installed, but exact CPU timings will vary by hardware.

## Licensing

Newly authored code in this repository is licensed under the **Apache License 2.0** (`LICENSE`). Upstream models and datasets carry separate terms (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): notably `datalab-to/indra-ocr` model weights use modified **AI Pubs OpenRAIL-M** terms and OmniDocBench is research-only).
