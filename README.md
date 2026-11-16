# toaster-ocr

Fast, CPU-only document OCR for low-power hardware (no GPU, no large VLM).

Indra is already fast on CPU (~1.9 s/page on 2 ARM64 cores) and handles prose, reading order, and LaTeX math (`math=True`) well, **but** table structure was a major weak spot. `toaster-ocr` adds a lightweight `PP-DocLayoutV3 + SLANet-1M` table pipeline behind a simple page-level switch (`1-switch`), giving three clean speed/quality options:

1. **`indra` (`--mode indra`):** Native Indra (`math=True`) only. Fastest (~1.9 s/page), no extra layout/table models.
2. **`1-switch` (`--mode indra-pp`):** Run Indra first, and only run `PP-DocLayoutV3` + `SLANet-1M` on pages where Indra detects a table. Prose/math pages stay ~1.9 s/page; table pages get full HTML structure repair (+18.7 TEDS overall).
3. **`Always-on tables` (`--mode indra-pp --always-layout`):** Always run `PP-DocLayoutV3` + `SLANet-1M` on every page regardless, catching rare tables that Indra missed (+20.3 TEDS overall).

![Table TEDS vs CPU latency across the 3 options on NEW100](benchmarks/pareto.png)

---

## Benchmarks (2 ARM64 CPU cores, no GPU)

Measured on CPUs `0,1` (`taskset -c 0,1 nice -n 10`, 2 intra-op threads, serial execution, `math=True` across all options). Both benchmarks are completely disjoint held-out sets. Raw metrics JSON: [`benchmarks/summary.json`](benchmarks/summary.json).

### Metric definitions
- **Table TEDS (Tree Edit Distance-based Similarity, higher is better):** Standard HTML table structure + cell content score (`0` to `1`, where `1.000` is an exact HTML tree and cell match).
- **Text ED (Edit Distance, lower is better):** Normalized character edit distance on matched text blocks (`0.0000` is a perfect match).
- **Order ED (Reading-order Edit Distance, lower is better):** Normalized edit distance on block reading order.
- **Pass rate (higher is better):** Percentage of deterministic `olmOCR-bench` unit tests passed.

### 1. Held-out OmniDocBench (`NEW100`: 100 unseen pages, 60% table / 40% control, 50 EN / 50 ZH)

| Option | Layout run on | Mean / page (median) | Table TEDS (higher is better) | TEDS struct-only (higher is better) | Text ED page avg (lower is better) | Order ED (lower is better) |
|---|---:|---:|---:|---:|---:|---:|
| **`indra`** (`math=True`) | `0%` | **1.91 s** (`1.63 s`) | `0.4308` | `0.6023` | `0.1329` | `0.2978` |
| **`1-switch`** (`indra-pp`, tables only) | **72%** | **5.94 s** (`6.24 s`) | **`0.6177`** | **`0.7736`** | **`0.1317`** | **`0.2722`** |
| **`Always-on tables`** (`--always-layout`) | `100%` | **7.00 s** (`6.43 s`) | **`0.6333`** | **`0.7892`** | **`0.1317`** | **`0.2722`** |

*(Note: `NEW100` was deliberately oversampled to 60% table pages, so 72/100 pages trigger layout. On the 28 skipped pages, `PP-DocLayoutV3` caught 2 edge-case tables: a tiny 2x2 exam stamp box and a formula-heavy uncertainty table. That explains `0.6177` vs `0.6333` TEDS, with zero change to text or reading order.)*

### 2. olmOCR-bench Stratified Pilot (`OLM24`: 24 pages, 188 official unit tests)

| Option | Layout run on | Mean / page (median) | Overall pass rate (higher is better) | `long_tiny_text` | `multi_column` | `table_tests` | `headers_footers` | `arxiv_math` | `old_scans` | `old_scans_math` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`indra`** (`math=True`) | `0%` | **2.25 s** (`1.70 s`) | `34.57%` (`65/188`) | `88.9%` | `84.6%` | `72.0%` | `31.3%` | `20.8%` | `4.2%` | `1.7%` |
| **`1-switch`** (`indra-pp`) | **37.5%** | **4.12 s** (**`3.31 s`**) | **`35.64%`** (`67/188`) | `88.9%` | **`92.3%`** | **`76.0%`** | `31.3%` | `20.8%` | `4.2%` | `1.7%` |
| **`Always-on tables`** (`--always-layout`) | `100%` | **6.68 s** (`5.95 s`) | **`35.64%`** (`67/188`) | `88.9%` | **`92.3%`** | **`76.0%`** | `31.3%` | `20.8%` | `4.2%` | `1.7%` |

- On `OLM24` (where 62.5% of pages have no tables), **`1-switch`** skips `PPv3 + SLANet` on `15/24` pages with **0 missed tables (`67/188 == 67/188`)**, cutting mean latency by **2.56 s/page** (`6.68 s -> 4.12 s`, median **`3.31 s`**).
- *Per-stage timing across all 124 pages:* `PP-DocLayoutV3` ~3.81 s/page mean, `SLANet-1M` ~1.14 s/page mean (~0.97 s median on table pages).

### Optional Chinese / CJK text switch (`--mode pp-v6`)
We also tested routing low-confidence Indra lines (`conf <= 0.96`) to `PP-OCRv6_small_rec_onnx` (21.2 MB ONNX). On **English**, it added ~3 to 4 s/page for negligible gain (`0.1349 -> 0.1332` ED on `NEW100`; `66/188` vs `67/188` on `OLM24`), so it is **not** needed for English. On **Simplified Chinese** (`NEW100`, `n=48`), it cut Text ED from `0.1287` to `0.1098` (`25 wins / 1 loss`), so `--mode pp-v6` remains available if you process Chinese documents.

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
toaster-ocr download indra ppv3 slanet
# Optional if you also want --mode pp-v6 for Chinese documents:
# toaster-ocr download ppocrv6
```

### 3. Run OCR on an image
```bash
# Fast native Indra only (~1.9s/page)
toaster-ocr run page.png --mode indra

# 1-switch (recommended): runs PP-DocLayoutV3 + SLANet only when Indra spots a table
toaster-ocr run page.png --mode indra-pp

# Always-on tables: run PP-DocLayoutV3 + SLANet on every page
toaster-ocr run page.png --mode indra-pp --always-layout
```

### 4. Start OpenAI-compatible HTTP server
Exposes `indra-ocr-local`, `indra-pp-local`, and `pp-v6-local` on `/v1/models`, `/v1/chat/completions` (base64 `image_url`), and `/v1/ocr`:
```bash
TOASTER_OCR_API_KEY="your-local-key" toaster-ocr serve --host 127.0.0.1 --port 8080
```

---

## Honest Limits & Next Focus

- **Where this CPU stack struggles (next thing to explore):** On 19th-century cursive handwriting (`old_scans` `1/24` and `old_scans_math` `1/59`) and heavy arXiv equations (`arxiv_math` `5/24`), small printed-line OCR + `MathReader` hit a hard ceiling. Finding better compact math/hard-text components for those failure cases is the next step.
- **Scoring notes:** OmniDocBench `NEW100` omits CDM; `OLM24` runs unmodified `allenai/olmocr` `tests.py` for `present`/`absent`/`order`/`table` and KaTeX 0.16.22 MathML symbol equivalence (`omni_parse_bench.equations`) for `math`.
- **Platform tested:** Benchmarks were measured on Linux ARM64 CPU; x86/macOS/Windows work in standard Python environments when dependencies are installed, but exact CPU timings will vary by hardware.

## Licensing

Newly authored code in this repository is licensed under the **Apache License 2.0** (`LICENSE`). Upstream models and datasets carry separate terms (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): notably `datalab-to/indra-ocr` model weights use modified **AI Pubs OpenRAIL-M** terms and OmniDocBench is research-only).
