# toaster-ocr

**Fast, pure-ONNX, CPU-only document OCR** — zero GPU, zero PyTorch, zero PaddlePaddle (`~214 MB` total ONNX footprint).

`toaster-ocr` layers four targeted, pure-ONNX CPU upgrades on top of **[Indra OCR `1.1.2`](https://github.com/datalab-to/indra-ocr)** (`math=True`):

1. **5-Fold-CV Table Ensemble (`SLANet-plus` 7.4 MB + `DeepDoc TSR` 12.2 MB + 100%-conserving `<br>` matcher):** Cross-validated router picks between `SLANet-plus`, `InfiniFlow/deepdoc` `tsr.onnx`, and native Indra with a sparse-grid fallback — lifting **OPB Tables from `59.60%` to `73.32%` (`+13.72 pts`, `+739` net tests)**, beating `DeepSeek-OCR-2`, `LightOnOCR-3-0.8B`, and `LightOnOCR-3-4B` on OPB Tables.
2. **`PP-DocLayout-S` ONNX (4.7 MB @ `conf=0.28`, ~73 ms/page on 2 ARM64 cores) + KV-Cached `pix2text-mfr-1.5`:** Splits `pix2text-mfr-1.5`'s uncached ONNX decoder locally into `mfr_dec_init.onnx` (4.5 MB) + `mfr_dec_step.onnx` (25.8 MB) (`9.6×` faster on CPU, `100%` token parity) — lifting **`olmOCR-bench` `old_scans_math` from `3.49%` to `40.61%` (`+37.12 pts`)** and **`arxiv_math` from `55.93%` to `66.72%` (`+10.79 pts`)**.
3. **Page-Adaptive `lm.npz` Beam Search:** Only boosts Indra's 46 MB n-gram beam (`LM_GATE → 0.985`, `LM_BEAM → 16`) on pages whose median line confidence is `< 0.91`, fixing spacing/character slips on degraded pages with zero slowdown on clean pages.
4. **PDF `textlayer.py` PUA/Ink Fix:** Stops rejecting valid born-digital PDF text layers that contain Private-Use-Area checkboxes (`Co`) or sparse lines.

---

## Verified Benchmarks

### 1. Omni Parse Bench (Full 2,937 Pages / 16,288 Unit Tests, 0 Errors)

Evaluated with `omni-parse-bench==1.7.1` across all 22 suites (`1.21 s/page` median on 4 vCPUs; note Datalab is the maintainer of both `omni_parse_bench` and `indra-ocr`).

| System | Hardware | OPB Headline ↑ | Text (8,499) ↑ | Tables (5,386) ↑ | Layout (2,403) ↑ |
|---|---|---:|---:|---:|---:|
| Stock `Indra OCR` (`1.1.2`) | CPU (ONNX) | `71.37%` | `62.55%` | `59.60%` | **`91.97%`** |
| **`toaster-ocr` (`v0.2`)** | **CPU (ONNX)** | **`77.97%`** (**`+6.60 pts`**) | **`69.49%`** (**`+6.94 pts`**) | **`73.32%`** (**`+13.72 pts`**) | `91.09%` (`-0.87`) |
| *Context: `DeepSeek-OCR-2`* | *GPU VLM* | *`75.43%`* | *`71.6%`* | *`68.1%`* | *`86.6%`* |
| *Context: `LightOnOCR-3-0.8B`* | *GPU VLM* | *`81.9%`* | *`81.7%`* | *`72.2%`* | *`91.7%`* |

- **Net test delta vs Stock Indra:** **`+1,308` net passing tests** (`1,707` wins / `399` losses; `15/22` suites improved, `3` tied, max single-suite regression `-3`).
- **Top suite gains:** `tables_derived` `64.0% → 77.4%` (`+728`), `olmo_arxiv_math` `66.7% → 75.2%` (`+280`), `olmo_old_scans_math` `13.8% → 49.6%` (`+161`), `synth_copies` `70.2% → 74.4%` (`+80`), `olmo_table_tests` `69.9% → 74.2%` (`+31`).

### 2. Independent Verification: AllenAI `olmOCR-bench` (1,403 PDFs / 7,019 Tests)

Scored with official `olmocr==0.4.27` Playwright/KaTeX harness. Both runs used `furniture=True` (headers/footers kept, so `headers_footers` is apples-to-apples).

| Suite | Stock `Indra OCR` | **`toaster-ocr` (`v0.2`)** | Delta |
|---|---:|---:|---:|
| `old_scans_math` (458) | `3.49%` | **`40.61%`** | **`+37.12 pts`** (`+170`) |
| `arxiv_math` (2,927) | `55.93%` | **`66.72%`** | **`+10.79 pts`** (`+316`) |
| `table_tests` (1,022) | `64.29%` | **`68.69%`** | **`+4.40 pts`** (`+45`) |
| `multi_column` (884) | `72.62%` | **`73.08%`** | **`+0.46 pts`** (`+4`) |
| `old_scans` (526) | `22.81%` | **`23.00%`** | `+0.19 pts` (`+1`) |
| `headers_footers` (760) | `26.97%` | **`27.11%`** | `+0.14 pts` (`+1`) |
| `long_tiny_text` (442) | **`76.70%`** | `76.02%` | `-0.68 pts` (`-3`) |
| **Overall (macro avg)** | **`46.12%`** | **`53.60%`** | **`+7.48 pts`** |
| **Micro total (7,019)** | `51.52%` (`3,616`) | **`59.13%` (`4,150`)** | **`+7.61 pts` (`+534` tests)** |

### Where it excels vs. where CPU limits remain

- **Strong on:** clean/scanned printed PDFs, multi-column layouts, complex tables (wired & borderless), born-digital PDFs, and display math (`LaTeX`).
- **Still bounded on:** 19th-century cursive handwriting (`~18%` OPB), heavily degraded historical prose scans (`~31%` OPB), and non-Latin/multilingual scripts outside Indra's training distribution — those tasks need a GPU VLM (e.g., `LightOnOCR-3-0.8B` or `Qwen3-VL`).

---

## Quickstart

```bash
pip install git+https://github.com/Im-Jam/toaster-ocr.git

# 1. Fetch pinned ONNX models (~214 MB total) and build KV-cached MFR decoders locally:
toaster-ocr download

# 2. OCR an image or PDF to Markdown:
toaster-ocr ocr document.pdf -o document.md --threads 2

# 3. Or launch the OpenAI-compatible HTTP server (zero extra web dependencies):
toaster-ocr serve --host 127.0.0.1 --port 8129 --threads 2
```

### Python API

```python
from toaster_ocr import ToasterOCR

engine = ToasterOCR(threads=2)
res = engine.read("page.png")      # or .pdf
print(res.markdown)
print(res.elapsed_seconds, res.stats)
```

### OpenAI-Compatible Server Endpoints

- `GET /health`
- `POST /v1/ocr` (`{"image": "data:image/png;base64,..."}`)
- `POST /v1/chat/completions` (standard `image_url` or PDF `file_data` base64 URI)
- `POST /v1/completions` (`{"prompt": "data:application/pdf;base64,..."}`)

Set `TOASTER_OCR_API_KEY` to enforce `Authorization: Bearer <key>`.

---

## Licensing & Provenance

The integration code in this repository is **Apache-2.0** (`LICENSE`). Model weights are downloaded on demand from Hugging Face under their own upstream licenses — notably `datalab-to/indra-ocr` weights are **Modified AI Pubs OpenRAIL-M**, while `SLANet_plus_onnx`, `DeepDoc TSR`, and `PP-DocLayout-S-ONNX` are **Apache-2.0**, and `pix2text-mfr-1.5` is **MIT**. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
