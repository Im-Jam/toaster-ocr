# toaster-ocr

**Fast, pure-ONNX, CPU-only document OCR (`~95 MB` total weights)** — zero GPU, zero PyTorch, zero PaddlePaddle.

`toaster-ocr` (`v0.3`) pairs **[LiteParse](https://github.com/run-llama/liteparse) (`OCRia-nano-1` + `LayoutNet`)** with our pure-ONNX display-math pipeline:

1. **`LiteParse` + `OCRia-nano-1` + `LayoutNet` (`55.8 MB`):** Native PDFium text-layer extraction (including inline `$math$`), `LayoutNet` grid/table reconstruction (`86.8%` on OPB `tables_derived`), and fast CPU OCR (`0.16–0.49s/page` on born-digital PDFs, `~1.0–1.4s/page` on scans).
2. **`PP-DocLayout-S` ONNX (`4.7 MB`, ~73 ms/page on 2 ARM64 cores) + Dynamic `QInt8` KV-Cached `pix2text-mfr-1.5` (`34.3 MB`):** Adds dedicated 2D display-equation recognition (which `OCRia-nano-1` lacks), skipping any formula boxes inside `LayoutNet` tables and using a conservative `conf=0.35` threshold on clean PDF text layers vs `0.28` on scans.

---

## Verified Benchmarks (`161` OPB Pages / `1,204` Official Unit Tests)

Evaluated locally with `omni-parse-bench==1.7.1` across six challenging OPB suites (`tables_derived`, `olmo_old_scans_math`, `olmo_arxiv_math`, `olmo_table_tests`, `olmo_old_scans`, `realdoc_pages`):

| Suite (`pages / tests`) | Stock `Indra 1.1.2` | `toaster-ocr v0.2` (`Indra` base) | Stock `OCRia-nano-1` | **`toaster-ocr v0.3`** |
|---|---:|---:|---:|---:|
| `tables_derived` (`30 / 325`) | `58.5%` | `67.7%` | **`86.8%`** | **`86.8%`** |
| `olmo_old_scans_math` (`36 / 450`) | `13.8%` | `49.6%` | `13.8%` | **`59.8%`** |
| `olmo_arxiv_math` (`25 / 129`) | `69.8%` | `78.3%` | `73.6%` | **`86.0%`** |
| `olmo_table_tests` (`30 / 155`) | `74.2%` | **`74.8%`** | `74.2%` | `74.2%` |
| `olmo_old_scans` (`20 / 65`) | **`43.1%`** | **`43.1%`** | `35.4%` | `35.4%` |
| `realdoc_pages` (`20 / 80`) | **`88.8%`** | `87.5%` | `~77.5%` | `77.5%` |
| **Total (`161 / 1,204`)** | `46.2%` (`556`) | `63.0%` (`758`) | `~53.1%` (`~639`) | **`71.6%` (`862`, `+8.6 pts` vs `v0.2`)** |

### 2-Core ARM64 Speed (`taskset -c 0,1`)

- **Born-digital PDF (`--no-mfr` / text-only):** `0.16–0.45s/page` median
- **Tables (`tables_derived`, `olmo_table_tests`):** `0.71–0.74s/page` median (`2–3×` faster than `v0.2`)
- **Scanned forms / prose (`realdoc_pages`, `olmo_old_scans`):** `1.05–1.37s/page` median
- **Dense display-math pages (`arxiv_math`, `old_scans_math`):** `5.33–6.94s/page` median (`QInt8` KV-MFR; pass `--no-mfr` for `0.17–1.30s/page`)

*(Note: `Indra`-based `v0.2.1` remains tagged in git history if you specifically need higher accuracy on Cyrillic/Korean/degraded historical non-math scans.)*

---

## Quickstart

```bash
pip install git+https://github.com/Im-Jam/toaster-ocr.git

# 1. Fetch pinned models (~95 MB total) and build QInt8 KV-cached MFR decoders locally:
toaster-ocr download

# 2. OCR an image or PDF to Markdown:
toaster-ocr ocr document.pdf -o document.md --threads 2

# Skip display-math MFR when max CPU speed matters:
toaster-ocr ocr document.pdf --no-mfr

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

The integration code in this repository is **Apache-2.0** (`LICENSE`). Model weights are downloaded on demand from Hugging Face / LiteParse under their own upstream licenses — notably `OCRia-nano-1` weights are **Modified AI Pubs OpenRAIL-M**, `LiteParse` and `PP-DocLayout-S-ONNX` are **Apache-2.0**, and `pix2text-mfr-1.5` is **MIT**. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
