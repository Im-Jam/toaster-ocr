# Third-Party Notices & Model Provenance

The Python integration code authored in this repository is licensed under the **Apache License 2.0** (`LICENSE`).

**No model weights or benchmark documents are bundled in this repository.** When you run `toaster-ocr download`, upstream models are fetched directly from Hugging Face / LiteParse under each upstream project's own license.

## Upstream Models & Code

1. **`LiteParse` (`run-llama/liteparse`) & `OCRia-nano-1` (`llamaindex/OCRia-nano-1`)**
   - **Code license (`liteparse`):** Apache-2.0 (`liteparse>=2.21.1`).
   - **Model weights (`ocria-nano`, ~55.8 MB):** **Modified AI Pubs OpenRAIL-M** (contains commercial/revenue & attribution terms — review the upstream `LICENSE` at `huggingface.co/llamaindex/OCRia-nano-1` before commercial deployment).

2. **`PP-DocLayout-S-ONNX` (`stefanj0/PP-DocLayout-S-ONNX`, `pp_doclayout_s.onnx`, 4.7 MB)**
   - **License:** Apache-2.0 (pinned revision `7e851b9e5ad66613d7318a3213ab3336ebe0a514`).

3. **`pix2text-mfr-1.5` (`breezedeus/pix2text-mfr-1.5`, 34.3 MB QInt8)**
   - **License:** MIT (pinned revision `1cef9f0bdcd6a4c63df7de1311fb0894593340cc`).
   - `toaster-ocr download` converts `decoder_model.onnx` locally into KV-cached `mfr_dec_init.onnx` + `mfr_dec_step.onnx` (`toaster_ocr/mfr_kv.py`) and applies dynamic `QInt8` quantization (`MatMul`/`Gemm`).

## Benchmark Evaluators & Datasets

- **Omni Parse Bench (`datalab-to/omni_parse_bench`):** Apache-2.0 unit-test evaluator + 2,937-page / 16,288-test dataset.
- **olmOCR-bench (`allenai/olmOCR-bench` & `allenai/olmocr`):** Dataset is ODC-BY-1.0; evaluator is Apache-2.0.
