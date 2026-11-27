# Third-Party Notices & Model Provenance

The Python integration code authored in this repository is licensed under the **Apache License 2.0** (`LICENSE`).

**No model weights or benchmark documents are bundled in this repository.** When you run `toaster-ocr download`, upstream models are fetched directly from Hugging Face / LiteParse under each upstream project's own license.

## Upstream Models & Code

1. **`LiteParse` (`run-llama/liteparse`) & `OCRia-nano-1` (`llamaindex/OCRia-nano-1`)**
   - **Code license (`liteparse`):** Apache-2.0 (`liteparse>=2.21.1`).
   - **Model weights (`ocria-nano`, ~55.8 MB):** **Modified AI Pubs OpenRAIL-M** (contains commercial/revenue & attribution terms — review the upstream `LICENSE` at `huggingface.co/llamaindex/OCRia-nano-1` before commercial deployment).

2. **`Indra OCR` (`datalab-to/indra-ocr`, optional for `fast` image route / `balanced` System MoE / `indra` mode)**
   - **Code license (`indra-ocr`):** Apache-2.0 (`indra-ocr>=1.1.2`).
   - **Model weights (`datalab-to/indra-ocr`, ~76 MB):** **Modified AI Pubs OpenRAIL-M** (pinned revision `bd02167fdf2005a19962cc15188f8496a2c87d3b`).

3. **`SLANet-plus` (`PaddlePaddle/SLANet_plus_onnx`, `slanet-plus.onnx`, 7.4 MB) & `DeepDoc TSR` (`InfiniFlow/deepdoc`, `tsr.onnx`, 12.2 MB)**
   - **License:** Apache-2.0 (pinned revisions `7dbe640e127602bf506815e822c09758de73c482` and `9c7aa2c730d7a242d7f04cf6109a6a239aefc717`).

4. **`PP-DocLayout-S-ONNX` (`stefanj0/PP-DocLayout-S-ONNX`, `pp_doclayout_s.onnx`, 4.7 MB)**
   - **License:** Apache-2.0 (pinned revision `7e851b9e5ad66613d7318a3213ab3336ebe0a514`).

5. **`pix2text-mfr-1.5` (`breezedeus/pix2text-mfr-1.5`, 34.3 MB QInt8)**
   - **License:** MIT (pinned revision `1cef9f0bdcd6a4c63df7de1311fb0894593340cc`).
   - `toaster-ocr download` converts `decoder_model.onnx` locally into KV-cached `mfr_dec_init.onnx` + `mfr_dec_step.onnx` (`toaster_ocr/mfr_kv.py`) and applies dynamic `QInt8` quantization (`MatMul`/`Gemm`).

## Benchmark Evaluators & Datasets

- **Omni Parse Bench (`datalab-to/omni_parse_bench`):** Apache-2.0 unit-test evaluator + 2,937-page / 16,288-test dataset.
- **olmOCR-bench (`allenai/olmOCR-bench` & `allenai/olmocr`):** Dataset is ODC-BY-1.0; evaluator is Apache-2.0.
