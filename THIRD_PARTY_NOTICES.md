# Third-Party Notices & Model Provenance

The Python integration code authored in this repository is licensed under the **Apache License 2.0** (`LICENSE`).

**No model weights or benchmark documents are bundled in this repository.** When you download or use the upstream models and benchmark datasets below, you must comply with each component's separate license and terms. Adding Apache-2.0 to this integration code does **not** relicense upstream models, weights, or datasets.

## Upstream Models & Code

1. **Indra OCR (`datalab-to/indra-ocr`)**
   - **Code license:** Apache-2.0 (`indra-ocr==1.1.2`, source commit `ca81e18f7403435a3545c8c8ae9c4a1cecdd32a4`).
   - **Model weights (`datalab-to/indra-ocr`):** Modified **AI Pubs OpenRAIL-M** (tested revision `bd02167fdf2005a19962cc15188f8496a2c87d3b`). Contains attribution/use restrictions, output share-alike provisions, and commercial/revenue thresholds. Review the upstream `LICENSE` before commercial or production deployment.

2. **`PP-OCRv6_small_rec_onnx` (`PaddlePaddle/PP-OCRv6_small_rec_onnx`)**
   - **License:** Apache-2.0 (tested revision `b8f84f0b80c529de40b4fbb3544b84fa7233a513`, `inference.onnx` + `inference.yml`).

3. **`PP-DocLayoutV3` (`PaddlePaddle/PP-DocLayoutV3`)**
   - **License:** Apache-2.0 (loaded via HuggingFace `transformers`).

4. **`SLANet-1M` (`dimtri009/SLANet-1M`)**
   - **Model card declaration:** MIT (tested revision `a8ea21bf31cebc0f04909ede8517e3029c32f19b`; note artifacts are Paddle inference files `inference.pdmodel`/`inference.pdiparams`). Upstream PaddleOCR source code is Apache-2.0.

## Benchmark Datasets & Evaluators

5. **OmniDocBench (`opendatalab/OmniDocBench`)**
   - Evaluator code (`0604d9843f1dfa3a292bd5f253ae83a1ba47cd24`) is Apache-2.0; the OmniDocBench dataset README states the dataset is for **research purposes only, not commercial use**.

6. **olmOCR-bench (`allenai/olmOCR-bench` & `allenai/olmocr`)**
   - Dataset (`allenai/olmOCR-bench`) is **ODC-BY-1.0**; `allenai/olmocr` evaluation code (`tests.py`, `table_parsing.py`, `repeatdetect.py`) is **Apache-2.0**.
