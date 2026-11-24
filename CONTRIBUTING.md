# Contributing

`toaster-ocr` is kept intentionally lean: pure-ONNX CPU inference (`onnxruntime`, zero PyTorch, zero PaddlePaddle, zero cloud APIs in core).

## Running tests

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

## Guidelines

1. Keep `core` dependencies CPU-only and free of `torch` / `paddlepaddle` / `transformers`.
2. Benchmark any pipeline change on held-out pages across both **Omni Parse Bench** (`datalab-to/omni_parse_bench`) and independent **`allenai/olmOCR-bench`** rather than tuning on a single metric.
3. Preserve fragment conservation when editing table or formula assembly — never silently drop OCR lines.
