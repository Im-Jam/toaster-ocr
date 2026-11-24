"""CLI entrypoint for `toaster-ocr`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .download import download_models
from .engine import ToasterOCR
from .paths import MODEL_SPECS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toaster-ocr", description="Pure-ONNX CPU document OCR")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download pinned ONNX models and build KV-cached MFR decoders")
    p_dl.add_argument("--only", nargs="*", choices=sorted(MODEL_SPECS.keys()), help="Specific components to fetch")
    p_dl.add_argument("--force", action="store_true", help="Re-download and rebuild even if cached")

    p_ocr = sub.add_parser("ocr", help="Run OCR on an image or PDF")
    p_ocr.add_argument("input", type=Path, help="Path to image (.png/.jpg/.webp) or .pdf")
    p_ocr.add_argument("-o", "--output", type=Path, help="Write Markdown output to file instead of stdout")
    p_ocr.add_argument("--threads", type=int, default=2, help="ONNX Runtime intra-op CPU threads (default: 2)")
    p_ocr.add_argument("--baseline", action="store_true", help="Run stock Indra baseline without FastCPUStack upgrades")
    p_ocr.add_argument("--no-mfr", action="store_true", help="Disable PP-DocLayout-S + KV-MFR formula replacement")
    p_ocr.add_argument("--json", action="store_true", help="Emit JSON with markdown, pages, timing, and stats")

    p_srv = sub.add_parser("serve", help="Start local OpenAI-compatible HTTP server")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8129)
    p_srv.add_argument("--threads", type=int, default=2)

    args = parser.parse_args(argv)

    if args.cmd == "download":
        res = download_models(args.only, force=args.force)
        for k, v in sorted(res.items()):
            print(f"{k}: {v}")
        return 0

    if args.cmd == "ocr":
        engine = ToasterOCR(threads=args.threads)
        result = engine.read(
            args.input,
            baseline=args.baseline,
            enable_math_mfr=not args.no_mfr,
        )
        payload = (
            json.dumps(
                {
                    "markdown": result.markdown,
                    "pages": result.pages,
                    "elapsed_seconds": result.elapsed_seconds,
                    "stats": result.stats,
                },
                ensure_ascii=False,
                indent=2,
            )
            if args.json
            else result.markdown
        )
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload + ("\n" if not payload.endswith("\n") else ""))
        return 0

    if args.cmd == "serve":
        from .server import serve

        serve(host=args.host, port=args.port, threads=args.threads)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
