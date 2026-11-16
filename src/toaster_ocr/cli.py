"""CLI entrypoint for toaster-ocr."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .download import download_models
from .paths import MODEL_SPECS
from .pipeline import MODES, ToasterOCR
from .server import serve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="toaster-ocr", description="CPU-first composable document OCR")
    sub = ap.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Download pinned model artifacts from HuggingFace")
    dl.add_argument("components", nargs="*", choices=[*MODEL_SPECS.keys()], help="Default: all")

    rn = sub.add_parser("run", help="Run OCR on a local image and print markdown or JSON")
    rn.add_argument("image", help="Input PNG/JPEG/WebP path")
    rn.add_argument("--mode", choices=MODES, default="pp-v6")
    rn.add_argument("--no-math", action="store_true", help="Disable Indra LaTeX math reader")
    rn.add_argument("--ppv6-threshold", type=float, default=None, help="Confidence threshold (default 0.96, 1.0=all lines)")
    rn.add_argument("--always-layout", action="store_true", help="Run PP-DocLayoutV3+SLANet even if Indra sees no Table")
    rn.add_argument("--threads", type=int, default=2)
    rn.add_argument("--json", action="store_true", help="Emit full JSON with timings")

    sv = sub.add_parser("serve", help="Start OpenAI-compatible HTTP OCR server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--threads", type=int, default=2)
    sv.add_argument("--api-key", default=os.environ.get("TOASTER_OCR_API_KEY"))

    args = ap.parse_args(argv)
    if args.cmd == "download":
        res = download_models(args.components or None)
        print(json.dumps({k: str(v) for k, v in res.items()}, indent=2))
        return 0
    if args.cmd == "run":
        ocr = ToasterOCR(threads=args.threads)
        try:
            res = ocr.run_image(
                args.image,
                mode=args.mode,
                math=not args.no_math,
                ppv6_threshold=args.ppv6_threshold,
                table_gate=(False if args.always_layout else None),
            )
        finally:
            ocr.close()
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(res["markdown"])
        return 0
    if args.cmd == "serve":
        serve(host=args.host, port=args.port, api_key=args.api_key, threads=args.threads)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
