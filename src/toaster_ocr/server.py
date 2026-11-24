"""Minimal stdlib OpenAI-compatible HTTP server for `toaster-ocr` (zero extra web dependencies)."""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image

from .engine import ToasterOCR

MAX_BODY_BYTES = 40 * 1024 * 1024
_ENGINE_LOCK = threading.Lock()


def _decode_data_uri(uri: str) -> tuple[bytes, str]:
    if uri.startswith("data:"):
        head, b64 = uri.split(",", 1)
        mime = head[5:].split(";", 1)[0].lower()
        return base64.b64decode(b64), mime
    if uri.startswith(("http://", "https://")):
        with urllib.request.urlopen(uri, timeout=30) as r:
            mime = (r.headers.get_content_type() or "").lower()
            return r.read(MAX_BODY_BYTES), mime
    raise ValueError("Unsupported URI scheme; expected data:... or http(s)://...")


def _extract_media_uri(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("prompt"), str):
        return payload["prompt"]
    if isinstance(payload.get("image"), str):
        return payload["image"]
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    iu = part.get("image_url")
                    if isinstance(iu, dict) and iu.get("url"):
                        return str(iu["url"])
                    if isinstance(iu, str):
                        return iu
                if part.get("type") == "file":
                    fd = (part.get("file") or {}).get("file_data")
                    if fd:
                        return str(fd)
    raise ValueError("No image_url, file_data, image, or prompt data URI found in request")


def make_handler(engine: ToasterOCR, api_key: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, obj: dict[str, Any]) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authorized(self) -> bool:
            if not api_key:
                return True
            auth = self.headers.get("Authorization", "")
            return auth == f"Bearer {api_key}"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "model": "toaster-ocr"})
            elif self.path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{"id": "toaster-ocr", "object": "model"}]})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._authorized():
                self._send_json(401, {"error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(413, {"error": "invalid content length"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                uri = _extract_media_uri(body)
                raw_bytes, mime = _decode_data_uri(uri)
                baseline = bool(body.get("baseline", False))
                with _ENGINE_LOCK:
                    if mime == "application/pdf" or raw_bytes[:4] == b"%PDF":
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                            tf.write(raw_bytes)
                            tmp_path = Path(tf.name)
                        try:
                            res = engine.read_pdf(tmp_path, baseline=baseline)
                        finally:
                            tmp_path.unlink(missing_ok=True)
                    else:
                        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                        res = engine.read_image(img, baseline=baseline)
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
                return

            if self.path == "/v1/ocr":
                self._send_json(
                    200,
                    {
                        "markdown": res.markdown,
                        "pages": res.pages,
                        "elapsed_seconds": res.elapsed_seconds,
                        "stats": res.stats,
                    },
                )
            elif self.path in ("/v1/chat/completions", "/v1/completions"):
                now = int(time.time())
                cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                if self.path == "/v1/completions":
                    self._send_json(
                        200,
                        {"id": cid, "object": "text_completion", "created": now, "model": "toaster-ocr", "choices": [{"index": 0, "text": res.markdown, "finish_reason": "stop"}]},
                    )
                else:
                    self._send_json(
                        200,
                        {
                            "id": cid,
                            "object": "chat.completion",
                            "created": now,
                            "model": "toaster-ocr",
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": res.markdown}, "finish_reason": "stop"}],
                        },
                    )
            else:
                self._send_json(404, {"error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


def serve(*, host: str = "127.0.0.1", port: int = 8129, threads: int = 2) -> None:
    engine = ToasterOCR(threads=threads)
    api_key = os.environ.get("TOASTER_OCR_API_KEY") or None
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, api_key=api_key))
    print(f"toaster-ocr listening on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
