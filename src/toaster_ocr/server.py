"""OpenAI-compatible & /v1/ocr HTTP server for toaster-ocr."""
from __future__ import annotations

import base64
import binascii
import hmac
import io
import json
import os
import tempfile
import time
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .pipeline import ToasterOCR

MAX_BODY_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

MODEL_MAP = {
    "indra-ocr-local": ("indra", True, None),
    "indra-pp-local": ("indra-pp", True, None),
    "pp-v6-local": ("pp-v6", True, None),
    "pp-v6-full-local": ("pp-v6-full", True, 1.0),
}
PUBLIC_MODELS = ["indra-ocr-local", "indra-pp-local", "pp-v6-local", "pp-v6-full-local"]


class APIError(Exception):
    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error", code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type
        self.code = code


def decode_image_source(src: str) -> bytes:
    if not isinstance(src, str) or not src.strip():
        raise APIError(400, "Image source must be a non-empty string")
    s = src.strip()
    if s.startswith("data:"):
        header, sep, payload = s.partition(",")
        if not sep or ";base64" not in header.lower():
            raise APIError(400, "data: URL must be base64-encoded")
        s = payload
    elif "://" in s or urllib.parse.urlparse(s).scheme in ("http", "https", "file", "ftp"):
        raise APIError(400, "Remote URLs are disabled; pass a base64 string or data:image/...;base64,... URI")
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise APIError(400, f"Invalid base64 image payload: {exc}") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise APIError(413 if len(raw) > MAX_IMAGE_BYTES else 400, "Image empty or exceeds 16 MiB limit")
    return raw


def normalize_image_to_png(raw: bytes, out_path: Path) -> tuple[int, int]:
    old_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            w, h = im.size
            if w <= 0 or h <= 0 or w * h > MAX_IMAGE_PIXELS:
                raise APIError(400, f"Image dimensions out of bounds: {w}x{h}")
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.save(out_path, format="PNG")
            return im.size
    except APIError:
        raise
    except Exception as exc:
        raise APIError(400, f"Unsupported or corrupt image: {exc}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = old_max


def extract_chat_images_and_text(messages: Any) -> tuple[list[str], str]:
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "'messages' must be a non-empty list")
    target = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"), messages[-1])
    if not isinstance(target, dict):
        raise APIError(400, "Invalid message entry")
    content = target.get("content")
    images: list[str] = []
    texts: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            typ = item.get("type")
            if typ == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif typ in ("image_url", "input_image"):
                url = item.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if isinstance(url, str):
                    images.append(url)
            elif typ == "image" and isinstance(item.get("image"), str):
                images.append(item["image"])
    return images, "\n".join(texts).strip()


class ServiceAdapter:
    """Bridges raw image bytes to ToasterOCR (replaceable by mocks in tests)."""

    def __init__(self, ocr: ToasterOCR | None = None):
        self.ocr = ocr or ToasterOCR()

    def run_bytes(
        self,
        raw_image: bytes,
        mode: str,
        math: bool = True,
        ppv6_threshold: float | None = None,
        table_gate: bool | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="toaster-req-") as td:
            in_png = Path(td) / "input.png"
            normalize_image_to_png(raw_image, in_png)
            return self.ocr.run_image(in_png, mode=mode, math=math, ppv6_threshold=ppv6_threshold, table_gate=table_gate)


def make_handler(service: Any, api_key: str | None = None):
    started = time.monotonic()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, obj: Any):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, err: APIError):
            self._send_json(err.status, {"error": {"message": err.message, "type": err.err_type, "code": err.code}})

        def _require_auth(self):
            if not api_key:
                return
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                raise APIError(401, "Missing Bearer API key", "authentication_error", "invalid_api_key")
            token = auth[7:].strip()
            if not hmac.compare_digest(token.encode("utf-8"), api_key.encode("utf-8")):
                raise APIError(401, "Invalid API key", "authentication_error", "invalid_api_key")

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise APIError(400, "Invalid Content-Length") from exc
            if length <= 0 or length > MAX_BODY_BYTES:
                raise APIError(413 if length > MAX_BODY_BYTES else 400, "Request body empty or exceeds 20 MiB")
            raw = self.rfile.read(length)
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise APIError(400, f"Malformed JSON body: {exc}") from exc
            if not isinstance(obj, dict):
                raise APIError(400, "JSON body must be an object")
            return obj

        def do_GET(self):
            try:
                path = urllib.parse.urlparse(self.path).path
                if path == "/health":
                    self._send_json(200, {"status": "ok", "uptime_seconds": round(time.monotonic() - started, 1), "models": PUBLIC_MODELS})
                    return
                self._require_auth()
                if path == "/v1/models":
                    self._send_json(
                        200,
                        {"object": "list", "data": [{"id": m, "object": "model", "created": 1700000000, "owned_by": "toaster-ocr"} for m in PUBLIC_MODELS]},
                    )
                    return
                raise APIError(404, f"Not found: {path}", "invalid_request_error", "not_found")
            except APIError as err:
                self._send_error(err)

        def do_POST(self):
            try:
                self._require_auth()
                path = urllib.parse.urlparse(self.path).path
                body = self._read_json()

                if path == "/v1/chat/completions":
                    model = body.get("model", "pp-v6-local")
                    if model not in MODEL_MAP:
                        raise APIError(404, f"Unknown model: {model}. Available: {PUBLIC_MODELS}", "invalid_request_error", "model_not_found")
                    mode, use_math, thr = MODEL_MAP[model]
                    stream = bool(body.get("stream", False))
                    images, user_text = extract_chat_images_and_text(body.get("messages"))
                    if not images:
                        reply = "OCR Document" if user_text.startswith("### Task:") else "Attach an image (PNG/JPEG/WebP) to run OCR."
                    else:
                        outputs = []
                        for idx, src in enumerate(images):
                            res = service.run_bytes(decode_image_source(src), mode, math=use_math, ppv6_threshold=thr)
                            md = res["markdown"] or "(no text detected)"
                            outputs.append(md if len(images) == 1 else f"<!-- image {idx + 1} -->\n\n{md}")
                        reply = "\n\n---\n\n".join(outputs)
                    self._send_chat(model, reply, stream)
                    return

                if path == "/v1/ocr":
                    model = body.get("model", "pp-v6-local")
                    if model not in MODEL_MAP:
                        raise APIError(404, f"Unknown model: {model}. Available: {PUBLIC_MODELS}", "invalid_request_error", "model_not_found")
                    mode, use_math, thr = MODEL_MAP[model]
                    if "math" in body:
                        use_math = bool(body["math"])
                    if "ppv6_threshold" in body:
                        thr = float(body["ppv6_threshold"])
                    tgate = body.get("table_gate")
                    src = body.get("image") or body.get("image_url")
                    if isinstance(src, dict):
                        src = src.get("url")
                    if not isinstance(src, str):
                        raise APIError(400, "Provide 'image' or 'image_url'")
                    res = service.run_bytes(decode_image_source(src), mode, math=use_math, ppv6_threshold=thr, table_gate=tgate)
                    self._send_json(200, {"model": model, **res})
                    return

                raise APIError(404, f"Not found: {path}", "invalid_request_error", "not_found")
            except APIError as err:
                self._send_error(err)
            except Exception as exc:
                self._send_error(APIError(500, f"Internal OCR error: {type(exc).__name__}: {exc}", "server_error"))

        def _send_chat(self, model: str, text: str, stream: bool):
            cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            now = int(time.time())
            if not stream:
                self._send_json(
                    200,
                    {
                        "id": cid,
                        "object": "chat.completion",
                        "created": now,
                        "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    },
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in (
                {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
                {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ):
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, fmt, *args):
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080, api_key: str | None = None, threads: int = 2):
    svc = ServiceAdapter(ToasterOCR(threads=threads))
    httpd = ThreadingHTTPServer((host, port), make_handler(svc, api_key=api_key))
    try:
        httpd.serve_forever()
    finally:
        svc.ocr.close()
