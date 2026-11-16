import base64
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from PIL import Image

from toaster_ocr.server import make_handler


class _MockService:
    def __init__(self):
        self.calls = []

    def run_bytes(self, raw_image: bytes, mode: str, math: bool = True, ppv6_threshold=None, table_gate=None):
        self.calls.append((len(raw_image), mode, math, ppv6_threshold, table_gate))
        return {
            "markdown": f"MOCK-{mode}-math={math}-thr={ppv6_threshold}",
            "parts_count": 1,
            "tables_attempted": 0,
            "tables_replaced": 0,
            "timings": {"total_seconds": 0.01},
        }


def _png_uri() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TestServer(unittest.TestCase):
    def test_auth_models_chat_and_ocr(self):
        svc = _MockService()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc, api_key="secret-key"))
        port = httpd.server_address[1]
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as r:
                self.assertEqual(r.status, 200)
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models")
            self.assertEqual(ctx.exception.code, 401)

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/ocr",
                data=json.dumps({"model": "pp-v6-local", "image": _png_uri()}).encode(),
                method="POST",
                headers={"Authorization": "Bearer secret-key", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as r:
                body = json.loads(r.read())
                self.assertEqual(body["markdown"], "MOCK-pp-v6-math=True-thr=None")

            req2 = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "indra-pp-local",
                        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _png_uri()}}]}],
                    }
                ).encode(),
                method="POST",
                headers={"Authorization": "Bearer secret-key", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req2) as r:
                body2 = json.loads(r.read())
                self.assertEqual(body2["choices"][0]["message"]["content"], "MOCK-indra-pp-math=True-thr=None")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
