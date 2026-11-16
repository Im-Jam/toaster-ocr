"""Unified ToasterOCR pipeline supporting indra, indra-pp, pp-v6 (cascade), and pp-v6-full."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .indra_engine import DEFAULT_PPV6_CONF_THRESHOLD, IndraEngine
from .ppv3_layout import PPDocLayoutV3Detector, regroup
from .slanet_tables import SLANetTables

MODES = ("indra", "indra-pp", "pp-v6", "pp-v6-full")


class _JsonWorker:
    """Optional subprocess bridge when PPv3 or SLANet dependencies live in separate venvs."""

    def __init__(self, python_exe: str, code: str, threads: int):
        env = dict(
            os.environ,
            OMP_NUM_THREADS=str(threads),
            OPENBLAS_NUM_THREADS=str(threads),
            MKL_NUM_THREADS=str(threads),
            PYTHONDONTWRITEBYTECODE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        self.proc = subprocess.Popen(
            [python_exe, "-u", "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=env,
        )
        line = self.proc.stdout.readline()
        if not line or not json.loads(line).get("ready"):
            raise RuntimeError(f"Subprocess worker failed to start (python={python_exe})")

    def call(self, req: dict[str, Any]) -> dict[str, Any]:
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Subprocess worker exited (rc={self.proc.poll()})")
        resp = json.loads(line)
        if not resp.get("success", False):
            raise RuntimeError(resp.get("error", "unknown worker error"))
        return resp

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


_PPV3_WORKER_CODE = """
import json, os, sys
_OUT = sys.stdout; sys.stdout = sys.stderr
from toaster_ocr.ppv3_layout import PPDocLayoutV3Detector
det = PPDocLayoutV3Detector(threads=int(os.environ.get("OCR_THREADS", "2")))
print(json.dumps({"ready": True}), file=_OUT, flush=True)
for line in sys.stdin:
    if not line.strip(): continue
    try:
        req = json.loads(line)
        res = det.detect(req["image_path"], int(req["width"]), int(req["height"]))
        print(json.dumps({"success": True, **res}, ensure_ascii=False), file=_OUT, flush=True)
    except Exception as e:
        print(json.dumps({"success": False, "error": f"{type(e).__name__}: {e}"}), file=_OUT, flush=True)
"""

_SLANET_WORKER_CODE = """
import json, os, sys, time
_OUT = sys.stdout; sys.stdout = sys.stderr
from toaster_ocr.slanet_tables import SLANetTables
sl = SLANetTables(threads=int(os.environ.get("OCR_THREADS", "2")))
print(json.dumps({"ready": True}), file=_OUT, flush=True)
for line in sys.stdin:
    if not line.strip(): continue
    try:
        req = json.loads(line)
        out = sl.repair_tables(req["image_path"], req["regrouped"])
        print(json.dumps({"success": True, **out}, ensure_ascii=False), file=_OUT, flush=True)
    except Exception as e:
        print(json.dumps({"success": False, "error": f"{type(e).__name__}: {e}"}), file=_OUT, flush=True)
"""


class ToasterOCR:
    """Lazy-loading orchestrator for CPU-first small OCR specialist modes."""

    def __init__(self, threads: int = 2):
        self.threads = threads
        self.lock = threading.Lock()
        self._indra: IndraEngine | None = None
        self._ppv3: PPDocLayoutV3Detector | _JsonWorker | None = None
        self._slanet: SLANetTables | _JsonWorker | None = None

    def _get_indra(self) -> IndraEngine:
        if self._indra is None:
            self._indra = IndraEngine(threads=self.threads)
        return self._indra

    def _detect_ppv3(self, image_path: str, width: int, height: int) -> dict[str, Any]:
        if self._ppv3 is None:
            py = os.environ.get("TOASTER_OCR_PPV3_PYTHON")
            self._ppv3 = _JsonWorker(py, _PPV3_WORKER_CODE, self.threads) if py else PPDocLayoutV3Detector(threads=self.threads)
        if isinstance(self._ppv3, _JsonWorker):
            return self._ppv3.call({"image_path": image_path, "width": width, "height": height})
        return self._ppv3.detect(image_path, width, height)

    def _repair_slanet(self, image_path: str, regrouped: dict[str, Any]) -> dict[str, Any]:
        if self._slanet is None:
            py = os.environ.get("TOASTER_OCR_SLANET_PYTHON")
            self._slanet = _JsonWorker(py, _SLANET_WORKER_CODE, self.threads) if py else SLANetTables(threads=self.threads)
        if isinstance(self._slanet, _JsonWorker):
            return self._slanet.call({"image_path": image_path, "regrouped": regrouped})
        return self._slanet.repair_tables(image_path, regrouped)

    def run_image(
        self,
        image_path: str | Path,
        *,
        mode: str = "pp-v6",
        math: bool = True,
        ppv6_threshold: float | None = None,
        table_gate: bool | None = None,
    ) -> dict[str, Any]:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        use_ppv6 = mode in ("pp-v6", "pp-v6-full")
        thr = 1.0 if mode == "pp-v6-full" else (DEFAULT_PPV6_CONF_THRESHOLD if ppv6_threshold is None else float(ppv6_threshold))
        # Default: pp-v6 gates PPv3+SLANet on Indra seeing a Table; indra-pp & pp-v6-full always run layout.
        gate_tables = (mode == "pp-v6") if table_gate is None else bool(table_gate)

        with self.lock, tempfile.TemporaryDirectory(prefix="toaster-ocr-") as td:
            t0 = time.monotonic()
            upright_png = Path(td) / "upright.png"
            native = self._get_indra().run_image(
                image_path,
                upright_out=upright_png,
                math=math,
                ppv6=use_ppv6,
                ppv6_threshold=thr,
            )
            has_indra_table = any(p.get("label") == "Table" for p in native.get("parts", []))
            if mode == "indra" or native.get("vertical") or not native.get("parts") or (gate_tables and not has_indra_table):
                return {
                    "markdown": native["markdown"],
                    "parts_count": len(native.get("parts", [])),
                    "tables_attempted": 0,
                    "tables_replaced": 0,
                    "table_layout_skipped": bool(gate_tables and not has_indra_table and mode != "indra"),
                    "timings": {
                        "indra_seconds": native["inference_seconds"],
                        "ppv6_seconds": native.get("ppv6_seconds", 0.0),
                        "ppv6_escalated": native.get("ppv6_escalated", 0),
                        "ppv6_candidates": native.get("ppv6_candidates", 0),
                        "total_seconds": time.monotonic() - t0,
                    },
                }

            ppv3 = self._detect_ppv3(native["image_path"], native["width"], native["height"])
            t_regroup = time.monotonic()
            regrouped, diag = regroup(native, ppv3["regions"], mode="tables_only")
            regroup_s = time.monotonic() - t_regroup
            has_tables = any(p.get("label") == "Table" for p in regrouped.get("parts", []))
            if has_tables:
                repaired = self._repair_slanet(native["image_path"], regrouped)
                md = repaired["markdown"]
                attempted = repaired.get("tables_attempted", 0)
                replaced = repaired.get("tables_replaced", 0)
                slanet_s = repaired.get("slanet_seconds", 0.0)
            else:
                md = regrouped["markdown"]
                attempted = 0
                replaced = 0
                slanet_s = 0.0

            return {
                "markdown": md,
                "parts_count": len(regrouped.get("parts", [])),
                "tables_attempted": attempted,
                "tables_replaced": replaced,
                "table_layout_skipped": False,
                "regroup_fallback": diag.get("fallback"),
                "timings": {
                    "indra_seconds": native["inference_seconds"],
                    "ppv6_seconds": native.get("ppv6_seconds", 0.0),
                    "ppv6_escalated": native.get("ppv6_escalated", 0),
                    "ppv6_candidates": native.get("ppv6_candidates", 0),
                    "ppv3_seconds": ppv3["inference_seconds"],
                    "regroup_seconds": regroup_s,
                    "slanet_seconds": slanet_s,
                    "total_seconds": time.monotonic() - t0,
                },
            }

    def close(self):
        if self._indra is not None:
            self._indra.close()
        if isinstance(self._ppv3, _JsonWorker):
            self._ppv3.close()
        if isinstance(self._slanet, _JsonWorker):
            self._slanet.close()
