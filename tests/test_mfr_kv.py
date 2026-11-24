"""Offline verification of the pure-ONNX MFR KV-cache splitter (no HF network required)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toaster_ocr.mfr_kv import (
    ENC_DIM,
    ENC_SEQ_LEN,
    HEAD_DIM,
    HIDDEN_DIM,
    N_HEADS,
    N_LAYERS,
    VOCAB_SIZE,
    build_mfr_kv_onnx_from_weights,
)


def _make_synthetic_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    w: dict[str, np.ndarray] = {
        "embed_tokens": rng.normal(0, 0.02, (VOCAB_SIZE, HIDDEN_DIM)).astype(np.float32),
        "embed_positions": rng.normal(0, 0.02, (514, HIDDEN_DIM)).astype(np.float32),
        "ln_emb_w": np.ones(HIDDEN_DIM, dtype=np.float32),
        "ln_emb_b": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "lm_head": rng.normal(0, 0.02, (HIDDEN_DIM, VOCAB_SIZE)).astype(np.float32),
    }
    for i in range(N_LAYERS):
        for k in ("sq", "sk", "sv", "so", "cq", "co"):
            w[f"l{i}_{k}_w"] = rng.normal(0, 0.02, (HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)
            w[f"l{i}_{k}_b"] = np.zeros(HIDDEN_DIM, dtype=np.float32)
        for k in ("ck", "cv"):
            w[f"l{i}_{k}_w"] = rng.normal(0, 0.02, (ENC_DIM, HIDDEN_DIM)).astype(np.float32)
            w[f"l{i}_{k}_b"] = np.zeros(HIDDEN_DIM, dtype=np.float32)
        for k in ("sln", "cln", "fln"):
            w[f"l{i}_{k}_w"] = np.ones(HIDDEN_DIM, dtype=np.float32)
            w[f"l{i}_{k}_b"] = np.zeros(HIDDEN_DIM, dtype=np.float32)
        w[f"l{i}_fc1_w"] = rng.normal(0, 0.02, (HIDDEN_DIM, 1024)).astype(np.float32)
        w[f"l{i}_fc1_b"] = np.zeros(1024, dtype=np.float32)
        w[f"l{i}_fc2_w"] = rng.normal(0, 0.02, (1024, HIDDEN_DIM)).astype(np.float32)
        w[f"l{i}_fc2_b"] = np.zeros(HIDDEN_DIM, dtype=np.float32)
    return w


class TestMFRKVSplitter(unittest.TestCase):
    def test_build_and_execute_kv_onnx(self) -> None:
        rng = np.random.default_rng(7)
        w = _make_synthetic_weights(rng)
        with tempfile.TemporaryDirectory() as tmp:
            init_p, step_p = build_mfr_kv_onnx_from_weights(w, tmp)
            self.assertTrue(init_p.is_file())
            self.assertTrue(step_p.is_file())

            s_init = ort.InferenceSession(str(init_p), providers=["CPUExecutionProvider"])
            s_step = ort.InferenceSession(str(step_p), providers=["CPUExecutionProvider"])

            ehs = rng.normal(0, 0.1, (2, ENC_SEQ_LEN, ENC_DIM)).astype(np.float32)
            ck, cv = s_init.run(None, {"encoder_hidden_states": ehs})
            self.assertEqual(ck.shape, (N_LAYERS, 2, N_HEADS, HEAD_DIM, ENC_SEQ_LEN))
            self.assertEqual(cv.shape, (N_LAYERS, 2, N_HEADS, ENC_SEQ_LEN, HEAD_DIM))

            sk = np.zeros((N_LAYERS, 2, N_HEADS, HEAD_DIM, 0), dtype=np.float32)
            sv = np.zeros((N_LAYERS, 2, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
            tok = np.array([1, 1], dtype=np.int64)
            for pos in range(4):
                logits, sk, sv = s_step.run(
                    None,
                    {
                        "token": tok,
                        "pos": np.array(pos, dtype=np.int64),
                        "self_k": sk,
                        "self_v": sv,
                        "cross_k": ck,
                        "cross_v": cv,
                    },
                )
                self.assertEqual(logits.shape, (2, VOCAB_SIZE))
                self.assertEqual(sk.shape, (N_LAYERS, 2, N_HEADS, HEAD_DIM, pos + 1))
                self.assertEqual(sv.shape, (N_LAYERS, 2, N_HEADS, pos + 1, HEAD_DIM))
                tok = logits.argmax(-1).astype(np.int64)


if __name__ == "__main__":
    unittest.main()
