"""Pure-ONNX KV-cache splitter for breezedeus/pix2text-mfr-1.5.

Converts the upstream uncached `decoder_model.onnx` (TrOCR / MBart-like 6-layer decoder)
into two clean ONNX graphs with identical mathematical output (~1.8e-4 FP32 diff, 100% token parity):
  - `mfr_dec_init.onnx` (~4.5 MB): precomputes 6-layer cross-attention K/V once per crop.
  - `mfr_dec_step.onnx` (~25.8 MB): O(1) single-token decode step with explicit self-attention KV cache.

This eliminates quadratic re-prefill on CPU (~9.6x speedup on 2 ARM64 cores: 3,058 ms -> 319 ms for 60 steps).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

N_LAYERS = 6
N_HEADS = 8
HEAD_DIM = 32
HIDDEN_DIM = 256
ENC_SEQ_LEN = 578
ENC_DIM = 384
VOCAB_SIZE = 1868


def extract_mfr_kv_weights(decoder_onnx_path: str | Path) -> dict[str, np.ndarray]:
    """Extract decoder weights from upstream `pix2text-mfr-1.5/decoder_model.onnx`."""
    m = onnx.load(str(decoder_onnx_path))
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    mm: dict[str, np.ndarray] = {}
    for n in m.graph.node:
        if n.op_type == "MatMul" and len(n.input) >= 2 and n.input[1] in inits:
            mm[n.output[0]] = inits[n.input[1]]

    w: dict[str, np.ndarray] = {
        "embed_tokens": inits["decoder.model.decoder.embed_tokens.weight"].astype(np.float32),
        "embed_positions": inits["decoder.model.decoder.embed_positions.weight"].astype(np.float32),
        "ln_emb_w": inits["decoder.model.decoder.layernorm_embedding.weight"].astype(np.float32),
        "ln_emb_b": inits["decoder.model.decoder.layernorm_embedding.bias"].astype(np.float32),
        "lm_head": mm["logits"].astype(np.float32),
    }
    scale = np.float32(1.0 / np.sqrt(float(HEAD_DIM)))
    for i in range(N_LAYERS):
        p = f"decoder.model.decoder.layers.{i}"
        mp = f"/decoder/decoder/layers.{i}"
        # Pre-fold 1/sqrt(head_dim) into q_proj weights & biases
        w[f"l{i}_sq_w"] = mm[f"{mp}/self_attn/q_proj/MatMul_output_0"].astype(np.float32) * scale
        w[f"l{i}_sq_b"] = inits[f"{p}.self_attn.q_proj.bias"].astype(np.float32) * scale
        w[f"l{i}_sk_w"] = mm[f"{mp}/self_attn/k_proj/MatMul_output_0"].astype(np.float32)
        w[f"l{i}_sk_b"] = inits[f"{p}.self_attn.k_proj.bias"].astype(np.float32)
        w[f"l{i}_sv_w"] = mm[f"{mp}/self_attn/v_proj/MatMul_output_0"].astype(np.float32)
        w[f"l{i}_sv_b"] = inits[f"{p}.self_attn.v_proj.bias"].astype(np.float32)
        w[f"l{i}_so_w"] = mm[f"{mp}/self_attn/out_proj/MatMul_output_0"].astype(np.float32)
        w[f"l{i}_so_b"] = inits[f"{p}.self_attn.out_proj.bias"].astype(np.float32)
        w[f"l{i}_sln_w"] = inits[f"{p}.self_attn_layer_norm.weight"].astype(np.float32)
        w[f"l{i}_sln_b"] = inits[f"{p}.self_attn_layer_norm.bias"].astype(np.float32)

        w[f"l{i}_cq_w"] = mm[f"{mp}/encoder_attn/q_proj/MatMul_output_0"].astype(np.float32) * scale
        w[f"l{i}_cq_b"] = inits[f"{p}.encoder_attn.q_proj.bias"].astype(np.float32) * scale
        w[f"l{i}_ck_w"] = mm[f"{mp}/encoder_attn/k_proj/MatMul_output_0"].astype(np.float32)
        w[f"l{i}_ck_b"] = inits[f"{p}.encoder_attn.k_proj.bias"].astype(np.float32)
        w[f"l{i}_cv_w"] = mm[f"{mp}/encoder_attn/v_proj/MatMul_output_0"].astype(np.float32)
        w[f"l{i}_cv_b"] = inits[f"{p}.encoder_attn.v_proj.bias"].astype(np.float32)
        w[f"l{i}_co_w"] = mm[f"{mp}/encoder_attn/out_proj/MatMul_output_0"].astype(np.float32)
        w[f"l{i}_co_b"] = inits[f"{p}.encoder_attn.out_proj.bias"].astype(np.float32)
        w[f"l{i}_cln_w"] = inits[f"{p}.encoder_attn_layer_norm.weight"].astype(np.float32)
        w[f"l{i}_cln_b"] = inits[f"{p}.encoder_attn_layer_norm.bias"].astype(np.float32)

        w[f"l{i}_fc1_w"] = mm[f"{mp}/fc1/MatMul_output_0"].astype(np.float32)
        w[f"l{i}_fc1_b"] = inits[f"{p}.fc1.bias"].astype(np.float32)
        w[f"l{i}_fc2_w"] = mm[f"{mp}/fc2/MatMul_output_0"].astype(np.float32)
        w[f"l{i}_fc2_b"] = inits[f"{p}.fc2.bias"].astype(np.float32)
        w[f"l{i}_fln_w"] = inits[f"{p}.final_layer_norm.weight"].astype(np.float32)
        w[f"l{i}_fln_b"] = inits[f"{p}.final_layer_norm.bias"].astype(np.float32)
    return w


def build_mfr_kv_onnx_from_weights(
    w: Mapping[str, np.ndarray],
    out_dir: str | Path,
    *,
    vocab_size: int | None = None,
    enc_seq_len: int = ENC_SEQ_LEN,
    enc_dim: int = ENC_DIM,
) -> tuple[Path, Path]:
    """Construct `mfr_dec_init.onnx` and `mfr_dec_step.onnx` directly from extracted weights."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    v_size = int(vocab_size or w["lm_head"].shape[1])

    # 1) mfr_dec_init.onnx: encoder_hidden_states [B, enc_seq_len, enc_dim] -> cross_k, cross_v
    init_inits = [
        numpy_helper.from_array(np.array([0, enc_seq_len, N_HEADS, HEAD_DIM], dtype=np.int64), "shape_b_seq_h_d"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axes_0"),
    ]
    init_nodes = []
    ck_outs, cv_outs = [], []

    for i in range(N_LAYERS):
        for kind in ("ck", "cv"):
            wn, bn = f"l{i}_{kind}_w", f"l{i}_{kind}_b"
            init_inits.append(numpy_helper.from_array(w[wn], wn))
            init_inits.append(numpy_helper.from_array(w[bn], bn))
            mm_out = f"l{i}_{kind}_mm"
            add_out = f"l{i}_{kind}_add"
            rsh_out = f"l{i}_{kind}_rsh"
            tr_out = f"l{i}_{kind}_tr"
            unsq_out = f"l{i}_{kind}_unsq"
            init_nodes.append(helper.make_node("MatMul", ["encoder_hidden_states", wn], [mm_out]))
            init_nodes.append(helper.make_node("Add", [mm_out, bn], [add_out]))
            init_nodes.append(helper.make_node("Reshape", [add_out, "shape_b_seq_h_d"], [rsh_out]))
            perm = [0, 2, 3, 1] if kind == "ck" else [0, 2, 1, 3]
            init_nodes.append(helper.make_node("Transpose", [rsh_out], [tr_out], perm=perm))
            init_nodes.append(helper.make_node("Unsqueeze", [tr_out, "axes_0"], [unsq_out]))
            (ck_outs if kind == "ck" else cv_outs).append(unsq_out)

    init_nodes.append(helper.make_node("Concat", ck_outs, ["cross_k"], axis=0))
    init_nodes.append(helper.make_node("Concat", cv_outs, ["cross_v"], axis=0))

    init_graph = helper.make_graph(
        init_nodes,
        "mfr_dec_init",
        [helper.make_tensor_value_info("encoder_hidden_states", TensorProto.FLOAT, ["B", enc_seq_len, enc_dim])],
        [
            helper.make_tensor_value_info("cross_k", TensorProto.FLOAT, [N_LAYERS, "B", N_HEADS, HEAD_DIM, enc_seq_len]),
            helper.make_tensor_value_info("cross_v", TensorProto.FLOAT, [N_LAYERS, "B", N_HEADS, enc_seq_len, HEAD_DIM]),
        ],
        init_inits,
    )
    init_model = helper.make_model(init_graph, opset_imports=[helper.make_opsetid("", 17)])
    init_model.ir_version = 8
    init_path = out / "mfr_dec_init.onnx"
    onnx.save(init_model, str(init_path))

    # 2) mfr_dec_step.onnx: single-token step with self-attn KV cache
    step_inits = [
        numpy_helper.from_array(w["embed_tokens"], "embed_tokens"),
        numpy_helper.from_array(w["embed_positions"], "embed_positions"),
        numpy_helper.from_array(w["ln_emb_w"], "ln_emb_w"),
        numpy_helper.from_array(w["ln_emb_b"], "ln_emb_b"),
        numpy_helper.from_array(w["lm_head"], "lm_head"),
        numpy_helper.from_array(np.array(16.0, dtype=np.float32), "scale_16"),
        numpy_helper.from_array(np.array(2, dtype=np.int64), "pos_offset_2"),
        numpy_helper.from_array(np.array([0, N_HEADS, 1, HEAD_DIM], dtype=np.int64), "shape_b_8_1_32"),
        numpy_helper.from_array(np.array([0, N_HEADS, HEAD_DIM, 1], dtype=np.int64), "shape_b_8_32_1"),
        numpy_helper.from_array(np.array([0, HIDDEN_DIM], dtype=np.int64), "shape_b_256"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axes_0"),
    ]
    for i in range(N_LAYERS):
        step_inits.append(numpy_helper.from_array(np.array(i, dtype=np.int64), f"idx_{i}"))
        for suffix in (
            "sq_w", "sq_b", "sk_w", "sk_b", "sv_w", "sv_b", "so_w", "so_b", "sln_w", "sln_b",
            "cq_w", "cq_b", "co_w", "co_b", "cln_w", "cln_b",
            "fc1_w", "fc1_b", "fc2_w", "fc2_b", "fln_w", "fln_b",
        ):
            name = f"l{i}_{suffix}"
            step_inits.append(numpy_helper.from_array(w[name], name))

    step_nodes = [
        helper.make_node("Gather", ["embed_tokens", "token"], ["tok_emb"], axis=0),
        helper.make_node("Mul", ["tok_emb", "scale_16"], ["tok_emb_scaled"]),
        helper.make_node("Add", ["pos", "pos_offset_2"], ["pos_idx"]),
        helper.make_node("Gather", ["embed_positions", "pos_idx"], ["pos_emb"], axis=0),
        helper.make_node("Add", ["tok_emb_scaled", "pos_emb"], ["emb_sum"]),
        helper.make_node("LayerNormalization", ["emb_sum", "ln_emb_w", "ln_emb_b"], ["x_0"], axis=-1, epsilon=1e-5),
    ]

    new_sk_list, new_sv_list = [], []
    for i in range(N_LAYERS):
        xin, xout = f"x_{i}", f"x_{i + 1}"
        # Self-attention projections
        step_nodes.extend([
            helper.make_node("MatMul", [xin, f"l{i}_sq_w"], [f"l{i}_sq_mm"]),
            helper.make_node("Add", [f"l{i}_sq_mm", f"l{i}_sq_b"], [f"l{i}_sq_add"]),
            helper.make_node("Reshape", [f"l{i}_sq_add", "shape_b_8_1_32"], [f"l{i}_sq"]),
            helper.make_node("MatMul", [xin, f"l{i}_sk_w"], [f"l{i}_sk_mm"]),
            helper.make_node("Add", [f"l{i}_sk_mm", f"l{i}_sk_b"], [f"l{i}_sk_add"]),
            helper.make_node("Reshape", [f"l{i}_sk_add", "shape_b_8_32_1"], [f"l{i}_sk"]),
            helper.make_node("MatMul", [xin, f"l{i}_sv_w"], [f"l{i}_sv_mm"]),
            helper.make_node("Add", [f"l{i}_sv_mm", f"l{i}_sv_b"], [f"l{i}_sv_add"]),
            helper.make_node("Reshape", [f"l{i}_sv_add", "shape_b_8_1_32"], [f"l{i}_sv"]),
            # Concat with past self KV
            helper.make_node("Gather", ["self_k", f"idx_{i}"], [f"l{i}_prev_sk"], axis=0),
            helper.make_node("Gather", ["self_v", f"idx_{i}"], [f"l{i}_prev_sv"], axis=0),
            helper.make_node("Concat", [f"l{i}_prev_sk", f"l{i}_sk"], [f"l{i}_cur_sk"], axis=3),
            helper.make_node("Concat", [f"l{i}_prev_sv", f"l{i}_sv"], [f"l{i}_cur_sv"], axis=2),
            helper.make_node("Unsqueeze", [f"l{i}_cur_sk", "axes_0"], [f"l{i}_cur_sk_u"]),
            helper.make_node("Unsqueeze", [f"l{i}_cur_sv", "axes_0"], [f"l{i}_cur_sv_u"]),
            # Self-attention scores & output
            helper.make_node("MatMul", [f"l{i}_sq", f"l{i}_cur_sk"], [f"l{i}_sa_scores"]),
            helper.make_node("Softmax", [f"l{i}_sa_scores"], [f"l{i}_sa_probs"], axis=-1),
            helper.make_node("MatMul", [f"l{i}_sa_probs", f"l{i}_cur_sv"], [f"l{i}_sa_ctx"]),
            helper.make_node("Reshape", [f"l{i}_sa_ctx", "shape_b_256"], [f"l{i}_sa_flat"]),
            helper.make_node("MatMul", [f"l{i}_sa_flat", f"l{i}_so_w"], [f"l{i}_so_mm"]),
            helper.make_node("Add", [f"l{i}_so_mm", f"l{i}_so_b"], [f"l{i}_so_add"]),
            helper.make_node("Add", [xin, f"l{i}_so_add"], [f"l{i}_sa_res"]),
            helper.make_node("LayerNormalization", [f"l{i}_sa_res", f"l{i}_sln_w", f"l{i}_sln_b"], [f"l{i}_sa_ln"], axis=-1, epsilon=1e-5),
            # Cross-attention
            helper.make_node("MatMul", [f"l{i}_sa_ln", f"l{i}_cq_w"], [f"l{i}_cq_mm"]),
            helper.make_node("Add", [f"l{i}_cq_mm", f"l{i}_cq_b"], [f"l{i}_cq_add"]),
            helper.make_node("Reshape", [f"l{i}_cq_add", "shape_b_8_1_32"], [f"l{i}_cq"]),
            helper.make_node("Gather", ["cross_k", f"idx_{i}"], [f"l{i}_ck"], axis=0),
            helper.make_node("Gather", ["cross_v", f"idx_{i}"], [f"l{i}_cv"], axis=0),
            helper.make_node("MatMul", [f"l{i}_cq", f"l{i}_ck"], [f"l{i}_ca_scores"]),
            helper.make_node("Softmax", [f"l{i}_ca_scores"], [f"l{i}_ca_probs"], axis=-1),
            helper.make_node("MatMul", [f"l{i}_ca_probs", f"l{i}_cv"], [f"l{i}_ca_ctx"]),
            helper.make_node("Reshape", [f"l{i}_ca_ctx", "shape_b_256"], [f"l{i}_ca_flat"]),
            helper.make_node("MatMul", [f"l{i}_ca_flat", f"l{i}_co_w"], [f"l{i}_co_mm"]),
            helper.make_node("Add", [f"l{i}_co_mm", f"l{i}_co_b"], [f"l{i}_co_add"]),
            helper.make_node("Add", [f"l{i}_sa_ln", f"l{i}_co_add"], [f"l{i}_ca_res"]),
            helper.make_node("LayerNormalization", [f"l{i}_ca_res", f"l{i}_cln_w", f"l{i}_cln_b"], [f"l{i}_ca_ln"], axis=-1, epsilon=1e-5),
            # FFN
            helper.make_node("MatMul", [f"l{i}_ca_ln", f"l{i}_fc1_w"], [f"l{i}_fc1_mm"]),
            helper.make_node("Add", [f"l{i}_fc1_mm", f"l{i}_fc1_b"], [f"l{i}_fc1_add"]),
            helper.make_node("Relu", [f"l{i}_fc1_add"], [f"l{i}_relu"]),
            helper.make_node("MatMul", [f"l{i}_relu", f"l{i}_fc2_w"], [f"l{i}_fc2_mm"]),
            helper.make_node("Add", [f"l{i}_fc2_mm", f"l{i}_fc2_b"], [f"l{i}_fc2_add"]),
            helper.make_node("Add", [f"l{i}_ca_ln", f"l{i}_fc2_add"], [f"l{i}_ff_res"]),
            helper.make_node("LayerNormalization", [f"l{i}_ff_res", f"l{i}_fln_w", f"l{i}_fln_b"], [xout], axis=-1, epsilon=1e-5),
        ])
        new_sk_list.append(f"l{i}_cur_sk_u")
        new_sv_list.append(f"l{i}_cur_sv_u")

    step_nodes.extend([
        helper.make_node("MatMul", [f"x_{N_LAYERS}", "lm_head"], ["logits"]),
        helper.make_node("Concat", new_sk_list, ["new_self_k"], axis=0),
        helper.make_node("Concat", new_sv_list, ["new_self_v"], axis=0),
    ])

    step_graph = helper.make_graph(
        step_nodes,
        "mfr_dec_step",
        [
            helper.make_tensor_value_info("token", TensorProto.INT64, ["B"]),
            helper.make_tensor_value_info("pos", TensorProto.INT64, []),
            helper.make_tensor_value_info("self_k", TensorProto.FLOAT, [N_LAYERS, "B", N_HEADS, HEAD_DIM, "T"]),
            helper.make_tensor_value_info("self_v", TensorProto.FLOAT, [N_LAYERS, "B", N_HEADS, "T", HEAD_DIM]),
            helper.make_tensor_value_info("cross_k", TensorProto.FLOAT, [N_LAYERS, "B", N_HEADS, HEAD_DIM, enc_seq_len]),
            helper.make_tensor_value_info("cross_v", TensorProto.FLOAT, [N_LAYERS, "B", N_HEADS, enc_seq_len, HEAD_DIM]),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", v_size]),
            helper.make_tensor_value_info("new_self_k", TensorProto.FLOAT, [N_LAYERS, "B", N_HEADS, HEAD_DIM, "T_next"]),
            helper.make_tensor_value_info("new_self_v", TensorProto.FLOAT, [N_LAYERS, "B", N_HEADS, "T_next", HEAD_DIM]),
        ],
        step_inits,
    )
    step_model = helper.make_model(step_graph, opset_imports=[helper.make_opsetid("", 17)])
    step_model.ir_version = 8
    step_path = out / "mfr_dec_step.onnx"
    onnx.save(step_model, str(step_path))
    return init_path, step_path


def convert_decoder_onnx_to_kv(decoder_onnx_path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    """Extract weights from `decoder_model.onnx` and emit `mfr_dec_init.onnx` + `mfr_dec_step.onnx`."""
    weights = extract_mfr_kv_weights(decoder_onnx_path)
    return build_mfr_kv_onnx_from_weights(weights, out_dir)


def quantize_onnx_int8(src_path: str | Path, dst_path: str | Path | None = None) -> Path:
    """Apply dynamic QInt8 quantization to MatMul/Gemm ops (~3.5x smaller, ~1.7-1.9x faster on CPU)."""
    import shutil
    import tempfile
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(src_path)
    dst = Path(dst_path) if dst_path is not None else src
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dst.parent)) as tmp:
        tmp_out = Path(tmp) / dst.name
        quantize_dynamic(
            str(src),
            str(tmp_out),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        shutil.move(str(tmp_out), str(dst))
    return dst


def quantize_mfr_dir(mfr_dir: str | Path) -> tuple[Path, Path, Path]:
    """Quantize encoder_model.onnx, mfr_dec_init.onnx, and mfr_dec_step.onnx to dynamic QInt8."""
    d = Path(mfr_dir)
    return (
        quantize_onnx_int8(d / "encoder_model.onnx"),
        quantize_onnx_int8(d / "mfr_dec_init.onnx"),
        quantize_onnx_int8(d / "mfr_dec_step.onnx"),
    )
