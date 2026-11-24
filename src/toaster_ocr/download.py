"""Download pinned ONNX models from Hugging Face and build KV-cached MFR decoders locally."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import onnx
import yaml

from .mfr_kv import convert_decoder_onnx_to_kv, quantize_mfr_dir
from .paths import MODEL_SPECS, indra_dir, stack_models_dir


def _embed_slanet_chars(onnx_path: Path, yml_path: Path) -> None:
    cfg = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    chars = list(cfg["PostProcess"]["character_dict"])
    model = onnx.load(str(onnx_path))
    existing = {p.key: p for p in model.metadata_props}
    if "character" in existing:
        existing["character"].value = "\n".join(chars)
    else:
        entry = model.metadata_props.add()
        entry.key = "character"
        entry.value = "\n".join(chars)
    onnx.save(model, str(onnx_path))


def download_models(
    components: Iterable[str] | None = None,
    *,
    force: bool = False,
    quantize_mfr: bool = True,
) -> dict[str, Path]:
    """Download pinned models and build `mfr_dec_init.onnx` / `mfr_dec_step.onnx` locally."""
    from huggingface_hub import hf_hub_download, snapshot_download

    wanted = set(components or MODEL_SPECS.keys())
    unknown = wanted - set(MODEL_SPECS.keys())
    if unknown:
        raise ValueError(f"Unknown model components: {sorted(unknown)}")

    out: dict[str, Path] = {}
    models_root = stack_models_dir()
    models_root.mkdir(parents=True, exist_ok=True)

    if "indra" in wanted:
        dst = indra_dir()
        dst.mkdir(parents=True, exist_ok=True)
        if force or not (dst / "det.onnx").is_file():
            spec = MODEL_SPECS["indra"]
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                local_dir=str(dst),
            )
        out["indra"] = dst

    if "slanet_plus" in wanted:
        dst_onnx = models_root / "slanet-plus.onnx"
        dst_yml = models_root / "slanet-plus.yml"
        if force or not dst_onnx.is_file():
            spec = MODEL_SPECS["slanet_plus"]
            onnx_dl = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"], filename="inference.onnx"))
            yml_dl = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"], filename="inference.yml"))
            shutil.copy2(onnx_dl, dst_onnx)
            shutil.copy2(yml_dl, dst_yml)
            _embed_slanet_chars(dst_onnx, dst_yml)
        out["slanet_plus"] = dst_onnx

    if "deepdoc_tsr" in wanted:
        dst_tsr = models_root / "deepdoc_tsr.onnx"
        if force or not dst_tsr.is_file():
            spec = MODEL_SPECS["deepdoc_tsr"]
            tsr_dl = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"], filename="tsr.onnx"))
            shutil.copy2(tsr_dl, dst_tsr)
        out["deepdoc_tsr"] = dst_tsr

    if "pp_doclayout_s" in wanted:
        dst_dl = models_root / "pp_doclayout_s.onnx"
        if force or not dst_dl.is_file():
            spec = MODEL_SPECS["pp_doclayout_s"]
            dl_dl = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"], filename="pp_doclayout_s.onnx"))
            shutil.copy2(dl_dl, dst_dl)
        out["pp_doclayout_s"] = dst_dl

    if "pix2text_mfr15" in wanted:
        mfr_dir = models_root / "pix2text_mfr15"
        mfr_dir.mkdir(parents=True, exist_ok=True)
        enc_dst = mfr_dir / "encoder_model.onnx"
        init_dst = mfr_dir / "mfr_dec_init.onnx"
        step_dst = mfr_dir / "mfr_dec_step.onnx"
        tok_dst = mfr_dir / "tokenizer.json"
        if force or not (enc_dst.is_file() and init_dst.is_file() and step_dst.is_file() and tok_dst.is_file()):
            spec = MODEL_SPECS["pix2text_mfr15"]
            enc_dl = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"], filename="encoder_model.onnx"))
            dec_dl = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"], filename="decoder_model.onnx"))
            tok_dl = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"], filename="tokenizer.json"))
            shutil.copy2(enc_dl, enc_dst)
            shutil.copy2(tok_dl, tok_dst)
            convert_decoder_onnx_to_kv(dec_dl, mfr_dir)
            if quantize_mfr:
                quantize_mfr_dir(mfr_dir)
        out["pix2text_mfr15"] = mfr_dir

    return out
