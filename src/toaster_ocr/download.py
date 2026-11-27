"""Download pinned models from Hugging Face / LiteParse and build QInt8 KV-cached MFR decoders."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import onnx

from .mfr_kv import convert_decoder_onnx_to_kv, quantize_mfr_dir
from .moe import ensure_x86_rec_latin_qdq_parity
from .paths import DOWNLOAD_COMPONENTS, MODEL_SPECS, OCRIA_NANO_PACKAGE, indra_dir, stack_models_dir


def _find_lit_bin() -> str:
    sibling = Path(sys.executable).with_name("lit")
    if sibling.is_file():
        return str(sibling)
    found = shutil.which("lit")
    if found:
        return found
    raise FileNotFoundError("LiteParse `lit` CLI not found; install `liteparse>=2.21.1` first.")


def _pull_ocria_nano(*, force: bool = False) -> Path:
    lit_bin = _find_lit_bin()
    if not force:
        chk = subprocess.run([lit_bin, "models", "list"], capture_output=True, text=True)
        if chk.returncode == 0 and OCRIA_NANO_PACKAGE in chk.stdout:
            for line in chk.stdout.splitlines():
                if OCRIA_NANO_PACKAGE in line:
                    p = Path(line.split()[-1])
                    ensure_x86_rec_latin_qdq_parity(p)
                    return p
    subprocess.run([lit_bin, "models", "pull", OCRIA_NANO_PACKAGE], check=True)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    p = base / "liteparse" / "models" / OCRIA_NANO_PACKAGE
    ensure_x86_rec_latin_qdq_parity(p)
    return p


def _embed_slanet_chars(onnx_path: Path, yml_path: Path) -> None:
    import yaml

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
    """Download pinned models and build QInt8 KV-cached `mfr_dec_init.onnx` / `mfr_dec_step.onnx`."""
    from huggingface_hub import hf_hub_download, snapshot_download

    wanted = set(components or DOWNLOAD_COMPONENTS)
    unknown = wanted - set(DOWNLOAD_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown model components: {sorted(unknown)}")

    out: dict[str, Path] = {}
    models_root = stack_models_dir()
    models_root.mkdir(parents=True, exist_ok=True)

    if "ocria_nano" in wanted:
        out["ocria_nano"] = _pull_ocria_nano(force=force)

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
            dl_dl = Path(
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    revision=spec["revision"],
                    filename="pp_doclayout_s.onnx",
                )
            )
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
            enc_dl = Path(
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    revision=spec["revision"],
                    filename="encoder_model.onnx",
                )
            )
            dec_dl = Path(
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    revision=spec["revision"],
                    filename="decoder_model.onnx",
                )
            )
            tok_dl = Path(
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    revision=spec["revision"],
                    filename="tokenizer.json",
                )
            )
            shutil.copy2(enc_dl, enc_dst)
            shutil.copy2(tok_dl, tok_dst)
            convert_decoder_onnx_to_kv(dec_dl, mfr_dir)
            if quantize_mfr:
                quantize_mfr_dir(mfr_dir)
        out["pix2text_mfr15"] = mfr_dir

    return out
