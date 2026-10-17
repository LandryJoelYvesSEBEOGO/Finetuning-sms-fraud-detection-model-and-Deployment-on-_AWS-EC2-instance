"""Locate, download and verify the exported ONNX model artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from sms_fraud.config import Settings

logger = logging.getLogger(__name__)

MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"
METADATA_FILE = "metadata.json"
ARTIFACT_FILES = (MODEL_FILE, TOKENIZER_FILE, METADATA_FILE)


class ArtifactError(RuntimeError):
    """The ONNX artifacts are missing, unreachable or corrupted."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    """Stream url to destination atomically (no partial file is ever visible)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, out, length=1 << 20)
        os.replace(tmp_name, destination)
    except Exception as exc:
        Path(tmp_name).unlink(missing_ok=True)
        raise ArtifactError(f"Could not download {url}: {exc}") from exc


def _verify(directory: Path, metadata: dict, expected_model_sha256: str | None) -> None:
    for name, expected in metadata.get("sha256", {}).items():
        actual = sha256_of(directory / name)
        if actual != expected:
            raise ArtifactError(f"{name}: checksum mismatch ({actual} != {expected})")
    if expected_model_sha256 and metadata["sha256"].get(MODEL_FILE) != expected_model_sha256:
        raise ArtifactError(
            f"{MODEL_FILE}: checksum does not match the pinned SMS_FRAUD_ONNX_SHA256"
        )


def load_metadata(directory: Path) -> dict:
    return json.loads((directory / METADATA_FILE).read_text())


def ensure_artifacts(settings: Settings) -> Path:
    """Return a directory containing the ONNX artifacts, downloading them if needed."""
    directory = settings.resolved_onnx_dir()
    if all((directory / name).is_file() for name in ARTIFACT_FILES):
        logger.info("Using ONNX artifacts from %s", directory)
        return directory

    if not settings.onnx_url:
        raise ArtifactError(
            f"ONNX artifacts not found in {directory} and SMS_FRAUD_ONNX_URL is not set. "
            "Run `python -m sms_fraud.export` or configure a download URL."
        )

    base_url = settings.onnx_url.rstrip("/")
    logger.info("Downloading ONNX artifacts from %s to %s", base_url, directory)
    # Metadata goes last: its presence marks a complete download.
    for name in (MODEL_FILE, TOKENIZER_FILE, METADATA_FILE):
        _download(f"{base_url}/{name}", directory / name)
    try:
        _verify(directory, load_metadata(directory), settings.onnx_sha256)
    except Exception:
        for name in ARTIFACT_FILES:
            (directory / name).unlink(missing_ok=True)
        raise
    return directory
