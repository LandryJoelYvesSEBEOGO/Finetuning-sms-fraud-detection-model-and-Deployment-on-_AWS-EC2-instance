import json
from pathlib import Path

import pytest

from sms_fraud.artifacts import (
    ARTIFACT_FILES,
    METADATA_FILE,
    MODEL_FILE,
    TOKENIZER_FILE,
    ArtifactError,
    ensure_artifacts,
    sha256_of,
)
from sms_fraud.config import Settings


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A fake release directory, served through file:// URLs."""
    directory = tmp_path / "remote"
    directory.mkdir()
    (directory / MODEL_FILE).write_bytes(b"fake onnx graph")
    (directory / TOKENIZER_FILE).write_text("{}")
    metadata = {
        "sha256": {name: sha256_of(directory / name) for name in (MODEL_FILE, TOKENIZER_FILE)}
    }
    (directory / METADATA_FILE).write_text(json.dumps(metadata))
    return directory


def _settings(tmp_path: Path, remote: Path | None, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        onnx_dir=str(tmp_path / "local"),
        onnx_url=remote.as_uri() if remote else None,
        **overrides,
    )


def test_downloads_missing_artifacts(tmp_path: Path, remote: Path) -> None:
    directory = ensure_artifacts(_settings(tmp_path, remote))
    for name in ARTIFACT_FILES:
        assert (directory / name).read_bytes() == (remote / name).read_bytes()


def test_uses_local_artifacts_without_downloading(tmp_path: Path, remote: Path) -> None:
    ensure_artifacts(_settings(tmp_path, remote))
    # Second call works even though the remote is gone.
    assert ensure_artifacts(_settings(tmp_path, None)) == tmp_path / "local"


def test_corrupted_download_is_rejected_and_removed(tmp_path: Path, remote: Path) -> None:
    (remote / MODEL_FILE).write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        ensure_artifacts(_settings(tmp_path, remote))
    assert not any((tmp_path / "local" / name).exists() for name in ARTIFACT_FILES)


def test_pinned_checksum_is_enforced(tmp_path: Path, remote: Path) -> None:
    with pytest.raises(ArtifactError, match="pinned"):
        ensure_artifacts(_settings(tmp_path, remote, onnx_sha256="0" * 64))


def test_pinned_checksum_accepts_matching_model(tmp_path: Path, remote: Path) -> None:
    pinned = sha256_of(remote / MODEL_FILE)
    ensure_artifacts(_settings(tmp_path, remote, onnx_sha256=pinned))


def test_missing_artifacts_without_url(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        ensure_artifacts(_settings(tmp_path, None))


def test_unreachable_url(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="Could not download"):
        ensure_artifacts(_settings(tmp_path, tmp_path / "does-not-exist"))
    assert not list((tmp_path / "local").iterdir())


def test_default_dir_is_tmp_on_vercel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERCEL", "1")
    assert Settings(_env_file=None).resolved_onnx_dir().name == "sms-fraud-onnx"
    monkeypatch.delenv("VERCEL")
    assert Settings(_env_file=None).resolved_onnx_dir() == Path("artifacts/onnx")
