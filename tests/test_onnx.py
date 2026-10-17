"""ONNX export and OnnxPredictor, checked against the PyTorch reference."""

import json
from pathlib import Path

import pytest

from sms_fraud.artifacts import METADATA_FILE, MODEL_FILE, TOKENIZER_FILE, sha256_of
from sms_fraud.config import Settings
from sms_fraud.export import check_parity, export, main
from sms_fraud.onnx_predictor import OnnxPredictor
from sms_fraud.predictor import TransformersPredictor
from tests.tiny_model import tiny_settings

pytestmark = pytest.mark.model

MESSAGES = ["you win a prize", "call now", "hello", "you win " * 400]


@pytest.fixture(scope="module")
def exported(tiny_model_dirs: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("onnx")
    export(tiny_settings(tiny_model_dirs), output)
    return output


def _onnx_settings(model_dirs: tuple[Path, Path], directory: Path, **overrides: object) -> Settings:
    return tiny_settings(model_dirs, onnx_dir=str(directory), onnx_url=None, **overrides)


def test_export_writes_artifacts_with_checksums(exported: Path) -> None:
    metadata = json.loads((exported / METADATA_FILE).read_text())
    assert metadata["precision"] == "fp32"
    assert metadata["pad_token"] == "[PAD]"
    for name in (MODEL_FILE, TOKENIZER_FILE):
        assert metadata["sha256"][name] == sha256_of(exported / name)


def test_onnx_matches_pytorch(tiny_model_dirs: tuple[Path, Path], exported: Path) -> None:
    reference = TransformersPredictor(tiny_settings(tiny_model_dirs))
    candidate = OnnxPredictor(_onnx_settings(tiny_model_dirs, exported))
    assert candidate.predict_proba(MESSAGES) == pytest.approx(
        reference.predict_proba(MESSAGES), abs=1e-5
    )


def test_padding_does_not_change_scores(tiny_model_dirs: tuple[Path, Path], exported: Path) -> None:
    predictor = OnnxPredictor(_onnx_settings(tiny_model_dirs, exported))
    batched = predictor.predict_proba(MESSAGES)
    single = [predictor.predict_proba([m])[0] for m in MESSAGES]
    assert batched == pytest.approx(single, abs=1e-5)


def test_fraud_label_index(tiny_model_dirs: tuple[Path, Path], exported: Path) -> None:
    p1 = OnnxPredictor(_onnx_settings(tiny_model_dirs, exported, fraud_label_index=1))
    p0 = OnnxPredictor(_onnx_settings(tiny_model_dirs, exported, fraud_label_index=0))
    assert p0.predict_proba(["win"])[0] + p1.predict_proba(["win"])[0] == pytest.approx(1.0)


def test_version_marks_onnx_backend(tiny_model_dirs: tuple[Path, Path], exported: Path) -> None:
    predictor = OnnxPredictor(_onnx_settings(tiny_model_dirs, exported))
    assert predictor.version.endswith("@local+onnx")


def test_check_parity_reports_tiny_gap(tiny_model_dirs: tuple[Path, Path], exported: Path) -> None:
    assert check_parity(tiny_settings(tiny_model_dirs), exported) < 1e-5


def test_cli_exits_zero_and_records_parity(
    tiny_model_dirs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir, adapter_dir = tiny_model_dirs
    monkeypatch.setenv("SMS_FRAUD_BASE_MODEL", str(base_dir))
    monkeypatch.setenv("SMS_FRAUD_ADAPTER_REPO", str(adapter_dir))
    monkeypatch.setenv("SMS_FRAUD_ADAPTER_REVISION", "")
    assert main(["--output", str(tmp_path)]) == 0
    metadata = json.loads((tmp_path / METADATA_FILE).read_text())
    assert metadata["parity_max_abs_diff"] < 1e-4
