"""Exercise TransformersPredictor end-to-end on a tiny, randomly initialised model."""

from pathlib import Path

import pytest

from sms_fraud.predictor import TransformersPredictor
from tests.tiny_model import tiny_settings

pytestmark = pytest.mark.model


def test_returns_one_probability_per_message(tiny_model_dirs: tuple[Path, Path]) -> None:
    predictor = TransformersPredictor(tiny_settings(tiny_model_dirs))
    scores = predictor.predict_proba(["you win a prize", "call now", "hello"])
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_long_messages_are_truncated(tiny_model_dirs: tuple[Path, Path]) -> None:
    predictor = TransformersPredictor(tiny_settings(tiny_model_dirs))
    [score] = predictor.predict_proba(["you win a prize " * 1000])
    assert 0.0 <= score <= 1.0


def test_head_is_restored_from_adapter(tiny_model_dirs: tuple[Path, Path]) -> None:
    """Two independent loads must agree: no randomly initialised layer is left."""
    messages = ["you win a prize", "call now"]
    first = TransformersPredictor(tiny_settings(tiny_model_dirs)).predict_proba(messages)
    second = TransformersPredictor(tiny_settings(tiny_model_dirs)).predict_proba(messages)
    assert first == pytest.approx(second, abs=1e-6)


def test_fraud_label_index_selects_the_class(tiny_model_dirs: tuple[Path, Path]) -> None:
    fraud_is_1 = TransformersPredictor(tiny_settings(tiny_model_dirs, fraud_label_index=1))
    fraud_is_0 = TransformersPredictor(tiny_settings(tiny_model_dirs, fraud_label_index=0))
    [p1] = fraud_is_1.predict_proba(["you win"])
    [p0] = fraud_is_0.predict_proba(["you win"])
    assert p0 + p1 == pytest.approx(1.0, abs=1e-6)


def test_version_reflects_revision(tiny_model_dirs: tuple[Path, Path]) -> None:
    predictor = TransformersPredictor(tiny_settings(tiny_model_dirs))
    assert predictor.version.endswith("@local")
