"""Exercise TransformersPredictor end-to-end on a tiny, randomly initialised model.

The base model is saved *without* a classification head, like
``distilbert-base-uncased`` on the Hub: the head must come from the adapter.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from sms_fraud.config import Settings  # noqa: E402
from sms_fraud.predictor import TransformersPredictor  # noqa: E402

pytestmark = pytest.mark.model

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "you", "win", "a", "prize", "call", "now"]


@pytest.fixture(scope="module")
def model_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertModel,
        DistilBertTokenizerFast,
    )

    root = tmp_path_factory.mktemp("tiny-model")
    base_dir, adapter_dir = root / "base", root / "adapter"

    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB))
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab_file))

    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(VOCAB),
        dim=16,
        n_layers=2,
        n_heads=2,
        hidden_dim=32,
        max_position_embeddings=512,
        num_labels=2,
    )
    DistilBertModel(config).save_pretrained(base_dir)

    classifier = DistilBertForSequenceClassification.from_pretrained(base_dir, num_labels=2)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=2, lora_alpha=4, target_modules=["q_lin", "v_lin"]
    )
    peft_model = get_peft_model(classifier, lora)
    peft_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return base_dir, adapter_dir


def _settings(model_dirs: tuple[Path, Path], **overrides: object) -> Settings:
    base_dir, adapter_dir = model_dirs
    return Settings(
        _env_file=None,
        base_model=str(base_dir),
        adapter_repo=str(adapter_dir),
        adapter_revision=None,
        **overrides,
    )


def test_returns_one_probability_per_message(model_dirs: tuple[Path, Path]) -> None:
    predictor = TransformersPredictor(_settings(model_dirs))
    scores = predictor.predict_proba(["you win a prize", "call now", "hello"])
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_long_messages_are_truncated(model_dirs: tuple[Path, Path]) -> None:
    predictor = TransformersPredictor(_settings(model_dirs))
    [score] = predictor.predict_proba(["you win a prize " * 1000])
    assert 0.0 <= score <= 1.0


def test_head_is_restored_from_adapter(model_dirs: tuple[Path, Path]) -> None:
    """Two independent loads must agree: no randomly initialised layer is left."""
    messages = ["you win a prize", "call now"]
    first = TransformersPredictor(_settings(model_dirs)).predict_proba(messages)
    second = TransformersPredictor(_settings(model_dirs)).predict_proba(messages)
    assert first == pytest.approx(second, abs=1e-6)


def test_fraud_label_index_selects_the_class(model_dirs: tuple[Path, Path]) -> None:
    fraud_is_1 = TransformersPredictor(_settings(model_dirs, fraud_label_index=1))
    fraud_is_0 = TransformersPredictor(_settings(model_dirs, fraud_label_index=0))
    [p1] = fraud_is_1.predict_proba(["you win"])
    [p0] = fraud_is_0.predict_proba(["you win"])
    assert p0 + p1 == pytest.approx(1.0, abs=1e-6)


def test_version_reflects_revision(model_dirs: tuple[Path, Path]) -> None:
    predictor = TransformersPredictor(_settings(model_dirs))
    assert predictor.version.endswith("@local")
