"""Model loading and inference."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Protocol

from sms_fraud.config import Settings

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """Anything able to turn messages into fraud probabilities."""

    version: str

    def predict_proba(self, messages: Sequence[str]) -> list[float]:
        """Return the fraud probability of each message, in order."""
        ...


class TransformersPredictor:
    """DistilBERT + LoRA adapter, merged and run with PyTorch on CPU."""

    def __init__(self, settings: Settings) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(
            "Loading model base=%s adapter=%s@%s",
            settings.base_model,
            settings.adapter_repo,
            settings.adapter_revision,
        )
        self._torch = torch
        # The adapter repository ships the tokenizer used during training.
        self._tokenizer = AutoTokenizer.from_pretrained(
            settings.adapter_repo, revision=settings.adapter_revision
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            settings.base_model, revision=settings.base_revision, num_labels=2
        )
        peft_model = PeftModel.from_pretrained(
            base, settings.adapter_repo, revision=settings.adapter_revision
        )
        # Merging the LoRA weights removes the adapter overhead at inference time.
        self._model = peft_model.merge_and_unload().eval()

        self._max_tokens = settings.max_tokens
        self._fraud_index = settings.fraud_label_index
        # Fast tokenizers are not safe to call concurrently with per-call
        # truncation/padding options, and CPU inference is compute-bound anyway.
        self._lock = threading.Lock()

        revision = (settings.adapter_revision or "local")[:7]
        self.version = f"{settings.adapter_repo}@{revision}"

    def predict_proba(self, messages: Sequence[str]) -> list[float]:
        torch = self._torch
        with self._lock:
            encoded = self._tokenizer(
                list(messages),
                truncation=True,
                max_length=self._max_tokens,
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self._model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[:, self._fraud_index]
        return probabilities.tolist()
