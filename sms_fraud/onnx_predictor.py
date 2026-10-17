"""Lightweight inference with ONNX Runtime and the Rust `tokenizers` library."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from sms_fraud.artifacts import MODEL_FILE, TOKENIZER_FILE, ensure_artifacts, load_metadata
from sms_fraud.config import Settings

logger = logging.getLogger(__name__)


class OnnxPredictor:
    """Runs the exported (fp32, LoRA-merged) model without PyTorch."""

    def __init__(self, settings: Settings) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = ensure_artifacts(settings)
        metadata = load_metadata(directory)

        tokenizer = Tokenizer.from_file(str(directory / TOKENIZER_FILE))
        # Configured once here: encode_batch is then safe to call concurrently.
        tokenizer.enable_truncation(max_length=settings.max_tokens)
        tokenizer.enable_padding(pad_id=metadata["pad_token_id"], pad_token=metadata["pad_token"])
        self._tokenizer = tokenizer

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(directory / MODEL_FILE), options, providers=["CPUExecutionProvider"]
        )
        self._fraud_index = settings.fraud_label_index

        source = metadata["source"]
        revision = (source.get("adapter_revision") or "local")[:7]
        self.version = f"{source['adapter_repo']}@{revision}+onnx"
        logger.info("ONNX model loaded from %s", directory)

    def predict_proba(self, messages: Sequence[str]) -> list[float]:
        encodings = self._tokenizer.encode_batch(list(messages))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        (logits,) = self._session.run(
            ["logits"], {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        # Numerically stable softmax.
        shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities = shifted / shifted.sum(axis=-1, keepdims=True)
        return probabilities[:, self._fraud_index].astype(float).tolist()
