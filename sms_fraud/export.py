"""Convert the LoRA-merged model to ONNX (fp32) and check it matches PyTorch.

Usage: python -m sms_fraud.export [--output artifacts/onnx] [--atol 1e-4]

The weights are not modified: the adapter is merged into the base model and
the graph is exported in full precision (no quantization). The export fails
if the ONNX probabilities differ from the PyTorch reference by more than atol.
Requires requirements-ml.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sms_fraud.artifacts import METADATA_FILE, MODEL_FILE, TOKENIZER_FILE, sha256_of
from sms_fraud.config import Settings
from sms_fraud.onnx_predictor import OnnxPredictor
from sms_fraud.predictor import TransformersPredictor, load_merged_model

logger = logging.getLogger("sms_fraud.export")

OPSET = 17

PARITY_SAMPLES = [
    "Hey, are we still on for lunch tomorrow at 12?",
    "URGENT: Your bank account has been suspended. Verify your details at http://bit.ly/x",
    "Congratulations! You've won a $1000 gift card. Claim now by replying with your PIN.",
    "Your package could not be delivered. Pay the $1.99 redelivery fee: http://track-pkg.co",
    "Mom, I lost my phone, this is my new number. Can you send me $200 for rent?",
    "Reminder: your dentist appointment is on Monday at 9:30am.",
    "ok",
    "win " * 600,  # longer than 512 tokens: exercises truncation
]

# Hand-labelled examples (True = fraud). Informational only: they never fail the
# export, they document how the published model behaves on typical messages.
SANITY_SET = [
    ("URGENT: Your bank account has been suspended. Verify your details at http://bit.ly/x", True),
    ("Congratulations! You've won a $1000 gift card. Claim now by replying with your PIN.", True),
    (
        "Your package could not be delivered. Pay the $1.99 redelivery fee: http://track-pkg.co",
        True,
    ),
    ("Mom, I lost my phone, this is my new number. Can you send me $200 for rent?", True),
    ("IRS notice: you owe back taxes. Call 1-800-555-0199 now to avoid arrest.", True),
    ("Your Netflix payment failed. Update your card within 24h: netflx-billing.com", True),
    (
        "FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry",
        True,
    ),
    ("Hey, are we still on for lunch tomorrow at 12?", False),
    ("Reminder: your dentist appointment is on Monday at 9:30am.", False),
    ("Can you pick up milk on your way home?", False),
    ("Your Uber is arriving in 3 minutes.", False),
    ("Happy birthday! Have a great day, see you tonight", False),
]


def export(settings: Settings, output: Path) -> dict:
    import onnxruntime
    import peft
    import torch
    import transformers

    tokenizer, model = load_merged_model(settings)
    output.mkdir(parents=True, exist_ok=True)

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.wrapped(input_ids=input_ids, attention_mask=attention_mask).logits

    # Two inputs of different lengths, so the traced graph includes padding masks.
    dummy = tokenizer(
        ["export sample", "a slightly longer export sample"], padding=True, return_tensors="pt"
    )
    with torch.inference_mode():
        torch.onnx.export(
            LogitsOnly(model),
            (dummy["input_ids"], dummy["attention_mask"]),
            str(output / MODEL_FILE),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
        )
    tokenizer.backend_tokenizer.save(str(output / TOKENIZER_FILE))

    metadata = {
        "format_version": 1,
        "source": {
            "base_model": settings.base_model,
            "base_revision": settings.base_revision,
            "adapter_repo": settings.adapter_repo,
            "adapter_revision": settings.adapter_revision,
        },
        "precision": "fp32",
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "opset": OPSET,
        "exported_with": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "sha256": {name: sha256_of(output / name) for name in (MODEL_FILE, TOKENIZER_FILE)},
    }
    _write_metadata(output, metadata)
    return metadata


def check_parity(settings: Settings, output: Path, samples: list[str] = PARITY_SAMPLES) -> float:
    """Return the max absolute probability gap between PyTorch and ONNX."""
    reference = TransformersPredictor(settings)
    candidate = OnnxPredictor(
        settings.model_copy(update={"onnx_dir": str(output), "onnx_url": None})
    )
    gap = 0.0
    # Batched and one-by-one, so padding cannot hide a discrepancy.
    for batch in [samples, *([s] for s in samples)]:
        expected = reference.predict_proba(batch)
        actual = candidate.predict_proba(batch)
        gap = max(gap, *(abs(e - a) for e, a in zip(expected, actual, strict=True)))
    return gap


def sanity_report(settings: Settings, output: Path) -> dict:
    """Score SANITY_SET with the exported model and log a readable table."""
    predictor = OnnxPredictor(
        settings.model_copy(update={"onnx_dir": str(output), "onnx_url": None})
    )
    texts = [text for text, _ in SANITY_SET]
    scores = predictor.predict_proba(texts)
    rows = []
    for (text, expected), score in zip(SANITY_SET, scores, strict=True):
        flagged = score >= settings.threshold
        rows.append({"text": text, "expected_fraud": expected, "fraud_score": round(score, 4)})
        mark = "ok  " if flagged == expected else "MISS"
        logger.info("%s expected=%-5s score=%.4f  %s", mark, expected, score, text[:70])
    agreed = sum((r["fraud_score"] >= settings.threshold) == r["expected_fraud"] for r in rows)
    logger.info("Sanity set: %d/%d messages classified as expected", agreed, len(rows))
    return {"threshold": settings.threshold, "agreement": f"{agreed}/{len(rows)}", "rows": rows}


def _write_metadata(output: Path, metadata: dict) -> None:
    (output / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    settings = Settings()
    metadata = export(settings, args.output)
    gap = check_parity(settings, args.output)
    metadata["parity_max_abs_diff"] = gap
    metadata["sanity_report"] = sanity_report(settings, args.output)
    _write_metadata(args.output, metadata)

    size_mb = (args.output / MODEL_FILE).stat().st_size / 1e6
    logger.info("Exported %s (%.1f MB), max |p_torch - p_onnx| = %.2e", MODEL_FILE, size_mb, gap)
    if gap > args.atol:
        logger.error("Parity check failed: %.2e > atol %.2e", gap, args.atol)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
