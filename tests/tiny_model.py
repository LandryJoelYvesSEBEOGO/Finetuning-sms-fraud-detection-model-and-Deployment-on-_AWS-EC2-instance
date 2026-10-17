"""Build a tiny, randomly initialised DistilBERT + LoRA adapter on disk.

The base model is saved *without* a classification head, like
``distilbert-base-uncased`` on the Hub: the head must come from the adapter.
"""

from pathlib import Path

from sms_fraud.config import Settings

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "you", "win", "a", "prize", "call", "now"]


def build_tiny_model(root: Path) -> tuple[Path, Path]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertModel,
        DistilBertTokenizerFast,
    )

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


def tiny_settings(model_dirs: tuple[Path, Path], **overrides: object) -> Settings:
    base_dir, adapter_dir = model_dirs
    return Settings(
        _env_file=None,
        base_model=str(base_dir),
        adapter_repo=str(adapter_dir),
        adapter_revision=None,
        **overrides,
    )
