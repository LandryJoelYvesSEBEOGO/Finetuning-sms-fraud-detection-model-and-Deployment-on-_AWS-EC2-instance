"""Runtime configuration, read from environment variables prefixed with ``SMS_FRAUD_``."""

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# GitHub release populated by the "Export ONNX model" workflow, tagged after
# the adapter revision it was converted from.
DEFAULT_ONNX_URL = (
    "https://github.com/LandryJoelYvesSEBEOGO/"
    "Finetuning-sms-fraud-detection-model-and-Deployment-on-_AWS-EC2-instance"
    "/releases/download/model-onnx-2cf2ba0"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMS_FRAUD_", env_file=".env", extra="ignore")

    # Model source. The adapter revision is pinned so that a push to the Hub
    # repository cannot silently change the model served in production.
    base_model: str = "distilbert-base-uncased"
    base_revision: str | None = None
    adapter_repo: str = "MrNWARMeilleur/sms_fraud_detection"
    adapter_revision: str | None = "2cf2ba00cdc8b1491094164ac104dfc070144a18"

    # Inference backend: "onnx" (production, lightweight) or "transformers"
    # (PyTorch reference, requires requirements-ml.txt).
    backend: Literal["onnx", "transformers"] = "onnx"
    # Directory holding model.onnx, tokenizer.json and metadata.json.
    # Defaults to artifacts/onnx, or to /tmp on Vercel (the only writable path).
    onnx_dir: str | None = None
    # Base URL to download the ONNX artifacts from when they are missing locally.
    onnx_url: str | None = DEFAULT_ONNX_URL
    # Optional expected SHA-256 of model.onnx, enforced after download.
    onnx_sha256: str | None = None

    # Index of the "fraud" class in the classifier output (LABEL_1).
    fraud_label_index: int = Field(1, ge=0, le=1)
    # Minimum fraud probability for a message to be flagged as fraud.
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    # Messages longer than this are truncated (DistilBERT supports up to 512).
    max_tokens: int = Field(512, ge=8, le=512)

    # Allowed browser origins. Empty means no CORS headers (same-origin only).
    # Example: SMS_FRAUD_CORS_ORIGINS='["https://my-app.vercel.app"]'
    cors_origins: list[str] = []
    docs_enabled: bool = True
    # Local development only: also serve the static frontend from this
    # directory (e.g. "public"). On Vercel, public/ is served by the CDN.
    frontend_dir: str | None = None
    log_level: str = "INFO"

    # Prediction requests allowed per client and per minute (0 disables).
    rate_limit_per_minute: int = Field(60, ge=0)
    # Trust X-Real-IP / X-Forwarded-For to identify clients. Defaults to true
    # on Vercel, whose edge sets them; never enable without a trusted proxy.
    trust_proxy_headers: bool | None = None

    def resolved_trust_proxy_headers(self) -> bool:
        if self.trust_proxy_headers is not None:
            return self.trust_proxy_headers
        return bool(os.environ.get("VERCEL"))

    def resolved_onnx_dir(self) -> Path:
        if self.onnx_dir:
            return Path(self.onnx_dir)
        if os.environ.get("VERCEL"):
            return Path(tempfile.gettempdir()) / "sms-fraud-onnx"
        return Path("artifacts/onnx")


@lru_cache
def get_settings() -> Settings:
    return Settings()
