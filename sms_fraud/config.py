"""Runtime configuration, read from environment variables prefixed with ``SMS_FRAUD_``."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMS_FRAUD_", env_file=".env", extra="ignore")

    # Model source. The adapter revision is pinned so that a push to the Hub
    # repository cannot silently change the model served in production.
    base_model: str = "distilbert-base-uncased"
    base_revision: str | None = None
    adapter_repo: str = "MrNWARMeilleur/sms_fraud_detection"
    adapter_revision: str | None = "2cf2ba00cdc8b1491094164ac104dfc070144a18"

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
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
