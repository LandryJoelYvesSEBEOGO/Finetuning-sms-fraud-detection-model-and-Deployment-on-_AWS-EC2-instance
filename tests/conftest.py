from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient

from sms_fraud.config import Settings
from sms_fraud.main import create_app


class FakePredictor:
    """Flags any message containing 'win' with a high fraud probability."""

    version = "fake@0000000"

    def __init__(self, settings: Settings) -> None:
        self.calls: list[list[str]] = []

    def predict_proba(self, messages: Sequence[str]) -> list[float]:
        self.calls.append(list(messages))
        return [0.97 if "win" in m.lower() else 0.03 for m in messages]


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, threshold=0.5)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings, predictor_factory=FakePredictor)
    with TestClient(app) as test_client:
        yield test_client
