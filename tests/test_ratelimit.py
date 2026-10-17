import pytest
from fastapi.testclient import TestClient

from sms_fraud.config import Settings
from sms_fraud.main import create_app
from sms_fraud.ratelimit import RateLimiter
from tests.conftest import FakePredictor


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_limiter_allows_up_to_the_limit_then_blocks() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)
    assert limiter.hit("a") is None
    assert limiter.hit("a") is None
    assert limiter.hit("a") == pytest.approx(60)
    # Other clients are not affected.
    assert limiter.hit("b") is None


def test_limiter_window_slides() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    assert limiter.hit("a") is None
    clock.now = 45
    assert limiter.hit("a") == pytest.approx(15)
    clock.now = 60
    assert limiter.hit("a") is None


def test_limiter_memory_is_bounded() -> None:
    limiter = RateLimiter(limit=1, max_clients=3, clock=FakeClock())
    for client in "abcd":
        limiter.hit(client)
    # "a" was evicted, so it is treated as a new client.
    assert limiter.hit("a") is None
    assert len(limiter._hits) == 3


def _client(**overrides: object) -> TestClient:
    settings = Settings(_env_file=None, **overrides)
    return TestClient(create_app(settings, predictor_factory=FakePredictor))


def test_api_returns_429_with_retry_after() -> None:
    with _client(rate_limit_per_minute=2) as client:
        for _ in range(2):
            assert client.post("/v1/predict", json={"message": "hi"}).status_code == 200
        response = client.post("/v1/predict/batch", json={"messages": ["hi"]})
    assert response.status_code == 429
    assert 1 <= int(response.headers["retry-after"]) <= 60


def test_health_endpoints_are_not_rate_limited() -> None:
    with _client(rate_limit_per_minute=1) as client:
        client.post("/v1/predict", json={"message": "hi"})
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200


def test_rate_limit_can_be_disabled() -> None:
    with _client(rate_limit_per_minute=0) as client:
        codes = {client.post("/v1/predict", json={"message": "hi"}).status_code for _ in range(5)}
    assert codes == {200}


def test_proxy_headers_identify_clients_only_when_trusted() -> None:
    with _client(rate_limit_per_minute=1, trust_proxy_headers=True) as client:
        assert client.post("/v1/predict", json={"message": "a"}, headers={"x-real-ip": "1.1.1.1"})
        second = client.post("/v1/predict", json={"message": "a"}, headers={"x-real-ip": "2.2.2.2"})
        assert second.status_code == 200
    with _client(rate_limit_per_minute=1, trust_proxy_headers=False) as client:
        client.post("/v1/predict", json={"message": "a"}, headers={"x-real-ip": "1.1.1.1"})
        spoofed = client.post(
            "/v1/predict", json={"message": "a"}, headers={"x-real-ip": "2.2.2.2"}
        )
        assert spoofed.status_code == 429


def test_proxy_headers_are_trusted_by_default_on_vercel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERCEL", raising=False)
    assert Settings(_env_file=None).resolved_trust_proxy_headers() is False
    monkeypatch.setenv("VERCEL", "1")
    assert Settings(_env_file=None).resolved_trust_proxy_headers() is True
