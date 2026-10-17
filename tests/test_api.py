from fastapi.testclient import TestClient

from sms_fraud.config import Settings
from sms_fraud.main import create_app
from sms_fraud.schemas import MAX_BATCH_SIZE, MAX_MESSAGE_CHARS
from tests.conftest import FakePredictor


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_model_version(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_version": "fake@0000000"}


def test_ready_is_503_before_model_is_loaded(settings: Settings) -> None:
    app = create_app(settings, predictor_factory=FakePredictor)
    # No context manager: the lifespan (model loading) never runs.
    client = TestClient(app)
    assert client.get("/ready").status_code == 503
    assert client.post("/v1/predict", json={"message": "hi"}).status_code == 503


def test_predict_fraud(client: TestClient) -> None:
    response = client.post("/v1/predict", json={"message": "You WIN a prize, click now"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "fraud"
    assert body["is_fraud"] is True
    assert body["fraud_score"] == 0.97
    assert body["threshold"] == 0.5
    assert body["model_version"] == "fake@0000000"
    assert body["latency_ms"] >= 0


def test_predict_legitimate(client: TestClient) -> None:
    body = client.post("/v1/predict", json={"message": "See you at 6pm"}).json()
    assert body["label"] == "legitimate"
    assert body["is_fraud"] is False


def test_threshold_is_configurable() -> None:
    app = create_app(Settings(_env_file=None, threshold=0.99), predictor_factory=FakePredictor)
    with TestClient(app) as client:
        body = client.post("/v1/predict", json={"message": "you win"}).json()
    assert body["is_fraud"] is False
    assert body["threshold"] == 0.99


def test_message_is_stripped(client: TestClient) -> None:
    client.post("/v1/predict", json={"message": "  hello  "})
    assert client.app.state.predictor.calls[-1] == ["hello"]


def test_rejects_empty_message(client: TestClient) -> None:
    assert client.post("/v1/predict", json={"message": "   "}).status_code == 422


def test_rejects_missing_message(client: TestClient) -> None:
    assert client.post("/v1/predict", json={}).status_code == 422


def test_rejects_too_long_message(client: TestClient) -> None:
    message = "a" * (MAX_MESSAGE_CHARS + 1)
    assert client.post("/v1/predict", json={"message": message}).status_code == 422


def test_batch_predict_keeps_order(client: TestClient) -> None:
    response = client.post("/v1/predict/batch", json={"messages": ["you win", "lunch?"]})
    assert response.status_code == 200
    labels = [p["label"] for p in response.json()["predictions"]]
    assert labels == ["fraud", "legitimate"]


def test_batch_rejects_empty_and_oversized(client: TestClient) -> None:
    assert client.post("/v1/predict/batch", json={"messages": []}).status_code == 422
    too_many = ["hi"] * (MAX_BATCH_SIZE + 1)
    assert client.post("/v1/predict/batch", json={"messages": too_many}).status_code == 422


def test_legacy_endpoint_is_gone(client: TestClient) -> None:
    assert client.post("/predict", json={"message": "hi"}).status_code == 404


def test_no_cors_headers_by_default(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_configured_origins() -> None:
    settings = Settings(_env_file=None, cors_origins=["https://app.example"])
    with TestClient(create_app(settings, predictor_factory=FakePredictor)) as client:
        allowed = client.get("/health", headers={"Origin": "https://app.example"})
        denied = client.get("/health", headers={"Origin": "https://evil.example"})
    assert allowed.headers["access-control-allow-origin"] == "https://app.example"
    assert "access-control-allow-origin" not in denied.headers


def test_docs_can_be_disabled() -> None:
    settings = Settings(_env_file=None, docs_enabled=False)
    with TestClient(create_app(settings, predictor_factory=FakePredictor)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_build_predictor_selects_backend(monkeypatch) -> None:
    from sms_fraud import main

    monkeypatch.setattr(main, "OnnxPredictor", lambda s: "onnx")
    monkeypatch.setattr(main, "TransformersPredictor", lambda s: "torch")
    assert main.build_predictor(Settings(_env_file=None)) == "onnx"
    assert main.build_predictor(Settings(_env_file=None, backend="transformers")) == "torch"
