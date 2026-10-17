"""FastAPI application."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from sms_fraud import __version__
from sms_fraud.config import Settings, get_settings
from sms_fraud.onnx_predictor import OnnxPredictor
from sms_fraud.predictor import Predictor, TransformersPredictor
from sms_fraud.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    Label,
    Prediction,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

logger = logging.getLogger("sms_fraud")

PredictorFactory = Callable[[Settings], Predictor]

router = APIRouter()


def build_predictor(settings: Settings) -> Predictor:
    """Instantiate the configured inference backend."""
    if settings.backend == "transformers":
        return TransformersPredictor(settings)
    return OnnxPredictor(settings)


def _get_predictor(request: Request) -> Predictor:
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return predictor


def _run(request: Request, messages: list[str]) -> tuple[list[Prediction], float, float, str]:
    predictor = _get_predictor(request)
    threshold: float = request.app.state.settings.threshold
    start = time.perf_counter()
    scores = predictor.predict_proba(messages)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    predictions = [
        Prediction(
            label=Label.FRAUD if score >= threshold else Label.LEGITIMATE,
            is_fraud=score >= threshold,
            fraud_score=round(score, 6),
        )
        for score in scores
    ]
    # Message contents are deliberately never logged.
    logger.info(
        "predict count=%d fraud=%d latency_ms=%.2f",
        len(predictions),
        sum(p.is_fraud for p in predictions),
        latency_ms,
    )
    return predictions, threshold, latency_ms, predictor.version


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe: the process is up."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(request: Request) -> ReadyResponse:
    """Readiness probe: the model is loaded and can serve predictions."""
    return ReadyResponse(model_version=_get_predictor(request).version)


# Endpoints are plain `def` on purpose: FastAPI runs them in a worker thread,
# so CPU-bound inference does not block the event loop.
@router.post("/v1/predict", response_model=PredictResponse, tags=["prediction"])
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Classify a single SMS message."""
    predictions, threshold, latency_ms, version = _run(request, [body.message])
    return PredictResponse(
        **predictions[0].model_dump(),
        threshold=threshold,
        model_version=version,
        latency_ms=latency_ms,
    )


@router.post("/v1/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """Classify up to 32 SMS messages in one call."""
    predictions, threshold, latency_ms, version = _run(request, body.messages)
    return BatchPredictResponse(
        predictions=predictions,
        threshold=threshold,
        model_version=version,
        latency_ms=latency_ms,
    )


def create_app(
    settings: Settings | None = None,
    predictor_factory: PredictorFactory = build_predictor,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = await run_in_threadpool(predictor_factory, settings)
        logger.info("Model ready: %s", app.state.predictor.version)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="SMS Fraud Detection API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.predictor = None

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    app.include_router(router)
    if settings.frontend_dir:
        # Mounted last so that API routes take precedence.
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


app = create_app()
