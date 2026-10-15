"""Request and response models of the public API (v1)."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

MAX_MESSAGE_CHARS = 2000
MAX_BATCH_SIZE = 32

Message = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS)
]


class Label(StrEnum):
    FRAUD = "fraud"
    LEGITIMATE = "legitimate"


class PredictRequest(BaseModel):
    message: Message = Field(examples=["URGENT: your account is locked, verify at http://bit.ly/x"])


class BatchPredictRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    label: Label
    is_fraud: bool
    fraud_score: float = Field(ge=0.0, le=1.0, description="Probability that the message is fraud")


class PredictionMeta(BaseModel):
    threshold: float
    model_version: str
    latency_ms: float


class PredictResponse(Prediction, PredictionMeta):
    pass


class BatchPredictResponse(PredictionMeta):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"
    model_version: str
