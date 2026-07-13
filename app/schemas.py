from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Emotion = Literal["neutral", "happy", "anger", "sad"]
ReliabilityReason = Literal[
    "OUTSIDE_FOUR_CLASS_SCOPE",
    "LOW_TOP_PROBABILITY",
    "SMALL_TOP_MARGIN",
    "LOW_VOICE_COVERAGE",
]
Probability = Annotated[float, Field(ge=0, le=1)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmotionProbabilities(ContractModel):
    neutral: Probability
    happy: Probability
    anger: Probability
    sad: Probability


class Reliability(ContractModel):
    level: Literal["high", "low"]
    reasons: list[ReliabilityReason]


class SegmentResult(ContractModel):
    index: NonNegativeInt
    start_seconds: NonNegativeFloat
    end_seconds: NonNegativeFloat
    is_silent: bool
    probabilities: EmotionProbabilities | None = None
    dominant_emotion: Emotion | None = None
    reliability: Reliability | None = None
    excluded_probability: Probability | None = None

    @model_validator(mode="after")
    def validate_segment_state(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")

        prediction_fields = (
            self.probabilities,
            self.dominant_emotion,
            self.reliability,
            self.excluded_probability,
        )
        # 静音段不得混入推理结果; 有声段必须提供完整结果, 避免产生歧义状态。
        if self.is_silent and any(value is not None for value in prediction_fields):
            raise ValueError("silent segments must not include prediction fields")
        if not self.is_silent and any(value is None for value in prediction_fields):
            raise ValueError("voiced segments must include all prediction fields")
        return self


class AnalysisResult(ContractModel):
    dominant_emotion: Emotion
    probabilities: EmotionProbabilities
    reliability: Reliability
    excluded_probability: Probability
    voiced_ratio: Probability
    duration_seconds: PositiveFloat
    device: str
    elapsed_ms: NonNegativeInt
    segments: list[SegmentResult]

    @field_validator("device")
    @classmethod
    def reject_blank_device(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("device must not be blank")
        return value


class HealthResponse(ContractModel):
    status: Literal["ok"]
    model_status: Literal["not_loaded", "loading", "loaded", "error"]
    device: str


class ProgressEvent(ContractModel):
    type: Literal["status", "progress"]
    current: NonNegativeInt
    total: NonNegativeInt
    message: str

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if self.current > self.total:
            raise ValueError("current must not exceed total")
        return self


class ResultEvent(ContractModel):
    type: Literal["result"] = "result"
    result: AnalysisResult


class PublicError(ContractModel):
    code: str
    message: str

    @field_validator("code", "message")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ErrorEvent(ContractModel):
    type: Literal["error"]
    error: PublicError
