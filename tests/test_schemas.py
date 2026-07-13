import json

import pytest
from pydantic import ValidationError

from app.errors import AppError
from app.schemas import (
    AnalysisResult,
    EmotionProbabilities,
    ErrorEvent,
    ProgressEvent,
    PublicError,
    Reliability,
    ResultEvent,
    SegmentResult,
)


def probabilities() -> EmotionProbabilities:
    return EmotionProbabilities(neutral=0.7, happy=0.1, anger=0.1, sad=0.1)


def reliability() -> Reliability:
    return Reliability(level="high", reasons=[])


def voiced_segment(**overrides: object) -> SegmentResult:
    values: dict[str, object] = {
        "index": 0,
        "start_seconds": 0.0,
        "end_seconds": 4.0,
        "is_silent": False,
        "probabilities": probabilities(),
        "dominant_emotion": "neutral",
        "reliability": reliability(),
        "excluded_probability": 0.08,
    }
    values.update(overrides)
    return SegmentResult(**values)  # type: ignore[arg-type]


def analysis_result() -> AnalysisResult:
    return AnalysisResult(
        dominant_emotion="neutral",
        probabilities=probabilities(),
        reliability=reliability(),
        excluded_probability=0.08,
        voiced_ratio=0.9,
        duration_seconds=4.0,
        device="cpu",
        elapsed_ms=120,
        segments=[voiced_segment()],
    )


def test_analysis_result_serializes_four_probability_contract() -> None:
    result = analysis_result()

    payload = result.model_dump(mode="json")

    assert payload["dominant_emotion"] == "neutral"
    assert set(payload["probabilities"]) == {"neutral", "happy", "anger", "sad"}
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0)
    assert json.loads(result.model_dump_json()) == payload


@pytest.mark.parametrize(
    ("field", "value"),
    [("neutral", -0.01), ("happy", 1.01)],
)
def test_probability_bounds_reject_invalid_values(field: str, value: float) -> None:
    values = {"neutral": 0.7, "happy": 0.1, "anger": 0.1, "sad": 0.1}
    values[field] = value

    with pytest.raises(ValidationError):
        EmotionProbabilities(**values)  # type: ignore[arg-type]


def test_silent_segment_requires_prediction_fields_to_be_none() -> None:
    segment = SegmentResult(
        index=0,
        start_seconds=0.0,
        end_seconds=4.0,
        is_silent=True,
        probabilities=None,
        dominant_emotion=None,
        reliability=None,
        excluded_probability=None,
    )

    assert segment.is_silent is True
    assert segment.probabilities is None

    with pytest.raises(ValidationError):
        voiced_segment(is_silent=True)


@pytest.mark.parametrize(
    "missing_field",
    ["probabilities", "dominant_emotion", "reliability", "excluded_probability"],
)
def test_voiced_segment_requires_every_prediction_field(missing_field: str) -> None:
    with pytest.raises(ValidationError):
        voiced_segment(**{missing_field: None})


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds"),
    [(2.0, 2.0), (2.0, 1.0)],
)
def test_segment_end_must_be_after_start(start_seconds: float, end_seconds: float) -> None:
    with pytest.raises(ValidationError):
        voiced_segment(start_seconds=start_seconds, end_seconds=end_seconds)


def test_progress_rejects_current_greater_than_total() -> None:
    with pytest.raises(ValidationError):
        ProgressEvent(type="progress", current=2, total=1, message="处理中")


def test_result_event_has_stable_default_type() -> None:
    event = ResultEvent(result=analysis_result())

    assert event.type == "result"
    assert json.loads(event.model_dump_json())["type"] == "result"


def test_schemas_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EmotionProbabilities(
            neutral=0.7,
            happy=0.1,
            anger=0.1,
            sad=0.1,
            fear=0.0,
        )


def test_public_error_rejects_blank_fields() -> None:
    with pytest.raises(ValidationError):
        PublicError(code=" ", message="safe")
    with pytest.raises(ValidationError):
        PublicError(code="AUDIO_INVALID", message="\t")


def test_error_event_serializes_only_public_error() -> None:
    event = ErrorEvent(
        type="error",
        error=PublicError(code="AUDIO_INVALID", message="音频无效"),
    )

    assert event.model_dump(mode="json") == {
        "type": "error",
        "error": {"code": "AUDIO_INVALID", "message": "音频无效"},
    }


def test_app_error_exposes_only_safe_public_fields() -> None:
    error = AppError("AUDIO_INVALID", "音频无效", status_code=422)

    assert str(error) == "音频无效"
    assert error.args == ("音频无效",)
    assert vars(error) == {
        "code": "AUDIO_INVALID",
        "public_message": "音频无效",
        "status_code": 422,
    }
