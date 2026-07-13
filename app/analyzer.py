from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.audio import DecodedAudio, segment_waveform
from app.config import Settings
from app.errors import AppError
from app.model import EmotionModelRuntime
from app.schemas import (
    AnalysisResult,
    Emotion,
    EmotionProbabilities,
    ProgressEvent,
    Reliability,
    ReliabilityReason,
    ResultEvent,
    SegmentResult,
)

TARGET_INDICES = {"anger": 0, "happy": 2, "neutral": 3, "sad": 4}


@dataclass(frozen=True, slots=True)
class ProjectedPrediction:
    probabilities: EmotionProbabilities
    excluded_probability: float
    dominant_emotion: Emotion
    reliability: Reliability


def project_probabilities(raw: np.ndarray) -> ProjectedPrediction:
    """投影四个目标类别，同时保留被排除类别的质量警示信号。"""
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise AppError("INVALID_MODEL_OUTPUT", "模型输出格式异常", 500)
    targets = {name: float(values[index]) for name, index in TARGET_INDICES.items()}
    total = sum(targets.values())
    if total <= 1e-12:
        raise AppError("OUTSIDE_TARGET_SCOPE", "当前语音不适合四分类判断", 422)
    normalized = {name: value / total for name, value in targets.items()}
    ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    excluded = float(np.clip(values[1] + values[5], 0.0, 1.0))
    reasons: list[ReliabilityReason] = []
    if excluded > 0.35:
        reasons.append("OUTSIDE_FOUR_CLASS_SCOPE")
    if ordered[0][1] < 0.45:
        reasons.append("LOW_TOP_PROBABILITY")
    if ordered[0][1] - ordered[1][1] < 0.12:
        reasons.append("SMALL_TOP_MARGIN")
    probabilities = EmotionProbabilities(
        neutral=normalized["neutral"],
        happy=normalized["happy"],
        anger=normalized["anger"],
        sad=normalized["sad"],
    )
    return ProjectedPrediction(
        probabilities=probabilities,
        excluded_probability=excluded,
        dominant_emotion=ordered[0][0],  # type: ignore[arg-type]
        reliability=Reliability(level="low" if reasons else "high", reasons=reasons),
    )


class EmotionAnalyzer:
    """协调分段推理与聚合，不关心 HTTP 或页面呈现。"""

    def __init__(self, settings: Settings, runtime: EmotionModelRuntime) -> None:
        self.settings = settings
        self.runtime = runtime

    def iter_analysis(self, audio: DecodedAudio):
        """逐段产出进度，最后产出结果，便于 HTTP 层直接流式传输。"""
        started = time.perf_counter()
        segments = segment_waveform(audio.waveform, self.settings)
        if not segments or all(item.is_silent for item in segments):
            raise AppError("NO_VOICE", "未检测到清晰人声，请更换音频后重试", 422)
        yield ProgressEvent(type="status", current=0, total=len(segments), message="模型准备中")

        results: list[SegmentResult] = []
        weighted: list[tuple[ProjectedPrediction, float]] = []
        voiced_samples = 0
        total_samples = sum(item.sample_count for item in segments)
        for current, segment in enumerate(segments, start=1):
            if segment.is_silent:
                results.append(
                    SegmentResult(
                        index=segment.index,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        is_silent=True,
                    )
                )
            else:
                prediction = project_probabilities(self.runtime.predict(segment.waveform))
                results.append(
                    SegmentResult(
                        index=segment.index,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        is_silent=False,
                        probabilities=prediction.probabilities,
                        dominant_emotion=prediction.dominant_emotion,
                        reliability=prediction.reliability,
                        excluded_probability=prediction.excluded_probability,
                    )
                )
                weight = segment.sample_count * float(np.clip(segment.rms, 0.02, 0.30))
                weighted.append((prediction, weight))
                voiced_samples += segment.sample_count
            yield ProgressEvent(
                type="progress",
                current=current,
                total=len(segments),
                message=f"正在分析第 {current}/{len(segments)} 段",
            )

        # 能量被裁剪到稳定区间，避免一次爆音压过整段通话的其他窗口。
        weights = np.array([weight for _, weight in weighted], dtype=np.float64)
        weights /= weights.sum()
        target_order = ("neutral", "happy", "anger", "sad")
        aggregate_values = {
            name: float(
                sum(
                    weight * getattr(prediction.probabilities, name)
                    for weight, (prediction, _) in zip(weights, weighted, strict=True)
                )
            )
            for name in target_order
        }
        excluded = float(
            sum(
                weight * prediction.excluded_probability
                for weight, (prediction, _) in zip(weights, weighted, strict=True)
            )
        )
        # 将聚合后的四类概率放回六类槽位，复用同一套可靠性阈值。
        raw = np.array(
            [
                aggregate_values["anger"] * (1 - excluded),
                excluded / 2,
                aggregate_values["happy"] * (1 - excluded),
                aggregate_values["neutral"] * (1 - excluded),
                aggregate_values["sad"] * (1 - excluded),
                excluded / 2,
            ]
        )
        overall = project_probabilities(raw)
        voiced_ratio = min(1.0, voiced_samples / max(total_samples, 1))
        reasons = list(overall.reliability.reasons)
        if voiced_ratio < 0.3:
            reasons.append("LOW_VOICE_COVERAGE")
        reliability = Reliability(level="low" if reasons else "high", reasons=reasons)
        result = AnalysisResult(
            dominant_emotion=overall.dominant_emotion,
            probabilities=overall.probabilities,
            reliability=reliability,
            excluded_probability=excluded,
            voiced_ratio=voiced_ratio,
            duration_seconds=audio.duration_seconds,
            device=self.runtime.device,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            segments=results,
        )
        yield ResultEvent(result=result)
