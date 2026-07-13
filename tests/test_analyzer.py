import numpy as np
import pytest

from app.analyzer import EmotionAnalyzer, project_probabilities
from app.audio import DecodedAudio
from app.config import Settings
from app.errors import AppError


class FakeRuntime:
    device = "cpu"

    def predict(self, _waveform: np.ndarray) -> np.ndarray:
        return np.array([0.1, 0.05, 0.15, 0.6, 0.08, 0.02])


def test_projection_renormalizes_four_classes() -> None:
    result = project_probabilities(np.array([0.2, 0.1, 0.3, 0.2, 0.1, 0.1]))
    assert result.probabilities.anger == pytest.approx(0.25)
    assert result.probabilities.happy == pytest.approx(0.375)
    assert result.excluded_probability == pytest.approx(0.2)


def test_analyzer_streams_progress_and_result() -> None:
    analyzer = EmotionAnalyzer(Settings(), FakeRuntime())  # type: ignore[arg-type]
    audio = DecodedAudio(np.full(7 * 16000, 0.2, dtype=np.float32), 16000)
    events = list(analyzer.iter_analysis(audio))
    assert events[0].type == "status"
    assert events[-1].type == "result"
    assert events[-1].result.probabilities.neutral > 0.6  # type: ignore[union-attr]


def test_all_silent_audio_is_rejected() -> None:
    analyzer = EmotionAnalyzer(Settings(), FakeRuntime())  # type: ignore[arg-type]
    with pytest.raises(AppError, match="未检测到清晰人声"):
        list(analyzer.iter_analysis(DecodedAudio(np.zeros(16000, dtype=np.float32), 16000)))
