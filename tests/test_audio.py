import numpy as np
import pytest

from app.audio import normalize_waveform, segment_waveform
from app.config import Settings
from app.errors import AppError


def test_normalize_downmixes_and_resamples() -> None:
    stereo = np.stack([np.ones(8000), np.zeros(8000)]).astype(np.float32)
    result = normalize_waveform(stereo, 8000, 16000)
    assert result.shape == (16000,)
    assert result.dtype == np.float32
    assert np.mean(result) == pytest.approx(0.5, abs=0.02)


def test_segment_keeps_tail_and_marks_silence() -> None:
    waveform = np.concatenate(
        [np.full(6 * 16000, 0.2, dtype=np.float32), np.zeros(2 * 16000, dtype=np.float32)]
    )
    segments = segment_waveform(waveform, Settings(window_seconds=6, hop_seconds=5))
    assert [(item.start_seconds, item.end_seconds) for item in segments] == [(0.0, 6.0), (5.0, 8.0)]
    assert not segments[0].is_silent
    assert segments[1].sample_count == 3 * 16000


def test_normalize_rejects_nonfinite() -> None:
    with pytest.raises(AppError, match="无效采样值"):
        normalize_waveform(np.array([0.0, np.nan], dtype=np.float32), 16000)
