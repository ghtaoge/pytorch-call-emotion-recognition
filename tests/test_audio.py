"""
test_audio —— 音频波形处理的单元测试

本模块测试 app.audio 中两个核心函数：
- normalize_waveform：立体声下混与重采样，将任意音频转为单声道 16kHz
- segment_waveform：滑动窗口分段，并标记静音段

同时验证对异常输入（非有限采样值）的拒绝。
"""

import numpy as np
import pytest

from app.audio import normalize_waveform, segment_waveform
from app.config import Settings
from app.errors import AppError


def test_normalize_downmixes_and_resamples() -> None:
    """验证 normalize_waveform 的下混与重采样功能。

    构造一个立体声信号：左声道全为 1.0，右声道全为 0.0，采样率 8000Hz。
    下混后应为左右均值 0.5；重采样到 16000Hz 后长度应翻倍为 16000 个采样点。
    均值应近似 0.5（允许 ±0.02 的误差，因重采样插值可能略有偏差）。
    """
    # 构造立体声信号：左声道=1.0，右声道=0.0，8000 采样点
    stereo = np.stack([np.ones(8000), np.zeros(8000)]).astype(np.float32)
    result = normalize_waveform(stereo, 8000, 16000)
    assert result.shape == (16000,)  # 重采样后长度应为 16000
    assert result.dtype == np.float32  # 数据类型应为 float32
    assert np.mean(result) == pytest.approx(0.5, abs=0.02)  # 下混均值应近似 0.5


def test_segment_keeps_tail_and_marks_silence() -> None:
    """验证 segment_waveform 的滑动窗口分段与静音标记。

    构造 8 秒音频：前 6 秒有信号（幅值 0.2），后 2 秒静音（幅值 0）。
    使用 window=6s, hop=5s 的滑动窗口分段：
    - 第 0 段：[0, 6] 秒，有信号，is_silent=False
    - 第 1 段：[5, 8] 秒，跨越静音区，is_silent 应为 True
    第 1 段的实际采样数应为 3*16000=48000（从 5 秒到 8 秒，共 3 秒）。
    """
    # 构造 8 秒音频：前 6 秒有信号，后 2 秒静音
    waveform = np.concatenate(
        [np.full(6 * 16000, 0.2, dtype=np.float32), np.zeros(2 * 16000, dtype=np.float32)]
    )
    # 使用 window=6s, hop=5s 的分段参数
    segments = segment_waveform(waveform, Settings(window_seconds=6, hop_seconds=5))
    # 验证分段的时间区间
    assert [(item.start_seconds, item.end_seconds) for item in segments] == [(0.0, 6.0), (5.0, 8.0)]
    assert not segments[0].is_silent  # 第一段应有信号（非静音）
    assert segments[1].sample_count == 3 * 16000  # 第二段长度为 3 秒的采样数


def test_normalize_rejects_nonfinite() -> None:
    """验证 normalize_waveform 拒绝包含非有限值的波形。

    传入包含 NaN 的波形时，应抛出 AppError，错误信息匹配"无效采样值"。
    这是防止模型推理出现异常的必要保护。
    """
    with pytest.raises(AppError, match="无效采样值"):
        # 传入包含 NaN 的波形，应被拒绝
        normalize_waveform(np.array([0.0, np.nan], dtype=np.float32), 16000)
