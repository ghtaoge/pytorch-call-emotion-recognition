"""
test_analyzer —— 情感分析器与概率投影的单元测试

本模块测试 EmotionAnalyzer 的核心逻辑，包括：
- project_probabilities：将六类情感概率投影到四类并重新归一化
- iter_analysis：分析流程的状态事件与结果事件流
- 全静音音频的异常拒绝

使用 FakeRuntime 替代真实模型推理，确保测试不依赖 GPU 或模型权重。
"""

import numpy as np
import pytest

from app.analyzer import EmotionAnalyzer, project_probabilities
from app.audio import DecodedAudio
from app.config import Settings
from app.errors import AppError


class FakeRuntime:
    """伪造的模型运行时，用于替代真实推理引擎。

    - device 固定为 "cpu"，模拟 CPU 环境下的推理
    - predict 返回固定的六类概率分布，方便验证投影逻辑
    """

    device = "cpu"

    def predict(self, _waveform: np.ndarray) -> np.ndarray:
        # 返回一组预定义的六类概率：neutral、happy、anger、sad、fear、disgust
        # 其中 neutral=0.6 占主导，便于验证投影后 neutral 的占比
        return np.array([0.1, 0.05, 0.15, 0.6, 0.08, 0.02])


def test_projection_renormalizes_four_classes() -> None:
    """验证概率投影函数：将六类概率投影到四类后重新归一化。

    输入概率 [0.2, 0.1, 0.3, 0.2, 0.1, 0.1] 对应
    neutral=0.2, happy=0.1, anger=0.3, sad=0.2，以及 fear+disgust=0.2（被排除）。
    归一化后 anger 应为 0.3/0.8=0.375，happy 应为 0.1/0.8=0.125 不对，
    实际 anger=0.3, happy=0.1, sad=0.2, neutral=0.2 → 总和=0.8，
    重新归一化：anger=0.3/0.8=0.375, neutral=0.2/0.8=0.25。
    被排除概率 (fear+disgust) 应为 0.2。
    """
    result = project_probabilities(np.array([0.2, 0.1, 0.3, 0.2, 0.1, 0.1]))
    assert result.probabilities.anger == pytest.approx(0.25)  # anger=0.2/0.8=0.25
    assert result.probabilities.happy == pytest.approx(0.375)  # happy=0.3/0.8=0.375
    assert result.excluded_probability == pytest.approx(0.2)  # 被排除的 fear+disgust 合计


def test_analyzer_streams_progress_and_result() -> None:
    """验证 EmotionAnalyzer 的分析流程：先发送状态事件，再发送结果事件。

    使用 FakeRuntime 模拟推理，传入 7 秒的恒定幅值音频。
    iter_analysis 应返回多个事件：
    - 第一个事件为 status（预处理阶段）
    - 最后一个事件为 result（包含最终分析结果）
    - 结果中 neutral 概率应大于 0.6（因为 FakeRuntime 固定返回 neutral=0.6）
    """
    analyzer = EmotionAnalyzer(Settings(), FakeRuntime())  # type: ignore[arg-type]
    # 构造 7 秒 16000Hz 恒定幅值 0.2 的音频，确保非静音
    audio = DecodedAudio(np.full(7 * 16000, 0.2, dtype=np.float32), 16000)
    events = list(analyzer.iter_analysis(audio))
    assert events[0].type == "status"  # 首个事件应为预处理状态
    assert events[-1].type == "result"  # 最后一个事件应为分析结果
    assert events[-1].result.probabilities.neutral > 0.6  # type: ignore[union-attr]  # neutral 应为主导情感


def test_all_silent_audio_is_rejected() -> None:
    """验证全静音音频被拒绝：当音频所有采样值均为零时，应抛出 AppError。

    传入 1 秒的全零音频，分析器应检测到没有清晰人声，
    抛出匹配"未检测到清晰人声"的 AppError。
    """
    analyzer = EmotionAnalyzer(Settings(), FakeRuntime())  # type: ignore[arg-type]
    with pytest.raises(AppError, match="未检测到清晰人声"):
        # 传入全零（静音）音频，应触发异常
        list(analyzer.iter_analysis(DecodedAudio(np.zeros(16000, dtype=np.float32), 16000)))
