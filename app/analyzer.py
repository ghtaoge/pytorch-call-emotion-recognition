"""
app.analyzer — 情绪分析协调模块

本模块负责协调音频分段推理与结果聚合，是整个分析流程的核心调度器。
不关心 HTTP 或页面呈现，专注于业务逻辑。

核心流程：
1. 将完整音频按滑动窗口切分为多个片段
2. 对每个有声片段执行模型推理，得到六类原始概率
3. 将六类概率投影为四类目标概率（anger, happy, neutral, sad）
   同时保留被排除类别（fear, surprise）的概率作为可靠性警示信号
4. 对所有有声片段的结果进行加权聚合（权重 = 采样点数 × RMS 能量）
5. 将聚合结果再次投影，得到整体情绪判定与可靠性评估

关键设计决策：
- 四类投影而非直接使用六类：通话场景中 fear 和 surprise 出现极少且不稳定，
  将其排除后重新归一化可提高四类的区分度
- 可靠性评估：基于被排除类别概率、顶部概率值、顶部概率差距三个指标
- 加权聚合：权重 = 采样点数 × RMS 能量（裁剪到 0.02~0.30 区间），
  防止一次爆音压过整段通话的其他窗口
- 流式输出：iter_analysis 是生成器，逐段产出进度事件，最后产出结果事件，
  便于 HTTP 层直接以 NDJSON 流式传输给前端
"""

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

# 目标四类情绪及其在原始六类数组中的索引位置
# fear(1) 和 surprise(5) 被排除，因为通话场景中这两类极少出现且模型对其不稳定
TARGET_INDICES = {"anger": 0, "happy": 2, "neutral": 3, "sad": 4}


@dataclass(frozen=True, slots=True)
class ProjectedPrediction:
    """
    投影后的预测结果 — 将六类概率压缩为四类并附带可靠性评估

    frozen=True + slots=True 保证不可变性与内存效率：
    - frozen=True: 实例创建后不可修改字段，防止推理结果被意外篡改
    - slots=True: 使用 __slots__ 代替 __dict__，减少约 40% 内存占用

    字段说明：
    - probabilities    : 四类目标概率（已归一化，总和为 1）
    - excluded_probability : 被排除的两类（fear + surprise）概率之和，作为可靠性警示信号
    - dominant_emotion : 概率最高的目标情绪类别
    - reliability      : 可靠性评估（high/low + 原因列表）
    """
    probabilities: EmotionProbabilities
    excluded_probability: float
    dominant_emotion: Emotion
    reliability: Reliability


def project_probabilities(raw: np.ndarray) -> ProjectedPrediction:
    """
    将六类原始概率投影为四类目标概率，同时保留被排除类别的质量警示信号

    此函数是六类→四类映射的核心逻辑，在以下两个阶段被调用：
    1. 单段推理后：将单个片段的六类概率投影为四类
    2. 整体聚合后：将加权聚合的六类概率再次投影为四类

    算法步骤：
    1. 输入校验：确保原始概率为六维有限数组
    2. 提取目标四类概率（按 TARGET_INDICES 映射）
    3. 重新归一化：除以四类总和，使概率重新分布到 [0, 1] 区间
       若总和接近零（≤ 1e-12），说明六类概率均极低，判定为不适合四分类
    4. 按概率降序排序，确定主导情绪与次主导情绪
    5. 计算被排除类别概率之和（fear + surprise）
    6. 可靠性评估：基于三个阈值判定

    可靠性评估阈值及其设计依据：
    - excluded_probability > 0.35 → "OUTSIDE_FOUR_CLASS_SCOPE"
      被排除类别占比过高（> 35%），说明此语音更接近 fear/surprise 而非目标四类，
      四分类结果的可信度较低。阈值 0.35 在经验测试中区分了明显的非目标语音。
    - 顶部概率 < 0.45 → "LOW_TOP_PROBABILITY"
      即使归一化后，最高类别的概率仍不到 45%，说明各类别分布过于均匀，
      缺乏明确的情绪倾向。阈值 0.45 保证至少有轻微倾向才能标记为 high。
    - 顶部与次顶部概率差 < 0.12 → "SMALL_TOP_MARGIN"
      前两名概率差距过小（< 12%），即使顶部概率较高，也难以确信主导情绪。
      阈值 0.12 在经验测试中区分了清晰倾向与模糊倾向。

    参数：
        raw — 模型输出的六类原始概率，形状 (6,)
              顺序为 [anger, fear, happy, neutral, sad, surprise]

    返回：
        ProjectedPrediction — 包含四类概率、排除概率、主导情绪和可靠性评估

    异常：
        AppError("INVALID_MODEL_OUTPUT", 500) — 输入格式异常
        AppError("OUTSIDE_TARGET_SCOPE", 422) — 四类概率总和接近零
    """
    values = np.asarray(raw, dtype=np.float64)
    # 输入校验：确保为六维有限数组
    if values.shape != (6,) or not np.isfinite(values).all():
        raise AppError("INVALID_MODEL_OUTPUT", "模型输出格式异常", 500)
    # 提取目标四类概率（按硬编码索引映射，不依赖模型配置中的标签名）
    targets = {name: float(values[index]) for name, index in TARGET_INDICES.items()}
    # 四类概率总和：用于重新归一化
    total = sum(targets.values())
    # 极低总和校验：若四类概率之和接近零，说明此语音不适合四分类判断
    # 1e-12 阈值避免了浮点精度导致的误判，同时排除了所有实际无效的情况
    if total <= 1e-12:
        raise AppError("OUTSIDE_TARGET_SCOPE", "当前语音不适合四分类判断", 422)
    # 重新归一化：将四类概率除以总和，使概率分布重新回到 [0, 1] 区间
    normalized = {name: value / total for name, value in targets.items()}
    # 按概率降序排序，用于确定主导情绪与次主导情绪
    ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    # 被排除类别概率之和（fear + surprise），作为可靠性警示信号
    # np.clip 保证在 [0, 1] 区间内，防止浮点误差导致超出范围
    excluded = float(np.clip(values[1] + values[5], 0.0, 1.0))
    # 可靠性评估：基于三个阈值判定
    reasons: list[ReliabilityReason] = []
    # 阈值 1：被排除类别占比 > 35%，说明此语音更接近 fear/surprise
    if excluded > 0.35:
        reasons.append("OUTSIDE_FOUR_CLASS_SCOPE")
    # 阈值 2：顶部概率 < 45%，说明各类别分布过于均匀，缺乏明确倾向
    if ordered[0][1] < 0.45:
        reasons.append("LOW_TOP_PROBABILITY")
    # 阈值 3：前两名概率差距 < 12%，说明难以确信主导情绪
    if ordered[0][1] - ordered[1][1] < 0.12:
        reasons.append("SMALL_TOP_MARGIN")
    # 构建四类概率模型（字段名固定，保证 API 契约稳定）
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
        # 若有任何可靠性原因，标记为 low；否则标记为 high
        reliability=Reliability(level="low" if reasons else "high", reasons=reasons),
    )


class EmotionAnalyzer:
    """
    情绪分析协调器 — 协调分段推理与聚合，不关心 HTTP 或页面呈现

    本类是整个分析流程的核心调度器，负责：
    1. 调用 segment_waveform 将完整音频切分为滑动窗口片段
    2. 对每个有声片段调用 runtime.predict() 执行模型推理
    3. 对推理结果调用 project_probabilities() 执行六类→四类投影
    4. 对所有有声片段的结果进行加权聚合
    5. 对聚合结果再次投影，得到整体情绪判定

    设计原则：
    - 本类不关心 HTTP 传输、页面呈现等细节，仅输出结构化事件
    - iter_analysis 是生成器函数，产出 ProgressEvent 和 ResultEvent
    - HTTP 层直接迭代此生成器，逐行以 NDJSON 格式传输给前端
    """

    def __init__(self, settings: Settings, runtime: EmotionModelRuntime) -> None:
        """
        初始化分析器

        参数：
            settings — 全局配置对象，提供窗口长度、步长等分段参数
            runtime — 模型运行时管理器，提供延迟加载与推理能力
        """
        self.settings = settings
        self.runtime = runtime

    def iter_analysis(self, audio: DecodedAudio):
        """
        逐段产出进度，最后产出结果，便于 HTTP 层直接流式传输

        此方法是生成器函数，产出两种事件：
        - ProgressEvent: 进度事件，包含当前段编号与总段数
        - ResultEvent: 结果事件，包含完整的分析结果

        算法流程：
        1. 切分音频为滑动窗口片段
        2. 校验：若无有声片段，抛出 AppError("NO_VOICE")
        3. 预热：产出初始进度事件，提示模型准备中
        4. 逐段推理：
           - 静音段：跳过推理，仅记录基本信息
           - 有声段：推理 → 投影 → 记录结果 + 累积加权预测
        5. 加权聚合：
           - 权重 = 采样点数 × RMS 能量（裁剪到 0.02~0.30）
           - RMS 裁剪防止一次爆音（rms > 0.30）压过整段通话的其他窗口
           - RMS 裁剪防止极低能量段（rms < 0.02）获得不合理权重
           - 归一化权重使总和为 1
        6. 将聚合后的四类概率放回六类槽位，复用同一套可靠性阈值
        7. 产出 ResultEvent

        参数：
            audio — 已解码的音频对象，包含波形数组与采样率

        产出：
            ProgressEvent — 进度事件（每段推理后产出一次）
            ResultEvent — 最终结果事件（所有段推理完成后产出一次）

        异常：
            AppError("NO_VOICE", 422) — 无有声片段可分析
            AppError(其他) — 推理过程中的各类异常
        """
        # 记录开始时间，用于计算总耗时（elapsed_ms）
        started = time.perf_counter()
        # 按滑动窗口切分音频为多个片段
        segments = segment_waveform(audio.waveform, self.settings)
        # 校验：确保至少有一个有声片段可分析
        if not segments or all(item.is_silent for item in segments):
            raise AppError("NO_VOICE", "未检测到清晰人声，请更换音频后重试", 422)
        # 产出初始进度事件，提示前端模型正在准备
        yield ProgressEvent(type="status", current=0, total=len(segments), message="模型准备中")

        # 结果累积容器
        results: list[SegmentResult] = []
        # 加权预测列表：每个元素为 (预测结果, 权重)，用于最终聚合
        weighted: list[tuple[ProjectedPrediction, float]] = []
        # 有声片段的采样点总数，用于计算有声覆盖率（voiced_ratio）
        voiced_samples = 0
        # 所有片段的采样点总数，用于计算有声覆盖率
        total_samples = sum(item.sample_count for item in segments)

        # 逐段推理
        for current, segment in enumerate(segments, start=1):
            if segment.is_silent:
                # 静音段：跳过推理，仅记录基本信息（索引、时间范围、静音标记）
                results.append(
                    SegmentResult(
                        index=segment.index,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        is_silent=True,
                    )
                )
            else:
                # 有声段：执行模型推理 → 六类→四类投影
                prediction = project_probabilities(self.runtime.predict(segment.waveform))
                # 记录完整的段级结果
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
                # 计算权重：采样点数 × RMS 能量（裁剪到稳定区间）
                # - RMS 裁剪下限 0.02：防止极低能量段获得不合理权重
                # - RMS 裁剪上限 0.30：防止一次爆音压过整段通话的其他窗口
                # - 采样点数因子：长片段自然获得更高权重，与直觉一致
                weight = segment.sample_count * float(np.clip(segment.rms, 0.02, 0.30))
                weighted.append((prediction, weight))
                voiced_samples += segment.sample_count
            # 产出进度事件，提示前端当前推理进度
            yield ProgressEvent(
                type="progress",
                current=current,
                total=len(segments),
                message=f"正在分析第 {current}/{len(segments)} 段",
            )

        # ---------- 加权聚合 ----------
        # 能量被裁剪到稳定区间，避免一次爆音压过整段通话的其他窗口。
        weights = np.array([weight for _, weight in weighted], dtype=np.float64)
        # 归一化权重：使总和为 1，保证加权平均的概率分布合理
        weights /= weights.sum()
        # 四类目标情绪的固定顺序，保证聚合结果的字段顺序一致
        target_order = ("neutral", "happy", "anger", "sad")
        # 对四类概率分别进行加权平均
        aggregate_values = {
            name: float(
                sum(
                    weight * getattr(prediction.probabilities, name)
                    for weight, (prediction, _) in zip(weights, weighted, strict=True)
                )
            )
            for name in target_order
        }
        # 对排除概率进行加权平均
        excluded = float(
            sum(
                weight * prediction.excluded_probability
                for weight, (prediction, _) in zip(weights, weighted, strict=True)
            )
        )
        # 将聚合后的四类概率放回六类槽位，复用同一套可靠性阈值。
        # 构造方式：四类各乘以 (1 - excluded) 占比，排除类各分得 excluded/2
        # 这样重新放入 project_probabilities() 后，可复用相同的可靠性评估逻辑
        raw = np.array(
            [
                aggregate_values["anger"] * (1 - excluded),
                excluded / 2,  # fear 的份额
                aggregate_values["happy"] * (1 - excluded),
                aggregate_values["neutral"] * (1 - excluded),
                aggregate_values["sad"] * (1 - excluded),
                excluded / 2,  # surprise 的份额
            ]
        )
        # 对聚合后的六类概率再次投影，得到整体情绪判定与可靠性评估
        overall = project_probabilities(raw)

        # 计算有声覆盖率：有声片段采样点数 / 总采样点数
        # min(1.0, ...) 防止浮点误差导致略超 1.0
        # max(total_samples, 1) 防止零除（虽然理论上 total_samples 不会为零）
        voiced_ratio = min(1.0, voiced_samples / max(total_samples, 1))
        # 补充可靠性评估：有声覆盖率 < 30% → "LOW_VOICE_COVERAGE"
        # 阈值 0.30：低于此值意味着超过 70% 的音频为静音，
        # 推理结果仅基于极少量有声片段，整体可信度较低
        reasons = list(overall.reliability.reasons)
        if voiced_ratio < 0.3:
            reasons.append("LOW_VOICE_COVERAGE")
        # 若有任何可靠性原因，标记为 low；否则标记为 high
        reliability = Reliability(level="low" if reasons else "high", reasons=reasons)

        # 构建完整的分析结果
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
        # 产出最终结果事件
        yield ResultEvent(result=result)
