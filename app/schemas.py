"""
app.schemas — API 数据契约模型

本模块定义了所有对外暴露的数据结构，基于 Pydantic BaseModel 实现。
这些模型既是 API 请求/响应的类型定义，也是数据校验的守门员。

设计原则：
1. ContractModel 基类：extra="forbid" — 严格禁止未声明的字段，
   防止客户端意外传入多余数据（避免隐式字段污染，保证 API 契约清晰）
   与 Settings 的 extra="ignore" 不同：Settings 接受多余环境变量（.env 可能混放其他项目变量），
   而 ContractModel 拒绝多余字段（API 契约应该严格且明确）
2. 类型别名：Probability、NonNegativeFloat 等带约束的 Annotated 类型，
   在模型字段声明中直接引用，避免重复书写校验逻辑
3. 互斥状态校验：SegmentResult 的 model_validator 确保静音段与有声段的
   字段组合互斥且完整，避免产生歧义状态
4. 事件模型：ProgressEvent / ResultEvent / ErrorEvent 为 NDJSON 流式输出的
   三种事件类型，每种事件都有独立的 type 标识，便于前端按类型分发处理

模型层次结构：
- EmotionProbabilities : 四类目标情绪概率（归一化后总和为 1）
- Reliability          : 可靠性评估（level + reasons）
- SegmentResult        : 单段分析结果（静音段或有声段）
- AnalysisResult       : 完整分析结果（聚合后的全局结果 + 分段详情）
- HealthResponse       : 健康检查响应
- ProgressEvent        : 进度事件（NDJSON 流式输出）
- ResultEvent          : 结果事件（NDJSON 流式输出）
- PublicError          : 可公开的错误信息（安全过滤后的）
- ErrorEvent           : 错误事件（NDJSON 流式输出）
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------- 类型别名：带约束的 Pydantic Annotated 类型 ----------
# Emotion — 四类目标情绪的枚举类型（Literal 保证值域固定）
Emotion = Literal["neutral", "happy", "anger", "sad"]
# ReliabilityReason — 可靠性降低原因的枚举类型（四种固定原因）
ReliabilityReason = Literal[
    "OUTSIDE_FOUR_CLASS_SCOPE",   # 被排除类别概率过高，语音更接近 fear/surprise
    "LOW_TOP_PROBABILITY",        # 顶部概率过低，各类别分布均匀
    "SMALL_TOP_MARGIN",           # 前两名概率差距过小
    "LOW_VOICE_COVERAGE",         # 有声覆盖率过低（< 30%）
]
# Probability — 概率值（0 ≤ x ≤ 1），用于情绪概率、有声覆盖率等
Probability = Annotated[float, Field(ge=0, le=1)]
# NonNegativeFloat — 非负浮点数（x ≥ 0），用于时间偏移等不允许为负的参数
NonNegativeFloat = Annotated[float, Field(ge=0)]
# PositiveFloat — 严格正浮点数（x > 0），用于时长等不允许为零的参数
PositiveFloat = Annotated[float, Field(gt=0)]
# NonNegativeInt — 非负整数（n ≥ 0），用于片段索引、段数等
NonNegativeInt = Annotated[int, Field(ge=0)]


class ContractModel(BaseModel):
    """
    API 契约模型基类 — 所有对外数据结构的统一基类

    配置项：
    - extra="forbid"：严格禁止未声明的字段。
      设计意图：防止客户端意外传入多余数据，避免隐式字段污染，
      保证 API 契约清晰且不随版本变更而模糊。
      与 Settings 的 extra="ignore" 不同：
      Settings 接受多余环境变量（因为 .env 文件可能混放其他项目的变量），
      而 ContractModel 拒绝多余字段（因为 API 契约应该严格且明确）。
    """
    model_config = ConfigDict(extra="forbid")


class EmotionProbabilities(ContractModel):
    """
    四类目标情绪概率 — 六类→四类投影后的结果

    字段顺序固定为 neutral, happy, anger, sad，
    与前端展示顺序一致（中性 > 积极 > 消极-愤怒 > 消极-悲伤）。

    各字段为 Probability 类型（0 ≤ x ≤ 1），
    注意：归一化后总和应为 1，但 Pydantic 未对总和做校验
    （因为总和校验需要 model_validator，而此处四个字段独立声明更简洁）。
    总和校验由 project_probabilities() 的算法逻辑保证（归一化后总和 = 1）。
    """
    neutral: Probability     # 中性情绪概率
    happy: Probability       # 开心情绪概率
    anger: Probability       # 愤怒情绪概率
    sad: Probability         # 悲伤情绪概率


class Reliability(ContractModel):
    """
    可靠性评估 — 判断分析结果的可信程度

    字段说明：
    - level : 可靠性等级，"high" 或 "low"
              "high" 表示无任何可靠性警告，结果可信
              "low" 表示至少有一个可靠性警告，结果需谨慎解读
    - reasons : 可靠性降低原因列表（空列表表示 "high"）
              每个原因对应一个特定的可靠性问题：
              - OUTSIDE_FOUR_CLASS_SCOPE : 被排除类别概率过高
              - LOW_TOP_PROBABILITY : 顶部概率过低
              - SMALL_TOP_MARGIN : 前两名概率差距过小
              - LOW_VOICE_COVERAGE : 有声覆盖率过低
    """
    level: Literal["high", "low"]   # high: 置信度较高; low: 需谨慎参考
    reasons: list[ReliabilityReason]  # 降级原因列表，空列表表示 high 级别


class SegmentResult(ContractModel):
    """
    单段分析结果 — 滑动窗口切分后每个片段的分析输出

    字段说明：
    - index          : 片段序号（从 0 开始），用于排序与定位
    - start_seconds  : 片段起始时间（秒）
    - end_seconds    : 片段结束时间（秒），必须大于 start_seconds
    - is_silent      : 是否为静音片段（RMS < 阈值）
    - probabilities  : 四类情绪概率（仅有声段有值，静音段为 None）
    - dominant_emotion: 主导情绪（仅有声段有值，静音段为 None）
    - reliability    : 可靠性评估（仅有声段有值，静音段为 None）
    - excluded_probability: 被排除类别概率（仅有声段有值，静音段为 None）

    互斥状态校验：
    - 静音段不得混入推理结果（四个预测字段均为 None）
    - 有声段必须提供完整结果（四个预测字段均不为 None）
    - 此校验防止产生歧义状态（如：静音段携带推理结果，或有声段缺少可靠性评估）
    """
    index: NonNegativeInt                      # 段序号（从 0 开始）
    start_seconds: NonNegativeFloat            # 段起始时间（秒）
    end_seconds: NonNegativeFloat              # 段结束时间（秒）
    is_silent: bool                            # 是否为静音段
    # 预测字段：有声段必有，静音段必无
    probabilities: EmotionProbabilities | None = None  # 四类概率（静音段为 None）
    dominant_emotion: Emotion | None = None              # 主导情绪（静音段为 None）
    reliability: Reliability | None = None               # 可靠性评级（静音段为 None）
    excluded_probability: Probability | None = None      # 被排除类概率（静音段为 None）

    @model_validator(mode="after")
    def validate_segment_state(self) -> Self:
        """
        模型级校验：保证静音段与有声段的字段组合互斥且完整

        校验规则：
        1. end_seconds > start_seconds：片段时间范围必须有效
        2. 静音段不得混入推理结果：四个预测字段均必须为 None
           原因：静音段没有可分析的语音内容，推理结果无意义
        3. 有声段必须提供完整结果：四个预测字段均必须不为 None
           原因：有声段经过完整推理流程，所有结果字段都应该被填充，
           缺少任何字段都会导致前端无法正确展示
        """
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
    """
    完整分析结果 — 聚合后的全局结果 + 分段详情

    此模型是 NDJSON 流式输出的最终事件内容，
    包含全局情绪判定、概率分布、可靠性评估和所有段的详细结果。

    字段说明：
    - dominant_emotion     : 全局主导情绪（加权聚合后的最高概率类别）
    - probabilities        : 全局四类概率分布（加权平均后归一化）
    - reliability          : 全局可靠性评估（包含段级与聚合级的所有原因）
    - excluded_probability : 全局被排除类别概率（加权平均）
    - voiced_ratio         : 有声覆盖率（有声采样点数 / 总采样点数）
    - duration_seconds     : 音频总时长（秒）
    - device               : 推理设备标识（"cuda" / "mps" / "cpu"）
    - elapsed_ms           : 分析总耗时（毫秒），用于性能监控
    - segments             : 分段详情列表，按时间顺序排列
    """
    dominant_emotion: Emotion                 # 主导情绪
    probabilities: EmotionProbabilities       # 加权聚合后的四类概率
    reliability: Reliability                  # 整体可靠性评级
    excluded_probability: Probability         # 被排除类（恐惧+惊讶）的加权概率
    voiced_ratio: Probability                 # 有效语音占比（有效采样点 / 总采样点）
    duration_seconds: PositiveFloat           # 音频总时长（秒）
    device: str                               # 推理设备标识（cuda/mps/cpu）
    elapsed_ms: NonNegativeInt                # 分析耗时（毫秒）
    segments: list[SegmentResult]             # 分段分析结果列表

    @field_validator("device")
    @classmethod
    def reject_blank_device(cls, value: str) -> str:
        """
        字段级校验：拒绝空白设备标识

        防止 device 字段为空字符串或纯空格，
        此类错误若不在此拦截，会导致前端无法正确展示推理设备信息。
        """
        if not value.strip():
            raise ValueError("device must not be blank")
        return value


class HealthResponse(ContractModel):
    """
    健康检查响应 — API /api/health 端点的返回格式

    字段说明：
    - status       : 服务状态，固定为 "ok"（只要端点能响应就说明服务运行正常）
    - model_status : 模型加载状态，可能值："not_loaded" / "loading" / "loaded" / "error"
    - device       : 推理设备标识
    """
    status: Literal["ok"]                                              # 服务状态（固定为 "ok"）
    model_status: Literal["not_loaded", "loading", "loaded", "error"]  # 模型生命周期状态
    device: str                                                        # 推理设备标识


class ProgressEvent(ContractModel):
    """
    进度事件 — NDJSON 流式输出的中间事件

    前端根据 type 字段区分事件类型：
    - "status"  : 初始状态事件，提示模型准备中
    - "progress": 进度更新事件，提示当前推理进度

    字段说明：
    - type    : 事件类型标识，"status" 或 "progress"
    - current : 已完成的段数
    - total   : 总段数
    - message : 人类可读的进度描述（用于前端展示）
    """
    type: Literal["status", "progress"]  # 事件类型：status(初始状态) / progress(分段进度)
    current: NonNegativeInt              # 当前已完成段数
    total: NonNegativeInt                # 总段数
    message: str                         # 面向用户的中文进度提示

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        """
        模型级校验：进度数值必须合理

        current 不得超过 total，防止前端显示进度异常（如 5/3）。
        允许 current == total（表示所有段已完成）。
        """
        if self.current > self.total:
            raise ValueError("current must not exceed total")
        return self


class ResultEvent(ContractModel):
    """
    结果事件 — NDJSON 流式输出的最终事件

    type 固定为 "result"，前端收到此事件后停止监听流。

    字段说明：
    - type   : 事件类型标识，固定为 "result"
    - result : 完整分析结果（AnalysisResult 对象）
    """
    type: Literal["result"] = "result"  # 事件类型（固定为 "result"）
    result: AnalysisResult              # 完整分析结果


class PublicError(ContractModel):
    """
    可公开的错误信息 — 安全过滤后的错误描述

    设计意图：
    仅保存可安全返回给调用方的字段（code + message），不包含：
    - 内部错误详情（堆栈跟踪、源代码位置）
    - 用户原始输入（音频文件名、上传数据）
    - 内部路径（临时文件路径、模型缓存路径）

    此模型保证错误响应不会泄露敏感信息，
    即使在第三方库抛出包含临时路径的异常时也能安全处理。

    字段说明：
    - code    : 错误代码标识（如 "NO_VOICE", "FILE_TOO_LARGE"），供前端逻辑判断
    - message : 人类可读的错误描述（如 "未检测到清晰人声"），供前端展示
    """
    code: str      # 机器可读的错误码
    message: str   # 面向用户的中文提示

    @field_validator("code", "message")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        """
        字段级校验：拒绝空白字符串

        错误代码和消息必须非空，否则前端无法正确判断错误类型或展示错误描述。
        """
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ErrorEvent(ContractModel):
    """
    错误事件 — NDJSON 流式输出的错误事件

    前端收到此事件后停止监听流并展示错误信息。
    此事件仅在分析过程中抛出 AppError 或未知异常时产生。

    字段说明：
    - type  : 事件类型标识，固定为 "error"
    - error : 可公开的错误信息（PublicError 对象）
    """
    type: Literal["error"]  # 事件类型（固定为 "error"）
    error: PublicError      # 公开错误对象


class AnalyzeUrlRequest(ContractModel):
    """
    URL 音频分析请求 — 接收音频文件 URL 地址

    字段说明：
    - url : 音频文件 URL，必须以 http:// 或 https:// 开头
            支持任意公网和内网 URL，不做私有 IP 过滤

    安全设计：
    - 协议白名单：仅允许 http 和 https，拒绝 file、ftp 等协议
    - URL 两端空白自动去除，防止用户误输入前后空格
    """
    url: str  # 音频文件 URL

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """
        字段级校验：URL 协议白名单 + 空白去除

        校验规则：
        1. 去除两端空白（防止用户误输入前后空格）
        2. URL 必须以 http:// 或 https:// 开头
           拒绝其他协议（file://、ftp:// 等）防止 SSRF
        """
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value
