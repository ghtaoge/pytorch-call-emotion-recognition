"""
test_schemas —— Pydantic 数据模型契约的单元测试

本模块测试 app.schemas 中定义的 Pydantic 模型，验证：
- AnalysisResult 的四类概率序列化契约（只输出 neutral/happy/anger/sad）
- EmotionProbabilities 的概率边界约束（0~1 之间）
- SegmentResult 的静音/有声段互斥规则
- ProgressEvent 的进度数值约束（current <= total）
- ResultEvent 的默认类型字段
- 各模型拒绝额外字段的严格模式
- PublicError 与 ErrorEvent 的序列化安全
- AppError 只暴露安全的公共字段

确保数据模型的契约在序列化与反序列化过程中始终一致。
"""

import json

import pytest
from pydantic import ValidationError

from app.errors import AppError
from app.schemas import (
    AnalyzeUrlRequest,
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
    """构造一组有效的四类概率分布，用于后续测试的默认值。

    neutral=0.7, happy=0.1, anger=0.1, sad=0.1，总和为 1.0。
    """
    return EmotionProbabilities(neutral=0.7, happy=0.1, anger=0.1, sad=0.1)


def reliability() -> Reliability:
    """构造一个有效的可靠性评级，用于后续测试的默认值。

    level="high", reasons=[]，表示高可靠度且无异常原因。
    """
    return Reliability(level="high", reasons=[])


def voiced_segment(**overrides: object) -> SegmentResult:
    """构造一个有效的有声段结果，可通过 overrides 覆盖任意字段。

    默认值：index=0, start=0.0, end=4.0, is_silent=False，
    以及完整的概率、可靠性、情感和排除概率。
    """
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
    """构造一个完整的分析结果，用于后续测试的默认值。

    包含主导情感、概率分布、可靠性评级、排除概率、有声占比等全部字段。
    """
    return AnalysisResult(
        dominant_emotion="neutral",
        probabilities=probabilities(),
        reliability=reliability(),
        excluded_probability=0.08,
        voiced_ratio=0.9,  # 有声占比 90%
        duration_seconds=4.0,  # 音频时长 4 秒
        device="cpu",  # 推理设备
        elapsed_ms=120,  # 推理耗时 120ms
        segments=[voiced_segment()],  # 包含一个有声段
    )


def test_analysis_result_serializes_four_probability_contract() -> None:
    """验证 AnalysisResult 序列化时只包含四类概率字段（契约测试）。

    序列化后的 probabilities 字典应仅包含 neutral、happy、anger、sad 四个键，
    且概率值总和应为 1.0。
    同时验证 model_dump 与 model_dump_json 的输出一致。
    """
    result = analysis_result()

    payload = result.model_dump(mode="json")

    assert payload["dominant_emotion"] == "neutral"  # 主导情感应为 neutral
    # 序列化后的概率字段应只包含四类情感键
    assert set(payload["probabilities"]) == {"neutral", "happy", "anger", "sad"}
    # 四类概率总和应为 1.0
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0)
    # JSON 序列化与字典序列化应产生相同结果
    assert json.loads(result.model_dump_json()) == payload


@pytest.mark.parametrize(
    ("field", "value"),
    [("neutral", -0.01), ("happy", 1.01)],
)
def test_probability_bounds_reject_invalid_values(field: str, value: float) -> None:
    """验证 EmotionProbabilities 拒绝超出 [0, 1] 范围的概率值。

    - neutral=-0.01：负概率，应被拒绝
    - happy=1.01：超过 1.0 的概率，应被拒绝
    确保概率值始终在合理范围内。
    """
    values = {"neutral": 0.7, "happy": 0.1, "anger": 0.1, "sad": 0.1}
    values[field] = value  # 将某个字段设为越界值

    with pytest.raises(ValidationError):
        EmotionProbabilities(**values)  # type: ignore[arg-type]  # 越界值应触发校验错误


def test_silent_segment_requires_prediction_fields_to_be_none() -> None:
    """验证静音段必须将所有预测相关字段设为 None。

    当 is_silent=True 时，以下字段必须为 None：
    - probabilities：无概率分布
    - dominant_emotion：无主导情感
    - reliability：无可靠性评级
    - excluded_probability：无排除概率

    同时验证有声段（is_silent=False）不能将这些字段设为 None。
    """
    # 构造一个合法的静音段：所有预测字段均为 None
    segment = SegmentResult(
        index=0,
        start_seconds=0.0,
        end_seconds=4.0,
        is_silent=True,
        probabilities=None,  # 静音段无概率分布
        dominant_emotion=None,  # 静音段无主导情感
        reliability=None,  # 静音段无可靠性评级
        excluded_probability=None,  # 静音段无排除概率
    )

    assert segment.is_silent is True  # 应为静音段
    assert segment.probabilities is None  # 预测字段应为 None

    # 有声段不能将 is_silent=True 与非 None 的预测字段组合
    with pytest.raises(ValidationError):
        voiced_segment(is_silent=True)  # is_silent=True 但预测字段非 None，应被拒绝


@pytest.mark.parametrize(
    "missing_field",
    ["probabilities", "dominant_emotion", "reliability", "excluded_probability"],
)
def test_voiced_segment_requires_every_prediction_field(missing_field: str) -> None:
    """验证有声段的每个预测字段都不能为 None。

    对于有声段（is_silent=False），以下字段均为必填：
    - probabilities、dominant_emotion、reliability、excluded_probability
    逐一将每个字段设为 None，确保都能触发 ValidationError。
    """
    with pytest.raises(ValidationError):
        # 将某个必填字段设为 None，应触发校验错误
        voiced_segment(**{missing_field: None})


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds"),
    [(2.0, 2.0), (2.0, 1.0)],
)
def test_segment_end_must_be_after_start(start_seconds: float, end_seconds: float) -> None:
    """验证分段的时间约束：end_seconds 必须严格大于 start_seconds。

    - start=2.0, end=2.0：起止时间相同，段长度为零，应被拒绝
    - start=2.0, end=1.0：结束时间早于起始时间，逻辑错误，应被拒绝
    """
    with pytest.raises(ValidationError):
        voiced_segment(start_seconds=start_seconds, end_seconds=end_seconds)  # 非法时间区间


def test_progress_rejects_current_greater_than_total() -> None:
    """验证 ProgressEvent 拒绝 current > total 的进度值。

    当已完成段数大于总段数时，进度数据不合理，应触发 ValidationError。
    """
    with pytest.raises(ValidationError):
        ProgressEvent(type="progress", current=2, total=1, message="处理中")  # current > total，应被拒绝


def test_result_event_has_stable_default_type() -> None:
    """验证 ResultEvent 的 type 字段具有稳定的默认值 "result"。

    创建 ResultEvent 后，type 应自动设为 "result"，
    且序列化后也应保持 "result"，确保客户端能正确识别事件类型。
    """
    event = ResultEvent(result=analysis_result())

    assert event.type == "result"  # 默认类型应为 "result"
    assert json.loads(event.model_dump_json())["type"] == "result"  # 序列化后也应保持 "result"


def test_schemas_reject_extra_fields() -> None:
    """验证各模型拒绝未定义的额外字段（严格模式）。

    尝试在 EmotionProbabilities 中添加 fear=0.0 字段，
    应触发 ValidationError，防止意外数据注入。
    """
    with pytest.raises(ValidationError):
        # 添加未定义的 fear 字段，应被拒绝
        EmotionProbabilities(
            neutral=0.7,
            happy=0.1,
            anger=0.1,
            sad=0.1,
            fear=0.0,  # 未定义的字段
        )


def test_public_error_rejects_blank_fields() -> None:
    """验证 PublicError 拒绝空白（纯空格/制表符）的 code 与 message。

    - code 为纯空格 " "：应被拒绝
    - message 为纯制表符 "\t"：应被拒绝
    确保公共错误信息始终包含有意义的文本。
    """
    with pytest.raises(ValidationError):
        PublicError(code=" ", message="safe")  # code 为纯空格，应被拒绝
    with pytest.raises(ValidationError):
        PublicError(code="AUDIO_INVALID", message="\t")  # message 为纯制表符，应被拒绝


def test_error_event_serializes_only_public_error() -> None:
    """验证 ErrorEvent 序列化时只暴露 PublicError 的公共字段。

    序列化后的 JSON 应只包含 type 和 error 两个键，
    error 内只包含 code 和 message（不含内部调试信息）。
    确保错误响应不泄露敏感的系统细节。
    """
    event = ErrorEvent(
        type="error",
        error=PublicError(code="AUDIO_INVALID", message="音频无效"),
    )

    # 验证序列化结果只包含公共字段
    assert event.model_dump(mode="json") == {
        "type": "error",
        "error": {"code": "AUDIO_INVALID", "message": "音频无效"},  # 只包含 code 和 message
    }


def test_app_error_exposes_only_safe_public_fields() -> None:
    """验证 AppError 只暴露安全的公共字段，不泄露内部细节。

    AppError 的 str() 应返回 public_message（面向用户的安全信息），
    而不包含 code 或 status_code。vars() 应只包含：
    - code：错误代码（用于 API 响应）
    - public_message：面向用户的安全消息
    - status_code：HTTP 状态码
    确保异常信息在日志与响应中的一致性与安全性。
    """
    error = AppError("AUDIO_INVALID", "音频无效", status_code=422)

    assert str(error) == "音频无效"  # str() 应返回面向用户的消息
    assert error.args == ("音频无效",)  # args 应只包含公共消息
    # vars() 应只包含三个安全的公共字段
    assert vars(error) == {
        "code": "AUDIO_INVALID",  # 错误代码
        "public_message": "音频无效",  # 面向用户的消息
        "status_code": 422,  # HTTP 状态码
    }


def test_analyze_url_request_accepts_valid_urls() -> None:
    """验证 AnalyzeUrlRequest 接受合法的 http/https URL。"""
    for url in ["http://example.com/audio.wav", "https://cdn.example.com/file.mp3"]:
        req = AnalyzeUrlRequest(url=url)
        assert req.url == url


def test_analyze_url_request_rejects_invalid_protocols() -> None:
    """验证 AnalyzeUrlRequest 拒绝非 http/https 协议的 URL。"""
    for url in ["ftp://example.com/audio.wav", "file:///tmp/audio.wav", "just-a-string"]:
        with pytest.raises(ValidationError):
            AnalyzeUrlRequest(url=url)


def test_analyze_url_request_rejects_blank_url() -> None:
    """验证 AnalyzeUrlRequest 拒绝空白 URL。"""
    with pytest.raises(ValidationError):
        AnalyzeUrlRequest(url="   ")


def test_analyze_url_request_rejects_extra_fields() -> None:
    """验证 AnalyzeUrlRequest 拒绝额外字段（严格模式）。"""
    with pytest.raises(ValidationError):
        AnalyzeUrlRequest(url="http://example.com/audio.wav", extra="nope")


def test_analyze_url_request_strips_whitespace() -> None:
    """验证 AnalyzeUrlRequest 对 URL 执行空白去除。"""
    req = AnalyzeUrlRequest(url="  https://example.com/audio.wav  ")
    assert req.url == "https://example.com/audio.wav"
