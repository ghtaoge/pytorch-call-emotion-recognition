"""
test_config —— 配置校验、缓存与冻结行为的单元测试

本模块测试 app.config 中 Settings 类的各项行为：
- 默认值的安全性与合理性
- 非法参数值的拒绝（如 hop > window、零值端口、负数阈值等）
- 环境变量覆盖机制
- get_settings 的缓存行为与缓存清除
- Settings 的冻结（不可变）与忽略额外输入

确保配置系统在生产环境中不会因错误参数而崩溃。
"""

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

# 所有可通过环境变量覆盖的配置键名
ENVIRONMENT_KEYS = (
    "MODEL_ID",
    "MAX_BYTES",
    "MAX_DURATION_SECONDS",
    "WINDOW_SECONDS",
    "HOP_SECONDS",
    "SILENCE_RMS_THRESHOLD",
    "HOST",
    "PORT",
)


def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """清除所有配置相关的环境变量，确保测试从纯净状态开始。

    使用 raising=False 防止在环境变量不存在时抛出异常。
    """
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_settings_use_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Settings 在无环境变量时使用安全的默认值。

    清除所有环境变量后创建 Settings，检查每个字段的默认值：
    - model_id: 默认使用 HuggingFace 上的中文情感识别模型
    - max_bytes: 默认 50MB（音频文件大小上限）
    - max_duration_seconds: 默认 300 秒（5 分钟）
    - window_seconds: 默认 6 秒（分析窗口）
    - hop_seconds: 默认 5 秒（滑动步长）
    - silence_rms_threshold: 默认 0.01（静音判定阈值）
    - host: 默认 127.0.0.1（仅本地访问）
    - port: 默认 8000
    """
    clear_settings_environment(monkeypatch)

    settings = Settings()

    assert settings.model_id == "xmj2002/hubert-base-ch-speech-emotion-recognition"  # 默认模型 ID
    assert settings.max_bytes == 50 * 1024 * 1024  # 默认最大文件大小 50MB
    assert settings.max_duration_seconds == 300.0  # 默认最大时长 5 分钟
    assert settings.window_seconds == 6.0  # 默认分析窗口 6 秒
    assert settings.hop_seconds == 5.0  # 默认滑动步长 5 秒
    assert settings.silence_rms_threshold == 0.01  # 默认静音阈值
    assert settings.host == "127.0.0.1"  # 默认仅本地访问
    assert settings.port == 8000  # 默认端口


def test_settings_reject_hop_larger_than_window() -> None:
    """验证 Settings 拒绝 hop_seconds > window_seconds 的配置。

    当滑动步长大于分析窗口时，分段逻辑会出错，因此必须拒绝。
    此测试确保 Pydantic 校验器捕获此逻辑错误。
    """
    with pytest.raises(ValidationError):
        Settings(window_seconds=5.0, hop_seconds=6.0)  # hop > window，应被拒绝


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_bytes", 0),  # 最大文件大小不能为零
        ("max_duration_seconds", 0),  # 最大时长不能为零
        ("window_seconds", 0),  # 分析窗口不能为零
        ("hop_seconds", 0),  # 滑动步长不能为零
        ("port", 0),  # 端口不能为零
        ("port", 65_536),  # 端口不能超过 65535
        ("silence_rms_threshold", -0.01),  # 阈值不能为负数
        ("silence_rms_threshold", 1.01),  # 阈值不能超过 1.0
        ("model_id", "  "),  # 模型 ID 不能为纯空白
        ("host", ""),  # 主机地址不能为空字符串
    ],
)
def test_settings_reject_invalid_limits(field: str, value: object) -> None:
    """验证 Settings 拒绝各种非法参数值。

    通过参数化测试覆盖多种边界情况：
    - 零值：max_bytes、max_duration、window、hop、port
    - 越界：port > 65535、threshold < 0 或 > 1
    - 空值：model_id 为空白、host 为空字符串
    每种情况都应触发 ValidationError。
    """
    with pytest.raises(ValidationError):
        Settings(**{field: value})  # type: ignore[arg-type]  # 传入非法值，应触发校验错误


def test_settings_honor_uppercase_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Settings 支持大写环境变量覆盖默认值。

    设置环境变量 SILENCE_RMS_THRESHOLD=0.25 后，
    创建 Settings 时应读取该值而非使用默认值 0.01。
    """
    monkeypatch.setenv("SILENCE_RMS_THRESHOLD", "0.25")

    assert Settings().silence_rms_threshold == 0.25  # 应使用环境变量的值


def test_get_settings_caches_until_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 get_settings 的缓存行为：缓存实例直到手动清除。

    第一次调用 get_settings() 会创建并缓存 Settings 实例。
    修改环境变量后再次调用，仍返回缓存的旧实例。
    调用 cache_clear() 后再调用，才会创建新实例并读取最新环境变量。
    """
    get_settings.cache_clear()  # 先清除缓存，确保从纯净状态开始
    monkeypatch.setenv("SILENCE_RMS_THRESHOLD", "0.2")

    first = get_settings()  # 第一次调用：创建并缓存实例
    monkeypatch.setenv("SILENCE_RMS_THRESHOLD", "0.3")  # 修改环境变量
    cached = get_settings()  # 第二次调用：返回缓存实例，不受环境变量变化影响
    get_settings.cache_clear()  # 清除缓存
    refreshed = get_settings()  # 第三次调用：创建新实例，读取最新环境变量
    get_settings.cache_clear()  # 清理缓存

    assert cached is first  # 缓存实例与首次实例应相同（同一对象）
    assert cached.silence_rms_threshold == 0.2  # 缓存实例仍使用旧的环境变量值
    assert refreshed is not first  # 刷新后应为新对象
    assert refreshed.silence_rms_threshold == 0.3  # 新实例应使用最新的环境变量值


def test_settings_are_frozen_and_ignore_extra_input() -> None:
    """验证 Settings 的冻结行为与忽略额外输入。

    - Settings 应忽略未定义的字段（如 unused_value），防止意外注入
    - Settings 实例创建后应不可修改（冻结），防止运行时篡改配置
    """
    # 传入未定义的字段，Settings 应忽略它
    settings = Settings(unused_value="ignored")  # type: ignore[call-arg]

    assert not hasattr(settings, "unused_value")  # 未定义字段不应出现在实例上
    # 尝试修改已创建的实例，应触发 ValidationError
    with pytest.raises(ValidationError):
        settings.port = 9000  # 冻结实例不允许修改字段
