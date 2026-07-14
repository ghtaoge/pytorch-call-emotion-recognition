"""
app.config — 全局配置模块

基于 Pydantic Settings 实现类型安全的环境变量与 .env 文件配置。
所有配置项均带有严格的类型校验（正数、归一化浮点、端口范围等），
防止非法值在启动阶段就进入系统，而非在运行时才暴露问题。

关键设计决策：
- frozen=True：配置对象一旦创建不可修改，避免运行期间被意外篡改
- extra="ignore"：忽略 .env 或环境变量中未声明的字段，允许 .env 文件混放其他项目的变量
- lru_cache 缓存：全局单例模式，保证整个进程中只创建一份 Settings 实例，
  既节省内存也保证所有模块引用同一份配置
"""

from functools import lru_cache
from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------- 类型别名：带约束的 Pydantic Annotated 类型 ----------
# PositiveFloat — 严格正浮点数（>0），用于时长等不允许为零的参数
PositiveFloat = Annotated[float, Field(gt=0)]
# PositiveInt — 严格正整数（>0），用于文件大小上限等不允许为零的参数
PositiveInt = Annotated[int, Field(gt=0)]
# NormalizedFloat — 归一化浮点数（0 ≤ x ≤ 1），用于阈值、比例等归一化参数
NormalizedFloat = Annotated[float, Field(ge=0, le=1)]
# Port — TCP 端口（1 ~ 65535），用于服务监听端口
Port = Annotated[int, Field(ge=1, le=65_535)]


class Settings(BaseSettings):
    """
    全局配置类 — 所有运行参数的唯一定义点

    配置来源优先级（Pydantic Settings 默认）：
    环境变量 > .env 文件 > 默认值

    各字段说明：
    - model_id          : HuggingFace 模型仓库标识符，决定加载哪个预训练权重
    - max_bytes         : 上传音频文件大小上限（字节），50 MB，防止恶意大文件耗尽内存
    - max_duration_seconds : 音频时长上限（秒），300 秒即 5 分钟，与业务需求匹配
    - window_seconds    : 滑动窗口长度（秒），6 秒 ≈ HuBERT 预训练片段长度，
                          太短则语义不完整，太长则边界模糊
    - hop_seconds       : 滑动窗口步长（秒），5 秒意味着相邻段有 1 秒重叠，
                          重叠区提供上下文过渡信息，避免边界处漏检
    - silence_rms_threshold : 静音判定阈值（归一化 RMS），0.01 对应约 -40 dB，
                              低于此值的片段视为静音，跳过模型推理以节省计算
    - host              : 服务监听地址，默认仅本地访问（127.0.0.1）
    - port              : 服务监听端口，默认 8000
    - url_download_timeout_seconds : URL 音频下载超时（秒），默认 60
    - url_max_redirects : URL 重定向最大次数，默认 5
    """

    model_config = SettingsConfigDict(
        # 从项目根目录的 .env 文件读取配置
        env_file=".env",
        env_file_encoding="utf-8",
        # 忽略未声明的环境变量，避免与其他项目的 .env 冲突
        extra="ignore",
        # 冻结实例：创建后不可修改字段，保证运行期间配置一致性
        frozen=True,
    )

    # HuggingFace 模型仓库 ID，包含组织名和模型名
    model_id: str = "xmj2002/hubert-base-ch-speech-emotion-recognition"
    # 上传文件大小上限：50 MB — 足以覆盖 5 分钟高质量音频，同时防止内存溢出
    max_bytes: PositiveInt = 50 * 1024 * 1024
    # 音频时长上限：300 秒（5 分钟）— 与通话情绪分析的业务场景匹配
    max_duration_seconds: PositiveFloat = 300.0
    # 滑动窗口长度：6 秒 — 与 HuBERT 预训练时使用的片段长度接近，
    # 保证模型在每个窗口内有足够的语义上下文
    window_seconds: PositiveFloat = 6.0
    # 滑动窗口步长：5 秒 — 与 window_seconds 的差值即为重叠长度（1 秒），
    # 重叠区提供过渡上下文，减少窗口边界处的情绪漏检
    hop_seconds: PositiveFloat = 5.0
    # 静音判定阈值：归一化 RMS < 0.01 ≈ -40 dB 视为静音，
    # 该阈值在安静通话与适度噪声之间取得平衡
    silence_rms_threshold: NormalizedFloat = 0.01
    # 服务监听地址：默认仅绑定本地回环，生产环境可通过环境变量改为 0.0.0.0
    host: str = "127.0.0.1"
    # 服务监听端口
    port: Port = 8000
    # URL 音频下载超时：默认 60 秒 — 足以覆盖大多数网络延迟，
    # 同时不至于让单个请求长时间阻塞服务
    url_download_timeout_seconds: PositiveFloat = 60.0
    # URL 重定向最大次数：限制重定向链长度，防止无限循环
    url_max_redirects: PositiveInt = 5

    @field_validator("model_id", "host")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        """
        字段级校验：拒绝空白字符串

        防止用户误设 model_id 或 host 为空格/空字符串，
        此类错误若不在此拦截，会在模型加载或网络绑定阶段才暴露，
        导致难以定位的错误信息。
        """
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_hop_size(self) -> Self:
        """
        模型级校验：步长不得超过窗口长度

        若 hop_seconds > window_seconds，滑动窗口之间会出现间隙（无重叠），
        导致间隙区域的情绪信息丢失。此约束保证相邻窗口始终有重叠覆盖。
        """
        if self.hop_seconds > self.window_seconds:
            raise ValueError("hop_seconds must not exceed window_seconds")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    获取全局配置单例

    使用 functools.lru_cache(maxsize=1) 实现进程级单例模式：
    - 首次调用时创建 Settings 实例（读取 .env 和环境变量）
    - 后续调用直接返回缓存实例，不重复解析配置文件
    - 保证整个进程中所有模块引用同一份配置对象
    - 由于 Settings.frozen=True，缓存实例不会被意外修改

    注意：lru_cache 是基于函数参数的缓存，此函数无参数，
    因此 maxsize=1 即可保证全局唯一。
    """
    return Settings()
