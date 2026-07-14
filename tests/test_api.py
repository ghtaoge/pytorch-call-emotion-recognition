"""
test_api —— FastAPI 端点的单元测试

本模块测试应用的 HTTP 端点行为，包括：
- /api/health：健康检查端点，不应触发模型加载
- /api/analyze：流式分析端点，应返回 NDJSON 格式的进度与结果
- /：前端工作台页面，应包含关键品牌标识

使用 FakeServices 模式替代真实服务依赖：
- FakeRuntime 模拟模型加载与推理
- FakeAnalyzer 模拟分析流程的事件流
- FakeServices 将上述伪装对象组装为依赖注入容器
"""

from collections.abc import Iterator

import numpy as np
from fastapi.testclient import TestClient

import app.main as main_module
from app.audio import DecodedAudio
from app.config import Settings
from app.schemas import (
    AnalysisResult,
    EmotionProbabilities,
    ProgressEvent,
    Reliability,
    ResultEvent,
)


class FakeRuntime:
    """伪造的模型运行时，模拟加载状态变化。

    - 初始状态为 "not_loaded"，调用 load() 后变为 "loaded"
    - device 固定为 "cpu"
    """

    status = "not_loaded"
    device = "cpu"

    def load(self) -> None:
        # 模拟模型加载过程：将状态从 "not_loaded" 更改为 "loaded"
        self.status = "loaded"


class FakeAnalyzer:
    """伪造的情感分析器，模拟 iter_analysis 的流式输出。

    依次产生两个事件：
    1. ProgressEvent —— 表示正在分析第 1/1 段
    2. ResultEvent   —— 包含完整的分析结果（dominant_emotion=neutral）
    """

    def iter_analysis(self, _audio: DecodedAudio) -> Iterator[ProgressEvent | ResultEvent]:
        # 首先发送进度事件，表示正在处理音频段
        yield ProgressEvent(type="progress", current=1, total=1, message="正在分析第 1/1 段")
        # 然后发送结果事件，包含完整的情感分析结果
        yield ResultEvent(
            result=AnalysisResult(
                dominant_emotion="neutral",  # 主导情感为 neutral
                probabilities=EmotionProbabilities(neutral=0.7, happy=0.1, anger=0.1, sad=0.1),  # 四类概率分布
                reliability=Reliability(level="high", reasons=[]),  # 可靠性等级为 high
                excluded_probability=0.05,  # 被排除的概率为 5%
                voiced_ratio=1.0,  # 有声占比为 100%
                duration_seconds=1.0,  # 音频时长 1 秒
                device="cpu",  # 推理设备为 CPU
                elapsed_ms=10,  # 推理耗时 10ms
                segments=[],  # 无分段详情
            )
        )


class FakeServices:
    """伪造的服务集合，将 FakeRuntime 与 FakeAnalyzer 组合为依赖注入对象。

    用于 create_app() 的 services 参数，替代真实的生产服务。
    """

    runtime = FakeRuntime()
    analyzer = FakeAnalyzer()


def test_health_does_not_load_model() -> None:
    """验证健康检查端点不会触发模型加载。

    访问 /api/health 时，应用应返回当前状态而不调用 runtime.load()。
    响应 JSON 应包含：
    - status: "ok"
    - model_status: "not_loaded"（因为未触发加载）
    - device: "cpu"
    """
    services = FakeServices()
    client = TestClient(main_module.create_app(Settings(), services))  # type: ignore[arg-type]
    response = client.get("/api/health")
    assert response.status_code == 200  # 健康检查应返回 200
    # 验证响应体包含正确的状态信息，且模型状态为 not_loaded
    assert response.json() == {"status": "ok", "model_status": "not_loaded", "device": "cpu"}


def test_analyze_stream_returns_progress_and_result(monkeypatch) -> None:
    """验证流式分析端点返回 NDJSON 格式的进度与结果事件。

    使用 monkeypatch 替换 decode_audio 函数，避免真实音频解码。
    发送 POST /api/analyze 请求后，应收到：
    - Content-Type 为 application/x-ndjson
    - 第一行包含 progress 类型事件
    - 最后一行包含 result 类型事件
    """
    # 用 monkeypatch 替换音频解码函数，返回固定的合成音频
    monkeypatch.setattr(
        main_module,
        "decode_audio",
        lambda _data, _filename, _settings: DecodedAudio(np.ones(16000, dtype=np.float32), 16000),
    )
    client = TestClient(main_module.create_app(Settings(), FakeServices()))  # type: ignore[arg-type]
    # 使用流式请求方式发送音频文件
    with client.stream(
        "POST", "/api/analyze", files={"audio": ("sample.wav", b"synthetic", "audio/wav")}
    ) as response:
        lines = list(response.iter_lines())
    assert response.status_code == 200  # 分析请求应返回 200
    assert response.headers["content-type"].startswith("application/x-ndjson")  # 应为 NDJSON 格式
    assert '"type":"progress"' in lines[0]  # 首行应为进度事件
    assert '"type":"result"' in lines[-1]  # 末行应为结果事件


def test_static_workbench_is_served() -> None:
    """验证前端工作台页面可正常访问，并包含品牌标识。

    访问 / 端点时应返回 200 状态码，且页面 HTML 中包含"声析"品牌关键词。
    """
    client = TestClient(main_module.create_app(Settings(), FakeServices()))  # type: ignore[arg-type]
    response = client.get("/")
    assert response.status_code == 200  # 页面应可正常访问
    assert "声析" in response.text  # 页面应包含品牌名称
