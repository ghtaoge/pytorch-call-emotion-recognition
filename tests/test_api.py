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
    status = "not_loaded"
    device = "cpu"

    def load(self) -> None:
        self.status = "loaded"


class FakeAnalyzer:
    def iter_analysis(self, _audio: DecodedAudio) -> Iterator[ProgressEvent | ResultEvent]:
        yield ProgressEvent(type="progress", current=1, total=1, message="正在分析第 1/1 段")
        yield ResultEvent(
            result=AnalysisResult(
                dominant_emotion="neutral",
                probabilities=EmotionProbabilities(neutral=0.7, happy=0.1, anger=0.1, sad=0.1),
                reliability=Reliability(level="high", reasons=[]),
                excluded_probability=0.05,
                voiced_ratio=1.0,
                duration_seconds=1.0,
                device="cpu",
                elapsed_ms=10,
                segments=[],
            )
        )


class FakeServices:
    runtime = FakeRuntime()
    analyzer = FakeAnalyzer()


def test_health_does_not_load_model() -> None:
    services = FakeServices()
    client = TestClient(main_module.create_app(Settings(), services))  # type: ignore[arg-type]
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_status": "not_loaded", "device": "cpu"}


def test_analyze_stream_returns_progress_and_result(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "decode_audio",
        lambda _data, _filename, _settings: DecodedAudio(np.ones(16000, dtype=np.float32), 16000),
    )
    client = TestClient(main_module.create_app(Settings(), FakeServices()))  # type: ignore[arg-type]
    with client.stream(
        "POST", "/api/analyze", files={"audio": ("sample.wav", b"synthetic", "audio/wav")}
    ) as response:
        lines = list(response.iter_lines())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert '"type":"progress"' in lines[0]
    assert '"type":"result"' in lines[-1]


def test_static_workbench_is_served() -> None:
    client = TestClient(main_module.create_app(Settings(), FakeServices()))  # type: ignore[arg-type]
    response = client.get("/")
    assert response.status_code == 200
    assert "声析" in response.text
