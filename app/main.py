from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import EmotionAnalyzer
from app.audio import decode_audio, read_limited_stream
from app.config import Settings, get_settings
from app.errors import AppError
from app.model import EmotionModelRuntime
from app.schemas import ErrorEvent, HealthResponse, PublicError

logger = logging.getLogger("emotion_app")
STATIC_DIR = Path(__file__).parent / "static"


class Services:
    def __init__(self, settings: Settings) -> None:
        self.runtime = EmotionModelRuntime(settings)
        self.analyzer = EmotionAnalyzer(settings, self.runtime)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_services = services or Services(resolved_settings)
    application = FastAPI(title="声析", docs_url="/api/docs", redoc_url=None)
    application.state.settings = resolved_settings
    application.state.services = resolved_services
    application.state.analysis_lock = threading.Lock()
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.exception_handler(AppError)
    async def handle_app_error(_request: object, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.public_message}},
        )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/docs/model-limitations", include_in_schema=False)
    def model_limitations() -> FileResponse:
        return FileResponse(Path(__file__).parent.parent / "docs" / "model-limitations.md")

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        runtime = resolved_services.runtime
        return HealthResponse(status="ok", model_status=runtime.status, device=runtime.device)

    @application.post("/api/model/load", response_model=HealthResponse)
    def load_model() -> HealthResponse:
        resolved_services.runtime.load()
        return HealthResponse(
            status="ok",
            model_status=resolved_services.runtime.status,
            device=resolved_services.runtime.device,
        )

    @application.post("/api/analyze")
    def analyze(audio: Annotated[UploadFile, File()]) -> StreamingResponse:
        if not application.state.analysis_lock.acquire(blocking=False):
            raise AppError("ANALYSIS_BUSY", "已有分析任务正在进行，请稍后重试", 429)
        try:
            data = read_limited_stream(audio.file, resolved_settings.max_bytes)
            decoded = decode_audio(data, audio.filename, resolved_settings)
        except Exception:
            application.state.analysis_lock.release()
            raise

        def stream() -> Iterator[str]:
            try:
                for event in resolved_services.analyzer.iter_analysis(decoded):
                    yield event.model_dump_json() + "\n"
            except AppError as exc:
                logger.warning("analysis_failed code=%s", exc.code)
                yield (
                    ErrorEvent(
                        type="error",
                        error=PublicError(code=exc.code, message=exc.public_message),
                    ).model_dump_json()
                    + "\n"
                )
            except Exception:
                # 默认日志不输出堆栈，避免第三方解码异常携带临时路径。
                logger.error("analysis_failed code=INTERNAL_ERROR")
                yield (
                    ErrorEvent(
                        type="error",
                        error=PublicError(code="INTERNAL_ERROR", message="服务暂时不可用，请重试"),
                    ).model_dump_json()
                    + "\n"
                )
            finally:
                application.state.analysis_lock.release()

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return application


app = create_app()
