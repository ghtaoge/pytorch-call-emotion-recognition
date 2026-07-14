"""
app.main — FastAPI 应用入口与 NDJSON 流式接口

本模块是整个应用的入口点，负责：
1. 创建 FastAPI 应用实例，配置全局异常处理器
2. 挂载静态文件服务（前端工作台）
3. 提供 REST API 端点：健康检查、模型加载、音频分析
4. 音频分析端点使用 NDJSON（Newline-Delimited JSON）流式传输，
   前端可实时接收进度更新和最终结果

关键设计 — 并发控制：
- 使用 threading.Lock 作为分析锁（analysis_lock），保证同一时刻只有一个分析任务
- acquire(blocking=False) 确保不阻塞等待，新请求立即收到 429 响应
- 锁在 finally 块中释放，即使分析过程中发生异常也不会造成死锁

流式传输设计：
- NDJSON 格式：每行一个 JSON 对象，以换行符分隔
- 三种事件类型：status（初始状态）、progress（分段进度）、result（最终结果）
- 客户端使用 ReadableStream 逐行读取，实现实时进度更新
- 异常事件也以 NDJSON 行发送，客户端可统一处理

安全设计：
- 读取上传内容使用分块限流（read_limited_stream），防止恶意超大文件耗尽内存
- 异常处理器仅返回 AppError 的公开字段（code + public_message），不泄露内部信息
- 默认异常日志不输出堆栈，避免第三方解码异常携带临时路径
"""

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
# 静态文件目录：前端工作台的 HTML/CSS/JS 均在此目录下
STATIC_DIR = Path(__file__).parent / "static"


class Services:
    """
    服务容器 — 聚合模型运行时与分析器，便于依赖注入和测试替换

    正常启动时通过 Services(resolved_settings) 创建真实实例；
    测试时通过 Services 参数注入 FakeRuntime 和 FakeAnalyzer，
    避免在单元测试中下载约 1.1 GB 的真实模型权重。
    """

    def __init__(self, settings: Settings) -> None:
        # 模型运行时：封装模型加载、设备选择和推理调用
        self.runtime = EmotionModelRuntime(settings)
        # 情绪分析器：协调分段推理、四分类投影和加权聚合
        self.analyzer = EmotionAnalyzer(settings, self.runtime)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """
    创建 FastAPI 应用实例 — 工厂模式，支持依赖注入

    参数：
        settings — 全局配置，若未指定则从环境变量/配置文件加载默认值
        services — 服务容器，若未指定则创建真实实例（包含模型加载和推理）
                   测试时可注入 FakeServices 以避免下载真实模型权重

    配置要点：
    - docs_url="/api/docs": Swagger 文档路径，不在根路径以免混淆前端页面
    - redoc_url=None: 禁用 ReDoc 文档，简化部署
    - analysis_lock: threading.Lock，保证同一时刻只有一个分析任务
      acquire(blocking=False) → 新请求立即收到 429 响应，不阻塞等待
    """
    resolved_settings = settings or get_settings()
    resolved_services = services or Services(resolved_settings)
    application = FastAPI(title="声析", docs_url="/api/docs", redoc_url=None)
    application.state.settings = resolved_settings
    application.state.services = resolved_services
    application.state.analysis_lock = threading.Lock()
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.exception_handler(AppError)
    async def handle_app_error(_request: object, exc: AppError) -> JSONResponse:
        """全局异常处理器 — 仅返回 AppError 的公开字段，不泄露内部信息"""
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.public_message}},
        )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """根路径 — 返回前端工作台 HTML 页面"""
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/docs/model-limitations", include_in_schema=False)
    def model_limitations() -> FileResponse:
        """模型限制文档 — 提供模型能力的中文说明"""
        return FileResponse(Path(__file__).parent.parent / "docs" / "model-limitations.md")

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """健康检查端点 — 报告服务状态和模型生命周期，不触发模型加载"""
        runtime = resolved_services.runtime
        return HealthResponse(status="ok", model_status=runtime.status, device=runtime.device)

    @application.post("/api/model/load", response_model=HealthResponse)
    def load_model() -> HealthResponse:
        """主动加载模型端点 — 允许用户在首次分析前预热模型缓存"""
        resolved_services.runtime.load()
        return HealthResponse(
            status="ok",
            model_status=resolved_services.runtime.status,
            device=resolved_services.runtime.device,
        )

    @application.post("/api/analyze")
    def analyze(audio: Annotated[UploadFile, File()]) -> StreamingResponse:
        """
        音频分析端点 — NDJSON 流式传输分析进度与结果

        流程：
        1. 获取分析锁 → 保证同一时刻只有一个分析任务
           若锁已被占用，立即返回 429（ANALYSIS_BUSY）
        2. 分块读取上传内容 → 防止恶意超大文件耗尽内存
        3. FFmpeg 解码音频 → 将压缩格式转为 16kHz 单声道浮点波形
        4. 流式输出分析结果 → 逐段产出 progress 事件，最后产出 result 事件
        5. 释放分析锁 → 在 finally 块中确保锁一定被释放

        异常处理：
        - AppError → 发送 ErrorEvent（仅包含公开错误码和提示）
        - 其他异常 → 发送 ErrorEvent(INTERNAL_ERROR)，日志不输出堆栈
          避免第三方解码异常携带临时路径信息
        """
        # 尝试获取分析锁：blocking=False 确保不阻塞等待，立即返回 429
        if not application.state.analysis_lock.acquire(blocking=False):
            raise AppError("ANALYSIS_BUSY", "已有分析任务正在进行，请稍后重试", 429)
        try:
            # 分块读取上传内容，防止恶意超大文件耗尽内存
            data = read_limited_stream(audio.file, resolved_settings.max_bytes)
            # FFmpeg 解码：将压缩格式转为 16kHz 单声道浮点波形
            decoded = decode_audio(data, audio.filename, resolved_settings)
        except Exception:
            application.state.analysis_lock.release()
            raise

        def stream() -> Iterator[str]:
            """流式输出生成器 — 逐段产出 NDJSON 行，前端可实时读取进度"""
            try:
                # 逐段产出分析事件：status → progress... → result
                for event in resolved_services.analyzer.iter_analysis(decoded):
                    yield event.model_dump_json() + "\n"
            except AppError as exc:
                # 应用层异常：仅记录错误码，不输出堆栈到日志
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
                # 确保分析锁一定被释放，即使发生异常也不会造成死锁
                application.state.analysis_lock.release()

        # NDJSON 格式：每行一个 JSON 对象，前端使用 ReadableStream 逐行读取
        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return application


# 模块级应用实例 — uvicorn 通过 app.main:app 引用此对象启动服务
app = create_app()
