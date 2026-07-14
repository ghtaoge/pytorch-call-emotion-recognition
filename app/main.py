"""
app.main — FastAPI 应用入口与 NDJSON 流式接口

本模块是整个应用的入口点，负责：
1. 创建 FastAPI 应用实例，配置全局异常处理器
2. 挂载静态文件服务（前端工作台）
3. 提供 REST API 端点：健康检查、模型加载、音频分析
4. 音频分析端点使用 NDJSON（Newline-Delimited JSON）流式传输，
   前端可实时接收进度更新和最终结果

应用生命周期：
1. create_app() 创建 FastAPI 实例，配置路由与异常处理器
2. Services 初始化 EmotionModelRuntime 和 EmotionAnalyzer
   注意：此时模型尚未加载（延迟加载策略），服务启动不会立即下载约 1.1 GB 权重
3. 模块级 app = create_app() 创建全局应用实例，供 uvicorn 启动
4. 首次调用 /api/analyze 或 /api/model/load 时触发模型加载

关键设计 — 并发控制：
- 使用 threading.Lock 作为分析锁（analysis_lock），保证同一时刻只有一个分析任务
  原因：模型推理（特别是 GPU 推理）对资源消耗极大，并发推理会导致：
  - GPU 内存溢出（CUDA OOM）
  - 推理速度大幅下降
  - 结果质量不可预测
- acquire(blocking=False) 确保不阻塞等待，新请求立即收到 429 响应
  这比阻塞等待更友好：调用方可以立即知道需要稍后重试，而非长时间等待无响应
- 锁在 finally 块中释放，即使分析过程中发生异常也不会造成死锁
  具体实现：读取阶段的异常在 except 中手动释放，流式阶段的异常在 finally 中自动释放

流式传输设计（NDJSON）：
- NDJSON 格式：每行一个 JSON 对象，以换行符（\\n）分隔
- 三种事件类型：status（初始状态）、progress（分段进度）、result（最终结果）
- 错误事件：error（分析过程中的异常，也以 NDJSON 行发送）
- 客户端使用 ReadableStream 逐行读取，实现实时进度更新
- 优势：相比 WebSocket 更简单（无需双向通信），相比一次性响应更实时

安全设计：
- 读取上传内容使用分块限流（read_limited_stream），防止恶意超大文件耗尽内存
- 异常处理器仅返回 AppError 的公开字段（code + public_message），不泄露内部信息
- 默认异常日志不输出堆栈，避免第三方解码异常携带临时路径
- 分析锁防止并发推理导致的资源竞争与内存溢出
"""

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

# 应用级日志器：用于记录分析过程中的警告和错误
# 默认不输出堆栈信息，避免第三方库异常（如 FFmpeg）携带临时文件路径
logger = logging.getLogger("emotion_app")
# 静态文件目录：前端工作台的 HTML/CSS/JS 均在此目录下
STATIC_DIR = Path(__file__).parent / "static"


class Services:
    """
    服务容器 — 聚合模型运行时与分析器，便于依赖注入和测试替换

    设计意图：
    - 将所有业务服务的创建逻辑集中在一个类中，避免散落在路由函数中
    - 支持依赖注入：测试时可通过 Services 参数注入 FakeRuntime 和 FakeAnalyzer，
      避免在单元测试中下载约 1.1 GB 的真实模型权重
    - 正常启动时通过 Services(resolved_settings) 创建真实实例

    生命周期：
    - Services 在 create_app() 中创建，绑定到 application.state.services
    - EmotionModelRuntime 在此时仅初始化（记录配置和设备选择），不加载模型
    - 模型加载发生在首次调用 predict() 或 /api/model/load 时

    字段说明：
    - runtime : 模型运行时管理器，封装延迟加载、设备选择与推理调用
    - analyzer : 情绪分析协调器，封装分段推理、概率投影与加权聚合
    """

    def __init__(self, settings: Settings) -> None:
        """
        创建服务容器

        参数：
            settings — 全局配置对象，传递给 runtime 和 analyzer

        注意：此方法不触发模型加载，仅创建运行时和分析器实例。
        模型加载在首次推理或 /api/model/load 端点调用时触发。
        """
        # 模型运行时：封装模型加载、设备选择和推理调用
        self.runtime = EmotionModelRuntime(settings)
        # 情绪分析器：协调分段推理、四分类投影和加权聚合
        self.analyzer = EmotionAnalyzer(settings, self.runtime)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """
    创建 FastAPI 应用实例 — 工厂模式，支持依赖注入

    此函数是应用的核心入口点，负责：
    1. 创建 FastAPI 实例并配置基本属性（标题、文档路径）
    2. 将 Settings、Services 和分析锁绑定到 application.state
    3. 注册全局异常处理器（AppError → JSONResponse）
    4. 挂载静态文件服务（前端工作台）
    5. 注册所有 API 路由

    工厂模式的优势：
    - 支持依赖注入：测试时可传入 FakeSettings 和 FakeServices
    - 支持多实例：理论上可创建多个应用实例（但当前设计为单实例）
    - 隔离配置：每次调用 create_app() 都使用独立的 Settings 和 Services

    参数：
        settings — 全局配置，若未指定则从环境变量/配置文件加载默认值
                   通过 get_settings() 获取缓存的单例实例
        services — 服务容器，若未指定则创建真实实例（包含模型加载和推理）
                   测试时可注入 FakeServices 以避免下载真实模型权重

    配置要点：
    - title="声析": 应用标题，显示在 Swagger 文档页面
    - docs_url="/api/docs": Swagger 文档路径，不在根路径以免混淆前端页面
    - redoc_url=None: 禁用 ReDoc 文档，简化部署
    - analysis_lock: threading.Lock，保证同一时刻只有一个分析任务
    """
    # 配置优先级：显式传入 > 默认值（从 .env 和环境变量加载）
    resolved_settings = settings or get_settings()
    # 服务优先级：显式传入 > 默认值（创建真实实例）
    resolved_services = services or Services(resolved_settings)
    # 创建 FastAPI 实例
    application = FastAPI(title="声析", docs_url="/api/docs", redoc_url=None)
    # 将配置和服务绑定到 application.state，供路由函数访问
    application.state.settings = resolved_settings
    application.state.services = resolved_services
    # 分析锁：保证同一时刻只有一个分析任务
    # threading.Lock 是非重入锁，同一线程也不能重复获取（与 RLock 不同）
    # 此设计符合业务需求：同一时刻只允许一个分析请求
    application.state.analysis_lock = threading.Lock()
    # 挂载静态文件服务：前端工作台的 HTML/CSS/JS 文件
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---------- 全局异常处理器 ----------
    @application.exception_handler(AppError)
    async def handle_app_error(_request: object, exc: AppError) -> JSONResponse:
        """
        全局异常处理器 — 仅返回 AppError 的公开字段，不泄露内部信息

        此处理器拦截所有 AppError 异常，将其转换为标准 JSON 格式：
        {"error": {"code": "...", "message": "..."}}

        安全设计关键：
        - 仅使用 exc.code 和 exc.public_message，不使用 Exception.message
          Exception.message 可能包含内部信息（如临时文件路径）
        - 不输出堆栈信息到响应体，避免泄露源代码结构
        - exc.status_code 直接映射为 HTTP 状态码，保证语义正确

        参数：
            _request — 当前请求对象（不使用，仅占位）
            exc — AppError 实例，携带 code/public_message/status_code
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.public_message}},
        )

    # ---------- 路由定义 ----------

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """
        根路径 — 返回前端工作台 HTML 页面

        include_in_schema=False：不在 Swagger 文档中显示此端点，
        因为它是前端页面而非 API 端点。
        """
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/docs/model-limitations", include_in_schema=False)
    def model_limitations() -> FileResponse:
        """
        模型限制文档 — 提供模型能力的中文说明

        此端点返回 docs/model-limitations.md 文件，
        供前端工作台的"模型限制"页面引用。
        include_in_schema=False：不在 Swagger 文档中显示此端点。
        """
        return FileResponse(Path(__file__).parent.parent / "docs" / "model-limitations.md")

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """
        健康检查端点 — 报告服务状态和模型生命周期，不触发模型加载

        此端点返回当前服务的健康状态和模型加载状态，
        供运维监控和前端状态展示使用。

        关键特性：
        - 不触发模型加载：仅读取 runtime.status 和 runtime.device，
          不会因为健康检查而触发约 1.1 GB 的模型下载
        - status 固定为 "ok"：只要端点能响应就说明服务运行正常
        - model_status 反映模型生命周期："not_loaded" / "loading" / "loaded" / "error"
        """
        runtime = resolved_services.runtime
        return HealthResponse(status="ok", model_status=runtime.status, device=runtime.device)

    @application.post("/api/model/load", response_model=HealthResponse)
    def load_model() -> HealthResponse:
        """
        主动加载模型端点 — 允许用户在首次分析前预热模型缓存

        此端点触发模型加载（下载约 1.1 GB 权重），
        适合在首次分析前调用以避免首次分析的长时间等待。

        设计意图：
        - 服务启动时不立即加载模型（延迟加载策略）
        - 用户可通过此端点主动触发加载，实现"预热"
        - 加载完成后 model_status 变为 "loaded"，后续分析无需等待

        注意：此端点不受分析锁限制，因为加载过程与推理过程独立。
        EmotionModelRuntime.load() 使用双重检查锁定保证只加载一次。
        """
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

        此端点是核心业务接口，接收上传的音频文件并返回 NDJSON 流式响应。

        流程：
        1. 获取分析锁 → 保证同一时刻只有一个分析任务
           若锁已被占用，立即返回 429（ANALYSIS_BUSY）
           acquire(blocking=False) 确保不阻塞等待
        2. 分块读取上传内容 → read_limited_stream 防止恶意超大文件耗尽内存
        3. FFmpeg 解码音频 → decode_audio 将压缩格式转为 16kHz 单声道浮点波形
           读取和解码阶段若发生异常，手动释放锁后抛出异常
        4. 流式输出分析结果 → 逐段产出 progress 事件，最后产出 result 事件
           流式阶段若发生异常，在 finally 块中自动释放锁
        5. 释放分析锁 → 在 finally 块中确保锁一定被释放

        锁的释放策略（两阶段）：
        - 读取/解码阶段：在 except 块中手动释放（因为此时 stream() 尚未创建）
        - 流式输出阶段：在 stream() 的 finally 块中自动释放
        此策略保证无论在哪个阶段发生异常，锁都会被释放，不会造成死锁。

        NDJSON 事件序列：
        - ProgressEvent(type="status", ...) — 模型准备中
        - ProgressEvent(type="progress", ...) — 分段进度（重复 N 次）
        - ResultEvent(type="result", ...) — 最终分析结果
        - ErrorEvent(type="error", ...) — 异常事件（仅发生异常时）

        异常处理：
        - AppError → 发送 ErrorEvent（仅包含公开错误码和提示）
          日志仅记录错误码，不输出堆栈（exc.code）
        - 其他异常 → 发送 ErrorEvent(INTERNAL_ERROR)
          日志仅记录 "INTERNAL_ERROR"，不输出堆栈
          原因：第三方库异常（如 FFmpeg subprocess）可能携带临时文件路径等敏感信息
        """
        # 尝试获取分析锁：blocking=False 确保不阻塞等待，立即返回 429
        # 若锁已被占用（另一个分析任务正在进行），直接抛出 429 错误
        if not application.state.analysis_lock.acquire(blocking=False):
            raise AppError("ANALYSIS_BUSY", "已有分析任务正在进行，请稍后重试", 429)
        try:
            # 分块读取上传内容，防止恶意超大文件耗尽内存
            # max_bytes 来自配置（默认 50 MB），read_limited_stream 实时累加检查
            data = read_limited_stream(audio.file, resolved_settings.max_bytes)
            # FFmpeg 解码：将压缩格式转为 16kHz 单声道浮点波形
            # 此步骤可能抛出 AppError（格式不支持、解码失败、文件过大等）
            decoded = decode_audio(data, audio.filename, resolved_settings)
        except Exception:
            # 读取/解码阶段异常：手动释放锁，因为 stream() 尚未创建
            # 若不在此释放，锁将永远被占用（死锁），因为后续的 finally 块在 stream() 中
            application.state.analysis_lock.release()
            raise

        def stream() -> Iterator[str]:
            """
            流式输出生成器 — 逐段产出 NDJSON 行，前端可实时读取进度

            此函数是 StreamingResponse 的内容生成器，
            每次迭代产出一行 NDJSON（JSON 对象 + 换行符）。

            事件序列：
            1. ProgressEvent(type="status", current=0, total=N) — 模型准备中
            2. ProgressEvent(type="progress", current=1..N, total=N) — 分段进度
            3. ResultEvent(type="result", ...) — 最终分析结果
            4. ErrorEvent(type="error", ...) — 异常事件（仅发生异常时）

            异常处理策略：
            - AppError：发送 ErrorEvent，仅包含公开字段（code + public_message）
              日志仅记录错误码（exc.code），不输出堆栈，避免泄露内部信息
            - 其他异常：发送 ErrorEvent(INTERNAL_ERROR)
              日志仅记录 "INTERNAL_ERROR"，不输出堆栈
              原因：第三方库异常可能携带临时文件路径等敏感信息
            - finally 块：确保分析锁一定被释放，即使发生异常也不会造成死锁
            """
            try:
                # 逐段产出分析事件：status → progress... → result
                # iter_analysis 是生成器函数，每次 yield 一个事件
                for event in resolved_services.analyzer.iter_analysis(decoded):
                    # 每个事件序列化为 JSON 并追加换行符，形成 NDJSON 格式
                    yield event.model_dump_json() + "\n"
            except AppError as exc:
                # 应用层异常：仅记录错误码，不输出堆栈到日志
                # exc.code 供运维排查，exc.public_message 供前端展示
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
                # 第三方库（如 torch, FFmpeg subprocess）的异常可能包含：
                # - 临时文件路径（如 C:\\Users\\...\\tmp_audio.wav）
                # - 模型缓存路径（如 ~/.cache/huggingface/...）
                # 这些信息不应出现在日志或响应中
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
                # 此 release() 对应上方 acquire(blocking=False) 的获取
                application.state.analysis_lock.release()

        # 返回 NDJSON 流式响应
        # media_type="application/x-ndjson" 是 NDJSON 的标准 MIME 类型
        # 前端使用 ReadableStream 逐行读取，实现实时进度更新
        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return application


# 模块级应用实例 — uvicorn 通过 app.main:app 引用此对象启动服务
# create_app() 在模块导入时执行，创建全局唯一的 FastAPI 实例
# 此时 Services 和 EmotionModelRuntime 已初始化，但模型尚未加载（延迟加载策略）
app = create_app()
