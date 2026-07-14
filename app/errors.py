"""
app.errors — 应用统一异常类

本模块定义了唯一的异常类 AppError，用于整个应用中的结构化错误处理。
所有业务异常均通过 AppError 抛出，保证错误信息的统一格式与安全性。

核心设计原则 — 安全错误报告：
1. 仅携带可安全返回给调用方的字段（code, public_message, status_code）
2. 不泄露内部信息：
   - 不包含原始异常堆栈（仅通过 raise ... from exc 保留链式关系供日志排查）
   - 不包含用户原始输入（音频文件名、上传数据）
   - 不包含内部路径（临时文件路径、模型缓存路径）
3. code 字段为机器可读的错误标识（如 "NO_VOICE", "FILE_TOO_LARGE"），
   供前端逻辑分支判断；public_message 为人类可读的中文提示，供前端展示

错误分类（按 status_code）：
- 400 (Bad Request)        : 客户端输入问题（空文件、无效音频）
- 408 (Request Timeout)    : 解码超时
- 413 (Payload Too Large)  : 文件过大、音频过长
- 415 (Unsupported Media)  : 不支持的音频格式
- 422 (Unprocessable)      : 业务逻辑拒绝（无语音、模型输出异常）
- 429 (Too Many Requests)  : 并发分析限制
- 500 (Internal Error)     : 服务内部异常（推理失败）
- 503 (Service Unavailable) : 模型加载失败、设备内存不足

异常传播路径：
AppError → main.py handle_app_error → JSONResponse({error: {code, message}})
AppError → main.py stream() → ErrorEvent({type: "error", error: {code, message}})

两条路径均仅提取 code + public_message + status_code，
不传播 Exception 的默认 message（可能包含内部信息）。
"""


class AppError(Exception):
    """
    应用统一异常类 — 结构化、安全的错误报告

    此类是整个应用中唯一的业务异常类型，所有错误场景均通过 AppError 报告。
    继承 Exception 以保证可被标准 try/except 捕获，
    同时携带结构化字段供 HTTP 层提取和转换。

    字段说明：
    - code           : 机器可读的错误代码标识，供前端逻辑分支判断
                       例："NO_VOICE", "FILE_TOO_LARGE", "MODEL_LOAD_FAILED"
                       所有 code 均为 UPPER_SNAKE_CASE 格式，保证可读性与一致性
    - public_message : 人类可读的中文错误提示，可安全返回给调用方
                       仅包含用户需要知道的信息（如"未检测到清晰人声"），
                       不包含内部技术细节（如临时文件路径、模型配置）
    - status_code    : HTTP 状态码，用于 JSONResponse 的状态码设置
                       默认为 400，特定场景使用其他状态码（如 503 表示模型不可用）

    线程安全：此类是不可变的（所有字段在 __init__ 中设置后不再修改），
    可安全地在多线程环境中传递和捕获。

    生命周期：
    1. 业务层创建 AppError 并抛出
    2. HTTP 层捕获并转换为 JSONResponse 或 ErrorEvent
    3. 调用方收到结构化错误响应

    安全设计关键：
    - 仅保存可安全返回给调用方的字段, 避免泄露原始输入或内部路径。
    - Exception 的默认 message（通过 super().__init__(message) 设置）
      仅用于日志排查，不直接返回给调用方
    - HTTP 层使用 public_message 而非 Exception.message 构建响应

    错误码命名约定：
    - 全大写蛇形命名（如 ANALYSIS_BUSY、NO_VOICE）
    - 前缀暗示错误来源：FILE_* → 文件校验, DECODE_* → 音频解码,
      MODEL_* → 模型加载, INFERENCE_* → 推理过程
    """

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        """
        创建应用异常

        参数：
            code        : 机器可读的错误代码标识（UPPER_SNAKE_CASE 格式）
                          例："NO_VOICE", "FILE_TOO_LARGE", "INFERENCE_FAILED"
            message     : 人类可读的中文错误提示（可安全公开）
                          例："未检测到清晰人声", "音频文件不能超过 50 MB"
            status_code : HTTP 状态码，默认 400（Bad Request）
                          特定场景使用其他状态码：
                          - 408: 解码超时
                          - 413: 文件过大/音频过长
                          - 415: 不支持的格式
                          - 422: 业务逻辑拒绝
                          - 429: 并发限制
                          - 500: 内部异常
                          - 503: 服务不可用（模型加载失败）
        """
        # Exception.message 仅用于日志排查，不直接返回给调用方
        super().__init__(message)
        # 仅保存可安全返回给调用方的字段, 避免泄露原始输入或内部路径。
        self.code = code
        self.public_message = message
        self.status_code = status_code
