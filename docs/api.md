# 本地 API

## `GET /api/health`

返回服务、模型和设备状态，不触发模型下载。

## `POST /api/model/load`

显式加载模型。首次调用可能下载约 1.1 GB 权重。

## `POST /api/analyze`

字段 `audio` 为 multipart 文件。响应类型是 `application/x-ndjson`，每行一个 JSON 事件：

```json
{"type":"progress","current":2,"total":8,"message":"正在分析第 2/8 段"}
{"type":"result","result":{"dominant_emotion":"neutral","probabilities":{"neutral":0.62,"happy":0.11,"anger":0.15,"sad":0.12}}}
```

错误码包括 `EMPTY_FILE`、`FILE_TOO_LARGE`、`UNSUPPORTED_FORMAT`、`DECODE_FAILED`、`DECODE_TIMEOUT`、`AUDIO_TOO_LONG`、`NO_VOICE`、`MODEL_LOAD_FAILED`、`INFERENCE_FAILED` 和 `ANALYSIS_BUSY`。响应不包含文件名、路径、音频或堆栈。

## `POST /api/analyze-url`

从 URL 下载音频并执行情绪分析，返回 NDJSON 流式响应。

**请求：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `url` | string | 音频文件 URL，必须以 `http://` 或 `https://` 开头 |

**请求示例：**

```json
{ "url": "https://example.com/call-audio.wav" }
```

**响应：** NDJSON 流式输出，格式与 `/api/analyze` 完全相同。

**新增错误码：**

| 错误码 | HTTP 状态码 | 说明 |
|--------|------------|------|
| `INVALID_URL` | 400 | URL 格式不合法（协议非 http/https） |
| `URL_DOWNLOAD_FAILED` | 400 | 音频下载失败（网络错误、404 等） |
| `URL_DOWNLOAD_TIMEOUT` | 408 | 音频下载超时 |
| `URL_FILE_TOO_LARGE` | 413 | 下载文件超过 50 MB 限制 |
| `ANALYSIS_BUSY` | 429 | 已有分析任务正在进行 |

分析阶段的错误复用现有错误码。
