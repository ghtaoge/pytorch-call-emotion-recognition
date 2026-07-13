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
