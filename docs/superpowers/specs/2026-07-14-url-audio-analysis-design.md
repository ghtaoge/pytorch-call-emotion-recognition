# URL 音频分析功能设计

> 日期: 2026-07-14
> 状态: 待审核

## 1. 目标

新增通过 URL 地址获取音频文件进行情绪分析的能力。用户可直接输入音频 URL，服务端下载后复用现有分析流程，返回相同的 NDJSON 流式结果。

**核心原则：最小侵入。** 音频解码、分段、推理、聚合流程完全复用，新增下载函数和独立端点，前端改动最小。

## 2. 范围

- 后端新增 `fetch_audio_from_url()` 下载函数
- 后端新增 `POST /api/analyze-url` 端点
- 后端新增 `AnalyzeUrlRequest` schema 和相关错误码
- 后端新增 `httpx` 依赖
- 后端新增 URL 相关配置项
- 前端新增"URL 地址"选项卡和输入框
- 前端复用 NDJSON 解析和结果渲染逻辑
- 测试覆盖新增模块

## 3. 后端设计

### 3.1 URL 音频下载 — `app/audio.py`

新增 `fetch_audio_from_url()` 函数：

```python
def fetch_audio_from_url(url: str, settings: Settings) -> tuple[bytes, str]:
```

**参数：**
- `url` — 音频文件 URL
- `settings` — 全局配置，提供 `max_bytes` 和 URL 相关配置

**返回：**
- `(bytes, inferred_filename)` — bytes 交给现有 `decode_audio()` 处理

**下载实现：**
- 使用 `httpx.Client` 同步下载（服务端是同步推理，不需要异步）
- 流式下载：使用 `httpx.stream()` 逐块读取，实时检查大小不超过 `max_bytes`
- 超时配置：连接 10 秒，读取 `url_download_timeout_seconds`（默认 60 秒）
- 重定向限制：最多 `url_max_redirects`（默认 5 次）

**filename 推断逻辑：**
1. 优先从 `Content-Disposition` header 的 `filename*` / `filename` 提取
2. 其次从 URL 路径最后一段提取
3. 兜底使用 `"downloaded_audio"` 作为 filename

**安全设计：**
- 协议白名单：仅允许 `http://` 和 `https://`，拒绝 `file://`、`ftp://` 等
- 不过滤私有 IP 地址（需求明确支持内网 URL）
- 下载大小复用 `max_bytes`（50 MB），流式检查

**错误处理：**
- URL 协议不合法 → `AppError("INVALID_URL", 400)`
- 下载失败（网络错误、404 等）→ `AppError("URL_DOWNLOAD_FAILED", 400)`
- 下载超时 → `AppError("URL_DOWNLOAD_TIMEOUT", 408)`
- 下载文件过大 → `AppError("URL_FILE_TOO_LARGE", 413)`
- Content-Type 非音频 → 不强制拦截，由 `decode_audio()` 的格式校验处理

### 3.2 API 端点 — `app/main.py`

新增 `POST /api/analyze-url` 端点：

```python
@application.post("/api/analyze-url")
def analyze_url(request: AnalyzeUrlRequest) -> StreamingResponse:
```

**请求模型（新增 schema）：**

```python
class AnalyzeUrlRequest(ContractModel):
    url: str  # 音频文件 URL，必须以 http:// 或 https:// 开头

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value
```

**响应格式：** 与 `/api/analyze` 完全相同 — NDJSON 流式输出

**端点逻辑流程：**
1. 获取分析锁 → 429 if busy（同现有端点）
2. 校验 URL 格式（AnalyzeUrlRequest 自动校验）
3. 调用 `fetch_audio_from_url()` 下载音频
4. 调用 `decode_audio()` 解码（复用现有流程）
5. 调用 `analyzer.iter_analysis()` 流式分析（复用）
6. 释放锁（finally 块，两阶段策略同现有端点）

**锁释放策略：**
- 下载/解码阶段异常 → except 块手动释放
- 流式阶段异常 → finally 块自动释放
- 与现有 `/api/analyze` 完全一致

### 3.3 配置扩展 — `app/config.py`

新增配置项：

| 字段 | 默认值 | 类型约束 | 说明 |
|------|--------|----------|------|
| `url_download_timeout_seconds` | 60.0 | PositiveFloat | URL 音频下载总超时 |
| `url_max_redirects` | 5 | PositiveInt | URL 重定向最大次数 |

### 3.4 错误码

新增错误码：

| 错误码 | HTTP 状态码 | 说明 |
|--------|------------|------|
| `INVALID_URL` | 400 | URL 格式不合法 |
| `URL_DOWNLOAD_FAILED` | 400 | 下载失败 |
| `URL_DOWNLOAD_TIMEOUT` | 408 | 下载超时 |
| `URL_FILE_TOO_LARGE` | 413 | 下载文件超过大小限制 |

分析阶段的错误复用现有错误码（NO_VOICE, ANALYSIS_BUSY, INFERENCE_FAILED 等）。

### 3.5 依赖新增

`httpx` 加入 `pyproject.toml` 的 dependencies。选择 httpx 而非 requests 的原因：
- httpx 支持 HTTP/2 和更现代的超时控制
- httpx 的流式下载 API 更简洁
- httpx 的重定向控制更灵活

## 4. 前端设计

### 4.1 分段控件扩展

现有分段控件有"上传音频"和"现场录音"，新增第三项"URL 地址"。

HTML 结构（`app/static/index.html`）：

```html
<div class="segmented-control">
  <button data-mode="upload" class="active">上传音频</button>
  <button data-mode="record">现场录音</button>
  <button data-mode="url">URL 地址</button>
</div>
```

### 4.2 URL 输入界面

URL 选项卡激活时显示：

- 文本输入框，placeholder: "输入音频文件 URL 地址"
- 实时校验：URL 必须以 `http://` 或 `https://` 开头，否则"开始分析"按钮禁用
- 输入框下方提示：支持格式 wav, mp3, flac, ogg, m4a, webm
- "开始分析"按钮（复用现有按钮样式和位置）

### 4.3 API 调用

URL 模式下的分析调用：

```javascript
const response = await fetch("/api/analyze-url", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ url: state.url }),
});
```

- 响应读取方式与上传分析完全相同（ReadableStream → NDJSON 解析）
- 进度展示、结果渲染完全复用 `handleEvent()` 和 `renderResult()` 逻辑

### 4.4 状态管理

- `state.url` 新增字段，存储当前输入的 URL
- URL 输入激活时隐藏上传/录音面板，反之亦然
- 切换选项卡时清空其他输入状态（state.file, state.recorder, state.url）
- URL 输入模式下不展示波形预览（无法先下载再分析）
- 分析完成后，音频播放器使用 URL 作为 `<audio>` 的 src

### 4.5 CSS 新增

- URL 输入框样式：与现有输入组件风格一致（warm-white 背景、vermilion focus）
- URL 选项卡样式：复用现有分段控件样式

## 5. 隐私影响

- URL 音频不在本地持久存储（下载到临时文件 → 分析 → 删除，同现有设计）
- 请求日志不记录完整 URL（仅记录协议和域名部分）
- 前端不缓存 URL，页面关闭后 URL 不保留

## 6. 测试策略

| 测试文件 | 新增内容 |
|----------|----------|
| `test_audio.py` | `fetch_audio_from_url()` 单元测试（mock httpx），包含协议校验、下载超时、文件过大、Content-Disposition 提取 |
| `test_api.py` | `/api/analyze-url` 端点测试（mock 下载 + fake 模型），包含正常分析、URL 格式错误、下载失败、并发冲突 |
| `test_schemas.py` | `AnalyzeUrlRequest` 校验测试：合法 URL、空白 URL、协议错误 |
| `test_config.py` | 新增配置项的默认值和校验测试 |
| `test_frontend.py` | URL 输入交互测试 |

## 7. 文档更新

- `docs/api.md` 新增 `/api/analyze-url` 端点文档
- `docs/privacy.md` 新增 URL 音频隐私说明
- `docs/model-limitations.md` 无变更（URL 输入不影响模型限制）

## 8. 不做的事（YAGNI）

- 不做 URL 音频缓存/预下载（简单流程优先）
- 不做 Content-Type 强制校验（由 decode_audio 处理）
- 不做 SSRF 私有 IP 过滤（需求支持内网）
- 不做批量 URL 分析
- 不做 URL 音频波形预览下载
- 不做独立下载服务端点
