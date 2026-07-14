# URL 音频分析 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add URL-based audio analysis to the existing speech emotion recognition workbench, allowing users to input an audio URL for analysis alongside the current upload and recording options.

**Architecture:** Minimal intrusion approach — new `fetch_audio_from_url()` function in `app/audio.py` downloads audio bytes, new `POST /api/analyze-url` endpoint in `app/main.py` receives URL JSON and reuses the entire decode→segment→predict→aggregate pipeline, new `AnalyzeUrlRequest` schema validates URL input. Frontend adds a third "URL 地址" tab with an input field, sending JSON POST to the new endpoint and reusing existing NDJSON stream parsing and result rendering.

**Tech Stack:** FastAPI, httpx (sync Client for URL download), Pydantic (AnalyzeUrlRequest), existing HuBERT + analyzer pipeline, vanilla JS + HTML.

## Global Constraints

- Python >=3.11, <3.14
- URL protocol whitelist: only `http://` and `https://`
- No SSRF private IP filtering (design explicitly supports internal URLs)
- Audio download size limit reuses `settings.max_bytes` (50 MB)
- All new schemas inherit `ContractModel` with `extra="forbid"`
- Error codes in UPPER_SNAKE_CASE format
- httpx must be moved from dev dependencies to production dependencies
- Frontend must match existing vermilion design system, Chinese language, no CDN fonts

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `app/audio.py` | Modify | Add `fetch_audio_from_url()` — URL download with streaming size check, timeout, redirect limit, filename inference from Content-Disposition or URL path |
| `app/config.py` | Modify | Add `url_download_timeout_seconds` (default 60.0) and `url_max_redirects` (default 5) settings |
| `app/schemas.py` | Modify | Add `AnalyzeUrlRequest` schema with URL protocol validation |
| `app/main.py` | Modify | Add `POST /api/analyze-url` endpoint — acquire lock, fetch URL, decode, stream analysis, release lock (two-phase strategy matching existing `/api/analyze`) |
| `app/static/index.html` | Modify | Add "URL 地址" tab button, URL input panel with text input and format hint, add `id` attributes for new elements |
| `app/static/styles.css` | Modify | Add URL input panel styles matching existing design system |
| `app/static/app.js` | Modify | Add `state.audioUrl`, URL tab switch logic, URL input validation, `analyzeUrl()` fetch function, update `switchMode`/`clearAll`/`releaseAudio` to handle URL mode |
| `pyproject.toml` | Modify | Move `httpx` from `dev` to production `dependencies` |
| `tests/test_audio.py` | Modify | Add `fetch_audio_from_url` tests with mock httpx |
| `tests/test_api.py` | Modify | Add `/api/analyze-url` endpoint tests with mock download |
| `tests/test_schemas.py` | Modify | Add `AnalyzeUrlRequest` validation tests |
| `tests/test_config.py` | Modify | Add new URL config fields default and validation tests |
| `docs/api.md` | Modify | Add `/api/analyze-url` endpoint documentation |
| `docs/privacy.md` | Modify | Add URL audio privacy note |

---

### Task 1: Add httpx to production dependencies and URL config fields

**Files:**
- Modify: `pyproject.toml:12-22` (dependencies section)
- Modify: `app/config.py:32-108` (Settings class and ENVIRONMENT_KEYS)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: Nothing (foundation task)
- Produces: `Settings.url_download_timeout_seconds: PositiveFloat = 60.0`, `Settings.url_max_redirects: PositiveInt = 5`, updated ENVIRONMENT_KEYS including `URL_DOWNLOAD_TIMEOUT_SECONDS` and `URL_MAX_REDIRECTS`

- [ ] **Step 1: Move httpx from dev to production dependencies in pyproject.toml**

In `pyproject.toml`, move `"httpx>=0.28,<1.0"` from `project.optional-dependencies.dev` to `project.dependencies`. The final dependencies list should be:

```toml
dependencies = [
  "fastapi>=0.115,<1.0",
  "uvicorn[standard]>=0.34,<1.0",
  "python-multipart>=0.0.20,<1.0",
  "pydantic-settings>=2.7,<3.0",
  "numpy>=1.26,<3.0",
  "scipy>=1.14,<2.0",
  "imageio-ffmpeg>=0.6,<1.0",
  "torch>=2.6,<3.0",
  "transformers>=4.48,<5.0",
  "httpx>=0.28,<1.0",
]
```

And remove `"httpx>=0.28,<1.0"` from the dev list.

- [ ] **Step 2: Add URL config fields to Settings class in app/config.py**

Add two new fields to the `Settings` class after `port`:

```python
    # URL 音频下载超时：默认 60 秒 — 足以覆盖大多数网络延迟，
    # 同时不至于让单个请求长时间阻塞服务
    url_download_timeout_seconds: PositiveFloat = 60.0
    # URL 重定向最大次数：限制重定向链长度，防止无限循环
    url_max_redirects: PositiveInt = 5
```

Also update the class docstring's field descriptions section to add these two fields.

- [ ] **Step 3: Write failing test for new config fields**

In `tests/test_config.py`, add to the `ENVIRONMENT_KEYS` tuple: `"URL_DOWNLOAD_TIMEOUT_SECONDS"` and `"URL_MAX_REDIRECTS"`.

Add a new test function:

```python
def test_settings_use_url_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 URL 相关配置的默认值。"""
    clear_settings_environment(monkeypatch)
    settings = Settings()
    assert settings.url_download_timeout_seconds == 60.0
    assert settings.url_max_redirects == 5
```

Add parametrized cases to the existing `test_settings_reject_invalid_limits`:

```python
        ("url_download_timeout_seconds", 0),  # 下载超时不能为零
        ("url_max_redirects", 0),  # 重定向次数不能为零
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: All tests PASS (including new defaults and validation)

- [ ] **Step 5: Install httpx and verify import**

Run: `pip install httpx>=0.28,<1.0`
Then: `python -c "import httpx; print('httpx OK')"`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml app/config.py tests/test_config.py
git commit -m "feat: add httpx dependency and URL download config fields"
```

---

### Task 2: Add AnalyzeUrlRequest schema

**Files:**
- Modify: `app/schemas.py` (add AnalyzeUrlRequest class)
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: `ContractModel` base class from `app/schemas.py`
- Produces: `AnalyzeUrlRequest` class with `url: str` field and `validate_url` field_validator

- [ ] **Step 1: Write the failing test**

In `tests/test_schemas.py`, add import for `AnalyzeUrlRequest` at the top:

```python
from app.schemas import (
    AnalyzeUrlRequest,
    ...existing imports...
)
```

Add test functions:

```python
def test_analyze_url_request_accepts_valid_urls() -> None:
    """验证 AnalyzeUrlRequest 接受合法的 http/https URL。"""
    for url in ["http://example.com/audio.wav", "https://cdn.example.com/file.mp3"]:
        req = AnalyzeUrlRequest(url=url)
        assert req.url == url


def test_analyze_url_request_rejects_invalid_protocols() -> None:
    """验证 AnalyzeUrlRequest 拒绝非 http/https 协议的 URL。"""
    for url in ["ftp://example.com/audio.wav", "file:///tmp/audio.wav", "just-a-string"]:
        with pytest.raises(ValidationError):
            AnalyzeUrlRequest(url=url)


def test_analyze_url_request_rejects_blank_url() -> None:
    """验证 AnalyzeUrlRequest 拒绝空白 URL。"""
    with pytest.raises(ValidationError):
        AnalyzeUrlRequest(url="   ")


def test_analyze_url_request_rejects_extra_fields() -> None:
    """验证 AnalyzeUrlRequest 拒绝额外字段（严格模式）。"""
    with pytest.raises(ValidationError):
        AnalyzeUrlRequest(url="http://example.com/audio.wav", extra="nope")


def test_analyze_url_request_strips_whitespace() -> None:
    """验证 AnalyzeUrlRequest 对 URL 执行空白去除。"""
    req = AnalyzeUrlRequest(url="  https://example.com/audio.wav  ")
    assert req.url == "https://example.com/audio.wav"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schemas.py::test_analyze_url_request_accepts_valid_urls -v`
Expected: FAIL with ImportError (AnalyzeUrlRequest not defined)

- [ ] **Step 3: Add AnalyzeUrlRequest to app/schemas.py**

After the `ErrorEvent` class, add:

```python
class AnalyzeUrlRequest(ContractModel):
    """
    URL 音频分析请求 — 接收音频文件 URL 地址

    字段说明：
    - url : 音频文件 URL，必须以 http:// 或 https:// 开头
            支持任意公网和内网 URL，不做私有 IP 过滤

    安全设计：
    - 协议白名单：仅允许 http 和 https，拒绝 file、ftp 等协议
    - URL 两端空白自动去除，防止用户误输入前后空格
    """
    url: str  # 音频文件 URL

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """
        字段级校验：URL 协议白名单 + 空白去除

        校验规则：
        1. 去除两端空白（防止用户误输入前后空格）
        2. URL 必须以 http:// 或 https:// 开头
           拒绝其他协议（file://、ftp:// 等）防止 SSRF
        """
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_schemas.py -v -k analyze_url`
Expected: All 5 new tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat: add AnalyzeUrlRequest schema with URL protocol validation"
```

---

### Task 3: Add fetch_audio_from_url() in app/audio.py

**Files:**
- Modify: `app/audio.py` (add `fetch_audio_from_url` function)
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `Settings` (for `max_bytes`, `url_download_timeout_seconds`, `url_max_redirects`), `AppError` from `app/errors`
- Produces: `fetch_audio_from_url(url: str, settings: Settings) -> tuple[bytes, str]` — returns (audio_bytes, inferred_filename) for downstream `decode_audio()`

- [ ] **Step 1: Write the failing tests**

In `tests/test_audio.py`, add import for `fetch_audio_from_url`:

```python
from app.audio import fetch_audio_from_url, normalize_waveform, segment_waveform
```

Add test functions:

```python
def test_fetch_audio_from_url_downloads_successfully(monkeypatch) -> None:
    """验证 fetch_audio_from_url 正常下载音频并推断文件名。"""
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-disposition": "attachment; filename=\"test_audio.wav\""}
    mock_response.iter_bytes.return_value = [b"fake audio data"]

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
    mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr("app.audio.httpx.Client", lambda **kwargs: mock_client)

    data, filename = fetch_audio_from_url("https://example.com/audio.wav", Settings())
    assert data == b"fake audio data"
    assert filename == "test_audio.wav"


def test_fetch_audio_from_url_infers_filename_from_url_path(monkeypatch) -> None:
    """验证无 Content-Disposition 时从 URL 路径推断文件名。"""
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.iter_bytes.return_value = [b"audio data"]

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
    mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr("app.audio.httpx.Client", lambda **kwargs: mock_client)

    data, filename = fetch_audio_from_url("https://cdn.example.com/path/to/file.mp3", Settings())
    assert filename == "file.mp3"


def test_fetch_audio_from_url_uses_fallback_filename(monkeypatch) -> None:
    """验证无 Content-Disposition 且无路径扩展名时使用兜底文件名。"""
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.iter_bytes.return_value = [b"audio data"]

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
    mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr("app.audio.httpx.Client", lambda **kwargs: mock_client)

    data, filename = fetch_audio_from_url("https://example.com/api/audio", Settings())
    assert filename == "downloaded_audio"


def test_fetch_audio_from_url_rejects_invalid_protocol() -> None:
    """验证 fetch_audio_from_url 拒绝非 http/https 协议的 URL。"""
    with pytest.raises(AppError, match="INVALID_URL"):
        fetch_audio_from_url("ftp://example.com/audio.wav", Settings())


def test_fetch_audio_from_url_rejects_oversized_file(monkeypatch) -> None:
    """验证 fetch_audio_from_url 拒绝超过大小限制的下载。"""
    from unittest.mock import MagicMock

    # 创建超过 50MB 的模拟数据块
    big_chunk = b"x" * (51 * 1024 * 1024)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.iter_bytes.return_value = [big_chunk]

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
    mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr("app.audio.httpx.Client", lambda **kwargs: mock_client)

    with pytest.raises(AppError, match="URL_FILE_TOO_LARGE"):
        fetch_audio_from_url("https://example.com/big.wav", Settings())


def test_fetch_audio_from_url_handles_download_failure(monkeypatch) -> None:
    """验证 fetch_audio_from_url 处理下载失败（网络错误）。"""
    import httpx

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.stream.side_effect = httpx.ConnectError("Connection refused")

    monkeypatch.setattr("app.audio.httpx.Client", lambda **kwargs: mock_client)

    with pytest.raises(AppError, match="URL_DOWNLOAD_FAILED"):
        fetch_audio_from_url("https://unreachable.example.com/audio.wav", Settings())


def test_fetch_audio_from_url_handles_timeout(monkeypatch) -> None:
    """验证 fetch_audio_from_url 处理下载超时。"""
    import httpx

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.stream.side_effect = httpx.ReadTimeout("Read timed out")

    monkeypatch.setattr("app.audio.httpx.Client", lambda **kwargs: mock_client)

    with pytest.raises(AppError, match="URL_DOWNLOAD_TIMEOUT"):
        fetch_audio_from_url("https://slow.example.com/audio.wav", Settings())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audio.py -v -k fetch_audio`
Expected: FAIL with ImportError (`fetch_audio_from_url` not defined)

- [ ] **Step 3: Implement fetch_audio_from_url**

Add `httpx` import at the top of `app/audio.py`, after existing imports:

```python
import httpx
```

Add the function after `read_limited_stream` (around line 143):

```python
def _extract_filename_from_headers(headers: httpx.Headers) -> str | None:
    """
    从 Content-Disposition header 提取文件名

    优先级：filename*（RFC 5987 编码） > filename

    参数：
        headers — httpx 响应头对象

    返回：
        提取的文件名字符串，若无 Content-Disposition 则返回 None
    """
    disposition = headers.get("content-disposition", "")
    if not disposition:
        return None
    # 尝试提取 filename*（RFC 5987 编码，如 filename*=UTF-8''test.wav）
    for part in disposition.split(";"):
        part = part.strip()
        if part.lower().startswith("filename*"):
            # 格式：filename*=UTF-8''encoded_name
            try:
                _, _, encoded = part.split("'", 2)
                return urllib.parse.unquote(encoded.strip())
            except ValueError:
                continue
    # 尝试提取 filename（基础格式，如 filename="test.wav"）
    for part in disposition.split(";"):
        part = part.strip()
        if part.lower().startswith("filename"):
            # 去除 filename= 前缀和可能的引号
            value = part.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    return None


def _extract_filename_from_url(url: str) -> str:
    """
    从 URL 路径推断文件名

    提取 URL 路径的最后一段作为文件名。
    若路径为空或最后一段无扩展名，返回兜底名 "downloaded_audio"。

    参数：
        url — 音频文件 URL

    返回：
        推断的文件名字符串
    """
    path = urllib.parse.urlparse(url).path
    if path:
        basename = path.rstrip("/").rsplit("/", 1)[-1]
        if basename and any(basename.lower().endswith(s) for s in SUPPORTED_SUFFIXES):
            return basename
    return "downloaded_audio"


def fetch_audio_from_url(url: str, settings: Settings) -> tuple[bytes, str]:
    """
    从 URL 下载音频文件，返回字节内容与推断的文件名

    此函数将远程音频文件下载到内存，供 decode_audio() 处理。
    采用流式下载 + 实时大小检查策略，防止恶意大文件耗尽内存。

    安全设计：
    - 协议白名单：仅允许 http:// 和 https://
    - 不过滤私有 IP（需求明确支持内网 URL）
    - 流式下载实时检查大小不超过 max_bytes
    - 重定向限制防止无限循环

    参数：
        url — 音频文件 URL
        settings — 全局配置，提供 max_bytes、url_download_timeout_seconds、url_max_redirects

    返回：
        (bytes, inferred_filename) — 音频字节内容与推断的文件名

    异常：
        AppError("INVALID_URL", 400) — URL 协议不合法
        AppError("URL_DOWNLOAD_FAILED", 400) — 下载失败
        AppError("URL_DOWNLOAD_TIMEOUT", 408) — 下载超时
        AppError("URL_FILE_TOO_LARGE", 413) — 下载文件过大
    """
    # 协议白名单校验
    stripped_url = url.strip()
    if not stripped_url.startswith(("http://", "https://")):
        raise AppError("INVALID_URL", "URL 必须以 http:// 或 https:// 开头", 400)

    timeout_config = httpx.Timeout(
        connect=10.0,
        read=settings.url_download_timeout_seconds,
        write=10.0,
        pool=10.0,
    )

    try:
        with httpx.Client(
            timeout=timeout_config,
            max_redirects=settings.url_max_redirects,
            follow_redirects=True,
        ) as client:
            with client.stream("GET", stripped_url) as response:
                if response.status_code >= 400:
                    raise AppError(
                        "URL_DOWNLOAD_FAILED",
                        f"音频下载失败（HTTP {response.status_code}）",
                        400,
                    )
                # 推断文件名：优先 Content-Disposition，其次 URL 路径，兜底默认
                filename = _extract_filename_from_headers(response.headers)
                if filename is None:
                    filename = _extract_filename_from_url(stripped_url)

                # 流式下载 + 实时大小检查
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes(chunk_size=READ_CHUNK_SIZE):
                    total += len(chunk)
                    if total > settings.max_bytes:
                        raise AppError(
                            "URL_FILE_TOO_LARGE",
                            "音频文件不能超过 50 MB",
                            413,
                        )
                    chunks.append(chunk)

                if not chunks:
                    raise AppError("URL_DOWNLOAD_FAILED", "下载的音频文件为空", 400)

                return b"".join(chunks), filename

    except httpx.TimeoutException as exc:
        raise AppError("URL_DOWNLOAD_TIMEOUT", "音频下载超时，请检查 URL 或重试", 408) from exc
    except httpx.HTTPStatusError as exc:
        raise AppError("URL_DOWNLOAD_FAILED", f"音频下载失败（HTTP {exc.response.status_code})", 400) from exc
    except (httpx.ConnectError, httpx.NetworkError) as exc:
        raise AppError("URL_DOWNLOAD_FAILED", "音频下载失败，请检查 URL 是否可访问", 400) from exc
    except AppError:
        raise  # 已转换的 AppError 直接抛出，不二次包装
    except Exception as exc:
        raise AppError("URL_DOWNLOAD_FAILED", "音频下载失败，请重试", 400) from exc
```

Also add `import urllib.parse` at the top of `app/audio.py` with the other imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audio.py -v -k fetch_audio`
Expected: All 7 new tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/audio.py tests/test_audio.py
git commit -m "feat: add fetch_audio_from_url with streaming download, size check, and filename inference"
```

---

### Task 4: Add POST /api/analyze-url endpoint

**Files:**
- Modify: `app/main.py` (add `analyze_url` endpoint and import `AnalyzeUrlRequest`, `fetch_audio_from_url`)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `AnalyzeUrlRequest` from `app/schemas.py`, `fetch_audio_from_url` from `app/audio.py`, `decode_audio` from `app/audio.py`, `Services`, `Settings`, `StreamingResponse`, `analysis_lock`, `ErrorEvent`, `PublicError`
- Produces: `POST /api/analyze-url` endpoint returning NDJSON StreamingResponse (same format as `/api/analyze`)

- [ ] **Step 1: Write the failing test**

In `tests/test_api.py`, add imports:

```python
from app.schemas import (
    AnalyzeUrlRequest,
    ...existing imports...
)
from app.audio import DecodedAudio, fetch_audio_from_url
```

Add test functions:

```python
def test_analyze_url_stream_returns_progress_and_result(monkeypatch) -> None:
    """验证 URL 分析端点返回 NDJSON 格式的进度与结果事件。"""
    # Mock fetch_audio_from_url 返回模拟音频字节
    monkeypatch.setattr(
        main_module,
        "fetch_audio_from_url",
        lambda _url, _settings: (b"synthetic_audio_data", "sample.wav"),
    )
    # Mock decode_audio 返回合成的解码音频
    monkeypatch.setattr(
        main_module,
        "decode_audio",
        lambda _data, _filename, _settings: DecodedAudio(np.ones(16000, dtype=np.float32), 16000),
    )
    client = TestClient(main_module.create_app(Settings(), FakeServices()))  # type: ignore[arg-type]
    with client.stream(
        "POST", "/api/analyze-url", json={"url": "https://example.com/audio.wav"}
    ) as response:
        lines = list(response.iter_lines())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert '"type":"progress"' in lines[0]
    assert '"type":"result"' in lines[-1]


def test_analyze_url_rejects_invalid_url_format() -> None:
    """验证 URL 分析端点拒绝非法 URL 格式。"""
    client = TestClient(main_module.create_app(Settings(), FakeServices()))  # type: ignore[arg-type]
    response = client.post("/api/analyze-url", json={"url": "ftp://example.com/audio.wav"})
    assert response.status_code == 422  # Pydantic validation error for invalid protocol


def test_analyze_url_handles_download_failure(monkeypatch) -> None:
    """验证 URL 分析端点处理下载失败。"""
    from app.errors import AppError

    monkeypatch.setattr(
        main_module,
        "fetch_audio_from_url",
        lambda _url, _settings: (_ for _ in ()).throw(AppError("URL_DOWNLOAD_FAILED", "下载失败", 400)),
    )
    client = TestClient(main_module.create_app(Settings(), FakeServices()))  # type: ignore[arg-type]
    response = client.post("/api/analyze-url", json={"url": "https://unreachable.example.com/audio.wav"})
    assert response.status_code == 400


def test_analyze_url_returns_429_when_busy() -> None:
    """验证 URL 分析端点在并发冲突时返回 429。"""
    # Mock fetch_audio_from_url 返回模拟音频字节
    monkeypatch.setattr(
        main_module,
        "fetch_audio_from_url",
        lambda _url, _settings: (b"synthetic_audio_data", "sample.wav"),
    )
    monkeypatch.setattr(
        main_module,
        "decode_audio",
        lambda _data, _filename, _settings: DecodedAudio(np.ones(16000, dtype=np.float32), 16000),
    )
    # 直接用真实 Services（不需要 FakeServices，因为已有 mock）
    # 此测试验证并发锁机制：先获取锁，再请求应得到 429
    app = main_module.create_app(Settings())
    # 手动获取分析锁，模拟并发占用
    lock_acquired = app.state.analysis_lock.acquire(blocking=False)
    assert lock_acquired
    client = TestClient(app)
    response = client.post("/api/analyze-url", json={"url": "https://example.com/audio.wav"})
    assert response.status_code == 429
    # 释放锁
    app.state.analysis_lock.release()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py -v -k analyze_url`
Expected: FAIL (endpoint not defined, 404)

- [ ] **Step 3: Add the endpoint to app/main.py**

Add imports at the top of `app/main.py`:

```python
from app.audio import decode_audio, fetch_audio_from_url, read_limited_stream
from app.schemas import AnalyzeUrlRequest, ErrorEvent, HealthResponse, PublicError
```

(Note: `read_limited_stream` and `decode_audio` are already imported; add `fetch_audio_from_url` and `AnalyzeUrlRequest`)

Add the endpoint after the existing `analyze` endpoint (around line 349):

```python
    @application.post("/api/analyze-url")
    def analyze_url(request: AnalyzeUrlRequest) -> StreamingResponse:
        """
        URL 音频分析端点 — 从 URL 下载音频并执行情绪分析

        此端点是新增的 URL 输入方式，接收 JSON body 中的 URL 参数，
        下载音频后复用完整的分析流程，返回 NDJSON 流式响应。

        流程：
        1. 获取分析锁 → 429 if busy（同现有 /api/analyze）
        2. 从 URL 下载音频 → fetch_audio_from_url 流式下载 + 实时大小检查
        3. FFmpeg 解码 → decode_audio（复用现有流程）
        4. 流式输出分析结果 → 复用 iter_analysis + NDJSON 流式传输
        5. 释放分析锁 → finally 块确保释放

        锁的释放策略（同现有 /api/analyze 的两阶段）：
        - 下载/解码阶段异常：except 块手动释放
        - 流式阶段异常：finally 块自动释放
        """
        if not application.state.analysis_lock.acquire(blocking=False):
            raise AppError("ANALYSIS_BUSY", "已有分析任务正在进行，请稍后重试", 429)
        try:
            data, filename = fetch_audio_from_url(request.url, resolved_settings)
            decoded = decode_audio(data, filename, resolved_settings)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py -v -k analyze_url`
Expected: All 4 new tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add POST /api/analyze-url endpoint with NDJSON streaming"
```

---

### Task 5: Add URL input tab to frontend HTML and CSS

**Files:**
- Modify: `app/static/index.html` (add "URL 地址" tab button, URL input panel)
- Modify: `app/static/styles.css` (add URL input styles, update segmented control to 3 columns)

**Interfaces:**
- Consumes: Existing HTML structure and CSS design system
- Produces: New DOM elements with IDs: `modeUrl`, `urlPanel`, `urlInput`, `urlHint`; CSS classes for URL panel and segmented 3-column layout

- [ ] **Step 1: Update segmented control in index.html to 3 columns**

In `app/static/index.html`, modify the segmented control div (currently 2 buttons) to add a third "URL 地址" button:

```html
        <div class="segmented" role="tablist" aria-label="音频来源">
          <button id="modeUpload" type="button" role="tab" aria-selected="true">上传音频</button>
          <button id="modeRecord" type="button" role="tab" aria-selected="false">现场录音</button>
          <button id="modeUrl" type="button" role="tab" aria-selected="false">URL 地址</button>
        </div>
```

- [ ] **Step 2: Add URL input panel in index.html**

After the `recordPanel` div (around line 88), add the URL input panel:

```html
        <!-- ====== URL 输入面板 ======
             输入音频文件 URL 地址，服务端下载后进行分析 -->
        <div id="urlPanel" class="source-panel url-panel" role="tabpanel" hidden>
          <label class="url-field">
            <strong>输入音频文件 URL 地址</strong>
            <input id="urlInput" type="url" placeholder="https://example.com/audio.wav" autocomplete="off" spellcheck="false" />
          </label>
          <span class="url-hint">支持 WAV、MP3、FLAC、OGG、M4A、WebM 格式</span>
        </div>
```

- [ ] **Step 3: Update CSS segmented control for 3 columns**

In `app/static/styles.css`, change the `.segmented` grid from 2 columns to 3:

```css
.segmented { width: 408px; height: 42px; padding: 3px; margin: 24px 0 18px; display: grid; grid-template-columns: 1fr 1fr 1fr; background: #ecece8; border-radius: 6px; }
```

(The width increases from 272px to 408px to accommodate 3 tabs evenly)

- [ ] **Step 4: Add URL panel CSS styles**

In `app/static/styles.css`, add after the `.record-panel` styles (around line 115):

```css
/* ====== URL 输入面板 ======
   居中布局，1px实线边框 + 圆角 + 浅灰底色，与录音面板视觉统一 */
.url-panel { display: grid; place-items: center; border: 1px solid var(--line); border-radius: var(--radius); background: #fafaf8; }

/* URL 输入字段标签：居中排列标题和输入框 */
.url-field { width: min(460px,100%); display: flex; flex-direction: column; gap: 10px; padding: 32px 0; text-align: center; }

/* URL 输入框：42px高，全宽，1px边框 + 圆角4px，浅灰底色
   焦点时：朱红描边 + 微红白底色 */
.url-field input { height: 42px; width: 100%; padding: 0 14px; border: 1px solid var(--line); border-radius: 4px; background: #fff; color: var(--ink); font-size: 14px; }
.url-field input:focus { outline: none; border-color: var(--vermilion); box-shadow: 0 0 0 3px rgba(198,60,47,.12); }
.url-field input::placeholder { color: var(--muted); }

/* URL 格式提示：辅助色12px */
.url-hint { color: var(--muted); font-size: 12px; margin-top: 4px; }
```

Also update the 860px media query to handle the wider segmented control:

```css
@media (max-width: 860px) { ...existing rules... .segmented { width: 100%; } ... }
```

(This already exists; no change needed since it already makes segmented full-width)

- [ ] **Step 5: Verify HTML loads correctly**

Run: `python -m pytest tests/test_frontend.py -v` (if exists), or manually open `http://127.0.0.1:8080` in browser and verify the 3-tab segmented control and URL panel appear.

- [ ] **Step 6: Commit**

```bash
git add app/static/index.html app/static/styles.css
git commit -m "feat: add URL input tab and panel to frontend HTML and CSS"
```

---

### Task 6: Add URL mode JavaScript logic

**Files:**
- Modify: `app/static/app.js` (add `state.audioUrl`, URL mode switching, URL validation, `analyzeUrl()` function)

**Interfaces:**
- Consumes: New DOM elements (`modeUrl`, `urlPanel`, `urlInput`, `urlHint`), existing `state`, `setStatus`, `showError`, `handleEvent`, `renderResult`
- Produces: `analyzeUrl()` function, `switchMode` updated for 3 modes, `clearAll`/`releaseAudio` updated for URL state

- [ ] **Step 1: Add `audioUrl` field to state object**

Modify the state object (line 35) to include `audioUrl`:

```javascript
const state = { file: null, url: null, recorder: null, chunks: [], timer: null, seconds: 0, result: null, audioUrl: "" };
```

- [ ] **Step 2: Update switchMode function for 3 modes**

Replace the existing `switchMode` function with a 3-mode version:

```javascript
function switchMode(mode) {
  if (state.recorder?.state === "recording" || state.recorder?.state === "paused") return;
  $("modeUpload").setAttribute("aria-selected", String(mode === "upload"));
  $("modeRecord").setAttribute("aria-selected", String(mode === "record"));
  $("modeUrl").setAttribute("aria-selected", String(mode === "url"));
  $("uploadPanel").hidden = mode !== "upload";
  $("recordPanel").hidden = mode !== "record";
  $("urlPanel").hidden = mode !== "url";
  showError();
}
```

- [ ] **Step 3: Update releaseAudio to handle URL state**

Update the `releaseAudio` function to also clear `state.audioUrl`:

```javascript
function releaseAudio() {
  if (state.url) URL.revokeObjectURL(state.url);
  state.url = null; state.file = null; state.audioUrl = ""; $("audioPlayer").removeAttribute("src"); $("audioPreview").hidden = true;
  $("analyzeButton").disabled = true; $("resultContent").hidden = true; $("resultEmpty").hidden = false; setStatus("等待音频");
}
```

- [ ] **Step 4: Add URL input validation and analysis button control**

Add a new function to validate URL input and control the analyze button:

```javascript
function validateUrlInput() {
  const value = $("urlInput").value.trim();
  const valid = value.startsWith("http://") || value.startsWith("https://");
  $("analyzeButton").disabled = !valid;
  state.audioUrl = valid ? value : "";
}
```

- [ ] **Step 5: Add analyzeUrl function**

Add a new `analyzeUrl` function after the existing `analyze` function:

```javascript
async function analyzeUrl() {
  if (!state.audioUrl) return;
  showError(); $("analyzeButton").disabled = true; setStatus("正在连接本地模型", true);
  try {
    const response = await fetch("/api/analyze-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: state.audioUrl }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error?.message || "分析失败");
    }
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n"); buffer = lines.pop() || "";
      for (const line of lines) if (line.trim()) handleEvent(JSON.parse(line));
      if (done) break;
    }
    if (buffer.trim()) handleEvent(JSON.parse(buffer));
  } catch (error) { showError(error.message || "分析失败，请重试"); setStatus("分析未完成"); }
  finally { $("analyzeButton").disabled = !state.audioUrl; }
}
```

- [ ] **Step 6: Update analyze function to dispatch based on mode**

Modify the existing `analyze` function to check current mode and dispatch accordingly. Replace the `analyze` function:

```javascript
async function analyze() {
  // URL 模式走独立端点
  if (state.audioUrl) return analyzeUrl();
  if (!state.file) return;
  showError(); $("analyzeButton").disabled = true; setStatus("正在连接本地模型", true); const body = new FormData(); body.append("audio", state.file, state.file.name);
  try { const response = await fetch("/api/analyze", { method: "POST", body }); if (!response.ok) { const payload = await response.json(); throw new Error(payload.error?.message || "分析失败"); } const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "";
    while (true) { const { value, done } = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), { stream: !done }); const lines = buffer.split("\n"); buffer = lines.pop() || ""; for (const line of lines) if (line.trim()) handleEvent(JSON.parse(line)); if (done) break; }
    if (buffer.trim()) handleEvent(JSON.parse(buffer));
  } catch (error) { showError(error.message || "分析失败，请重试"); setStatus("分析未完成"); } finally { $("analyzeButton").disabled = !state.file; }
}
```

- [ ] **Step 7: Add event listeners for URL mode**

In the event listeners section, add:

```javascript
$("modeUrl").addEventListener("click", () => switchMode("url"));
$("urlInput").addEventListener("input", validateUrlInput);
```

- [ ] **Step 8: Update analyzeButton finally block for URL mode**

The existing `analyze` function's finally block sets `$("analyzeButton").disabled = !state.file`. This needs to also consider `state.audioUrl` when in URL mode. Modify both the `analyze` and `analyzeUrl` finally blocks:

For `analyze`:
```javascript
finally { $("analyzeButton").disabled = !state.file && !state.audioUrl; }
```

For `analyzeUrl`:
```javascript
finally { $("analyzeButton").disabled = !state.audioUrl; }
```

- [ ] **Step 9: Verify the frontend works**

Start the server (`python -m uvicorn app.main:app --port 8080`) and open `http://127.0.0.1:8080` in a browser. Verify:
- The segmented control shows 3 tabs: 上传音频 / 现径录音 / URL 地址
- Clicking "URL 地址" shows the URL input panel
- Entering a valid URL enables the "开始分析" button
- Entering an invalid URL keeps the button disabled
- Switching back to upload/record modes hides the URL panel

- [ ] **Step 10: Commit**

```bash
git add app/static/app.js
git commit -m "feat: add URL mode logic to frontend JavaScript"
```

---

### Task 7: Update API documentation

**Files:**
- Modify: `docs/api.md` (add `/api/analyze-url` endpoint documentation)
- Modify: `docs/privacy.md` (add URL audio privacy note)

**Interfaces:**
- Consumes: Design spec for error codes and endpoint behavior
- Produces: Updated documentation

- [ ] **Step 1: Read existing docs/api.md to understand the format**

Read the current `docs/api.md` to match the existing documentation style.

- [ ] **Step 2: Add /api/analyze-url section to docs/api.md**

Add a new section documenting the endpoint, matching the existing style:

```markdown
### POST /api/analyze-url

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
```

- [ ] **Step 3: Add URL audio privacy note to docs/privacy.md**

Add a paragraph about URL audio privacy handling, matching the existing style:

```markdown
### URL 音频

- URL 音频下载到内存临时缓冲区，经 FFmpeg 解码后立即用于分析
- 分析完成后，内存中的音频数据随请求结束自动释放，不在本地持久存储
- 请求日志仅记录 URL 的协议和域名部分，不记录完整路径，防止泄露内网地址
- 前端不缓存 URL，页面关闭或刷新后 URL 输入自动清除
```

- [ ] **Step 4: Commit**

```bash
git add docs/api.md docs/privacy.md
git commit -m "docs: add /api/analyze-url endpoint and URL privacy documentation"
```

---

### Task 8: Integration smoke test

**Files:**
- None new (uses existing running server)

**Interfaces:**
- Consumes: All previous tasks — full backend + frontend implementation
- Produces: Verified end-to-end URL audio analysis workflow

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All existing and new tests PASS

- [ ] **Step 2: Start the server**

```bash
PORT=8080 python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

- [ ] **Step 3: Verify health endpoint still works**

```bash
powershell -Command "Invoke-RestMethod -Uri http://127.0.0.1:8080/api/health"
```

Expected: `{ status: "ok", model_status: "...", device: "..." }`

- [ ] **Step 4: Test URL endpoint with a valid public audio URL**

Use PowerShell to send a test request:

```powershell
$body = '{"url": "https://www2.cs.uic.edu/~i101/SoundFiles/Baalthazar.wav"}'
Invoke-WebRequest -Uri http://127.0.0.1:8080/api/analyze-url -Method POST -ContentType "application/json" -Body $body
```

Expected: 200 response with NDJSON content-type (the actual analysis may take time due to model loading on first call)

- [ ] **Step 5: Test URL endpoint with invalid URL**

```powershell
$body = '{"url": "ftp://example.com/audio.wav"}'
Invoke-WebRequest -Uri http://127.0.0.1:8080/api/analyze-url -Method POST -ContentType "application/json" -Body $body
```

Expected: 422 response (Pydantic validation error for protocol)

- [ ] **Step 6: Open browser and verify frontend URL mode**

Open `http://127.0.0.1:8080`, click "URL 地址" tab, enter a URL, click "开始分析", verify NDJSON streaming progress and result rendering.

- [ ] **Step 7: Final commit if any minor fixes needed**

```bash
git add -A
git commit -m "fix: integration adjustments for URL audio analysis"
```
