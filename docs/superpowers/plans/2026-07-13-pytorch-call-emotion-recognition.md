# PyTorch Call Emotion Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a privacy-first local PyTorch web application that analyzes uploaded or browser-recorded Mandarin calls into an overall four-emotion summary and a segmented timeline.

**Architecture:** A FastAPI process serves a native HTML/CSS/JavaScript workbench and a small local API. Audio decoding, HuBERT inference, probability projection, and HTTP transport live in separate modules; analysis progress streams as newline-delimited JSON, and routine tests inject fake models so they never download the 1.1 GB checkpoint.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, PyTorch, Transformers, NumPy, SciPy, imageio-ffmpeg, Pydantic, pytest, Ruff, mypy, Playwright, native HTML/CSS/JavaScript.

---

## File Responsibility Map

- `app/config.py`: immutable environment-backed limits and model settings.
- `app/errors.py`: public error codes and safe exception boundary.
- `app/schemas.py`: emotion, segment, result, progress, health, and error contracts.
- `app/audio.py`: limited upload reads, FFmpeg decode, normalization, segmentation, and silence detection.
- `app/model.py`: HuBERT classification head, lazy runtime, device selection, and CPU retry.
- `app/analyzer.py`: six-to-four projection, reliability evaluation, weighted aggregation, and streamed progress events.
- `app/main.py`: application factory, static files, health/model-load/analyze endpoints, concurrency guard, and safe logs.
- `app/static/index.html`: semantic Chinese workbench shell.
- `app/static/styles.css`: responsive visual system and stable component dimensions.
- `app/static/app.js`: state machine, upload, browser WAV recording, waveform, streamed API parsing, and timeline interaction.
- `app/static/icons/*.svg`: local Lucide icon assets with no CDN dependency.
- `tests/fakes.py`: fake processor, fake HuBERT model, generated audio, and log helpers shared by tests.
- `tests/test_config.py`: configuration validation.
- `tests/test_schemas.py`: API contract serialization.
- `tests/test_audio.py`: decode, normalize, limit, segment, and silence behavior.
- `tests/test_model.py`: lazy loading, inference mode, label order, device choice, and fallback.
- `tests/test_analyzer.py`: projection, thresholds, aggregation, silence, and progress.
- `tests/test_api.py`: endpoint and streaming integration behavior.
- `tests/test_privacy.py`: sensitive-pattern, filename, path, and cleanup regression checks.
- `tests/test_frontend.py`: semantic DOM contract and local-only asset references.
- `tests/e2e/test_workbench.py`: desktop/mobile visual and interaction checks.
- `scripts/download_model.py`: explicit checkpoint warm-up.
- `scripts/smoke_test_model.py`: opt-in real-model inference.
- `scripts/privacy_scan.py`: tracked-file and staged-diff privacy scan.
- `docs/*.md`, root community files, and `.github/workflows/tests.yml`: user, architecture, limitation, privacy, contribution, security, and CI documentation.

All Python and JavaScript implementation tasks include concise Chinese comments for the non-obvious decisions: sample-rate conversion, safe subprocess invocation, HuBERT pooling, inference mode, six-to-four projection, weighted aggregation, stream buffering, WAV encoding, object URL cleanup, and privacy boundaries. Comments explain intent and trade-offs without narrating elementary syntax.

## Task 1: Repository Scaffold and Tooling

**Files:**
- Create: `app/__init__.py`
- Create: `tests/__init__.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: Create the package markers**

Use empty `app/__init__.py` and `tests/__init__.py` files so imports behave consistently on Windows and Linux.

- [ ] **Step 2: Define runtime and development dependencies**

Create `pyproject.toml` with this project metadata and tool configuration:

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pytorch-call-emotion-recognition"
version = "0.1.0"
description = "Local-first Mandarin call emotion recognition with PyTorch"
readme = "README.md"
requires-python = ">=3.11,<3.14"
license = { text = "MIT" }
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
]

[project.optional-dependencies]
dev = [
  "httpx>=0.28,<1.0",
  "mypy>=1.14,<2.0",
  "pytest>=8.3,<9.0",
  "pytest-asyncio>=0.25,<1.0",
  "pytest-cov>=6.0,<7.0",
  "pytest-playwright>=0.7,<1.0",
  "ruff>=0.9,<1.0",
]

[tool.setuptools.packages.find]
include = ["app*"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]
markers = ["model_smoke: downloads and runs the real checkpoint"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["app"]
```

Create `requirements.txt` containing `-e .` and `requirements-dev.txt` containing `-e .[dev]`.

- [ ] **Step 3: Define ignored local and generated content**

Create `.gitignore` with Python caches, `.venv/`, `.env`, model files (`*.bin`, `*.safetensors`, `*.pt`, `*.pth`), Hugging Face caches, temporary audio, coverage, Playwright output, IDE state, OS metadata, and `docs/superpowers/visuals/`. Create `.env.example` with non-sensitive defaults for model ID, byte limit, duration limit, window size, hop size, silence threshold, host, and port.

- [ ] **Step 4: Install the development environment**

Run: `python -m venv .venv`

Run: `.\.venv\Scripts\python -m pip install -U pip`

Run: `.\.venv\Scripts\python -m pip install -r requirements-dev.txt`

Expected: installation completes without resolving dependencies from private indexes.

- [ ] **Step 5: Verify tool discovery**

Run: `.\.venv\Scripts\python -m pytest --collect-only`

Expected: exit code 5 because no tests exist yet, with no import or configuration error.

- [ ] **Step 6: Commit**

```bash
git add app/__init__.py tests/__init__.py pyproject.toml requirements.txt requirements-dev.txt .gitignore .env.example
git commit -m "build: scaffold Python application"
```

## Task 2: Configuration, Errors, and API Contracts

**Files:**
- Create: `app/config.py`
- Create: `app/errors.py`
- Create: `app/schemas.py`
- Create: `tests/test_config.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing configuration and schema tests**

```python
# tests/test_config.py
from pydantic import ValidationError
import pytest

from app.config import Settings


def test_settings_reject_hop_larger_than_window() -> None:
    with pytest.raises(ValidationError):
        Settings(window_seconds=5.0, hop_seconds=6.0)


def test_settings_use_privacy_safe_defaults() -> None:
    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.max_bytes == 50 * 1024 * 1024
    assert settings.max_duration_seconds == 300.0
```

```python
# tests/test_schemas.py
import pytest

from app.schemas import AnalysisResult, EmotionProbabilities, Reliability


def test_analysis_result_serializes_chinese_facing_contract() -> None:
    result = AnalysisResult(
        dominant_emotion="neutral",
        probabilities=EmotionProbabilities(neutral=0.7, happy=0.1, anger=0.1, sad=0.1),
        reliability=Reliability(level="high", reasons=[]),
        excluded_probability=0.08,
        voiced_ratio=0.9,
        duration_seconds=4.0,
        device="cpu",
        elapsed_ms=120,
        segments=[],
    )
    payload = result.model_dump(mode="json")
    assert payload["dominant_emotion"] == "neutral"
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_config.py tests/test_schemas.py -v`

Expected: FAIL because `app.config` and `app.schemas` do not exist.

- [ ] **Step 3: Implement validated settings and stable contracts**

Implement `Settings(BaseSettings)` with the fields from `.env.example`, a model validator enforcing `0 < hop_seconds <= window_seconds`, and an `lru_cache`-backed `get_settings()`.

Implement `AppError` as:

```python
class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status_code = status_code
```

Implement strict Pydantic schemas for `EmotionProbabilities`, `Reliability`, `SegmentResult`, `AnalysisResult`, `HealthResponse`, `ProgressEvent`, `ResultEvent`, and `ErrorEvent`. Probability fields use `Field(ge=0, le=1)`; public enum literals are fixed to `neutral`, `happy`, `anger`, `sad`, and event types are fixed literals.

- [ ] **Step 4: Run tests and static checks**

Run: `.\.venv\Scripts\python -m pytest tests/test_config.py tests/test_schemas.py -v`

Expected: PASS.

Run: `.\.venv\Scripts\python -m ruff check app tests`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/errors.py app/schemas.py tests/test_config.py tests/test_schemas.py
git commit -m "feat: define configuration and API contracts"
```

## Task 3: Audio Decode, Normalization, and Segmentation

**Files:**
- Create: `app/audio.py`
- Create: `tests/fakes.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing pure-audio tests**

```python
import numpy as np
import pytest

from app.audio import normalize_waveform, segment_waveform
from app.config import Settings
from app.errors import AppError


def test_normalize_waveform_downmixes_and_resamples() -> None:
    stereo = np.stack([np.ones(8000), np.zeros(8000)]).astype(np.float32)
    result = normalize_waveform(stereo, source_rate=8000, target_rate=16000)
    assert result.dtype == np.float32
    assert result.shape == (16000,)
    assert np.mean(result) == pytest.approx(0.5, abs=0.02)


def test_segment_waveform_keeps_tail_and_marks_silence() -> None:
    waveform = np.concatenate([
        np.ones(6 * 16000, dtype=np.float32) * 0.2,
        np.zeros(2 * 16000, dtype=np.float32),
    ])
    segments = segment_waveform(waveform, Settings(window_seconds=6, hop_seconds=5))
    assert [(item.start_seconds, item.end_seconds) for item in segments] == [(0.0, 6.0), (5.0, 8.0)]
    assert segments[0].is_silent is False
    assert segments[1].sample_count == 3 * 16000


def test_normalize_waveform_rejects_non_finite_samples() -> None:
    with pytest.raises(AppError, match="音频包含无效采样值"):
        normalize_waveform(np.array([0.0, np.nan], dtype=np.float32), 16000, 16000)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_audio.py -v`

Expected: FAIL because `app.audio` does not exist.

- [ ] **Step 3: Implement pure normalization and segmentation**

Add frozen `AudioSegment` and `DecodedAudio` dataclasses. `normalize_waveform()` accepts `[samples]`, `[channels, samples]`, or `[samples, channels]`, downmixes channels, uses `scipy.signal.resample_poly`, rejects non-finite or empty audio, clips only after validating finite values, and returns contiguous `float32`.

`segment_waveform()` calculates sample-exact windows and hops, keeps the tail, calculates RMS, and marks silence with `rms < settings.silence_rms_threshold`. It must never pad the stored segment; padding remains the feature processor's responsibility.

- [ ] **Step 4: Add failing upload-limit and FFmpeg tests**

Use `io.BytesIO` and an injected subprocess runner to assert `read_limited_stream()` raises `FILE_TOO_LARGE` before retaining more than `max_bytes + chunk_size`, `decode_audio()` invokes `imageio_ffmpeg.get_ffmpeg_exe()` with an argument list and `shell=False`, output is 16 kHz mono float32, timeout maps to `DECODE_TIMEOUT`, and the randomized temporary input is deleted after both success and failure.

- [ ] **Step 5: Implement bounded read and safe FFmpeg decode**

Use 1 MiB chunks. Validate filename suffix against `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`; do not log or return the name. Write bytes to `NamedTemporaryFile(delete=False, suffix=validated_suffix)`, run:

```python
[
    ffmpeg_exe, "-v", "error", "-nostdin", "-i", temp_path,
    "-t", str(settings.max_duration_seconds + 1),
    "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "pipe:1",
]
```

Convert stdout using `np.frombuffer(..., dtype="<f4")`, enforce exact duration after decode, and unlink in `finally`. Map public failures to `UNSUPPORTED_FORMAT`, `DECODE_FAILED`, `DECODE_TIMEOUT`, `AUDIO_TOO_LONG`, or `INVALID_AUDIO`.

- [ ] **Step 6: Run audio tests**

Run: `.\.venv\Scripts\python -m pytest tests/test_audio.py -v`

Expected: PASS with no real FFmpeg process in unit tests.

- [ ] **Step 7: Commit**

```bash
git add app/audio.py tests/fakes.py tests/test_audio.py
git commit -m "feat: add privacy-safe audio preprocessing"
```

## Task 4: Lazy HuBERT Runtime

**Files:**
- Create: `app/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing runtime tests with injected factories**

```python
import pytest
import numpy as np
import torch

from app.model import EmotionModelRuntime, RAW_LABELS


def test_runtime_loads_once_and_predicts_in_inference_mode(fake_factories) -> None:
    runtime = EmotionModelRuntime(device="cpu", factories=fake_factories)
    first = runtime.predict(np.zeros(16000, dtype=np.float32))
    second = runtime.predict(np.zeros(16000, dtype=np.float32))
    assert fake_factories.model_load_count == 1
    assert fake_factories.model.training is False
    assert fake_factories.model.grad_enabled_values == [False, False]
    assert first.shape == second.shape == (6,)
    assert RAW_LABELS == ("anger", "fear", "happy", "neutral", "sad", "surprise")
    assert torch.tensor(first).sum().item() == pytest.approx(1.0)
```

Add tests for CUDA preference, explicit CPU configuration, invalid six-logit shape, non-finite logits, load failure mapping, and one CPU retry after a simulated CUDA out-of-memory error.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_model.py -v`

Expected: FAIL because `app.model` does not exist.

- [ ] **Step 3: Implement the checkpoint-compatible model**

Implement `HubertClassificationHead` with dense, tanh, dropout, and output projection. Implement `HubertForSpeechClassification(HubertPreTrainedModel)` with `HubertModel`, mean pooling across time, and a six-output classification head using `config.num_class`.

Implement `EmotionModelRuntime` with a lock-protected lazy `_load()`, `Wav2Vec2FeatureExtractor.from_pretrained()`, `HubertForSpeechClassification.from_pretrained()`, `.eval()`, `torch.inference_mode()`, `softmax`, finite/shape validation, and an injectable factory bundle for tests. Do not use `trust_remote_code=True`.

Device resolution order is explicit configured device, CUDA, MPS, then CPU. CUDA out-of-memory clears cache, moves the loaded model to CPU, updates the public device, and retries once; a second failure maps to `INFERENCE_FAILED`.

- [ ] **Step 4: Run model tests and type checks**

Run: `.\.venv\Scripts\python -m pytest tests/test_model.py -v`

Expected: PASS without network access.

Run: `.\.venv\Scripts\python -m mypy app`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/model.py tests/test_model.py tests/fakes.py
git commit -m "feat: add lazy PyTorch HuBERT runtime"
```

## Task 5: Projection, Reliability, Aggregation, and Progress

**Files:**
- Create: `app/analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write failing projection tests**

```python
import numpy as np
import pytest

from app.analyzer import project_probabilities


def test_projection_renormalizes_only_four_target_classes() -> None:
    raw = np.array([0.20, 0.10, 0.30, 0.20, 0.10, 0.10])
    projected = project_probabilities(raw)
    assert projected.probabilities.anger == pytest.approx(0.25)
    assert projected.probabilities.happy == pytest.approx(0.375)
    assert projected.probabilities.neutral == pytest.approx(0.25)
    assert projected.probabilities.sad == pytest.approx(0.125)
    assert projected.excluded_probability == pytest.approx(0.20)
```

Add one exact test for each low-reliability rule: excluded probability over `0.35`, top target under `0.45`, and top-two margin under `0.12`. Boundary values equal to thresholds remain reliable.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_analyzer.py -v`

Expected: FAIL because `app.analyzer` does not exist.

- [ ] **Step 3: Implement pure projection and reliability functions**

Use a fixed raw-index mapping, never model-config label text. Return `ProjectedPrediction` with the four Pydantic probabilities, excluded probability, dominant emotion, and `Reliability`. Reliability reasons are stable codes: `OUTSIDE_FOUR_CLASS_SCOPE`, `LOW_TOP_PROBABILITY`, and `SMALL_TOP_MARGIN`.

- [ ] **Step 4: Write failing aggregate and progress tests**

Create three generated segments: one voiced neutral, one silent, one voiced happy. Assert the silent segment is returned for the timeline but is never passed to the model; overall weights use `sample_count * clip(rms, 0.02, 0.30)`; progress events advance `1..total`; final output includes all segments, voiced ratio, device, and elapsed milliseconds. An all-silent input raises `NO_VOICE`.

- [ ] **Step 5: Implement `EmotionAnalyzer.iter_analysis()`**

The synchronous iterator first yields a `status` event, then one `progress` event per segment, then one `result` event. It calls the runtime only for voiced segments, aggregates target and excluded probabilities with normalized weights, applies the same reliability rules to the aggregate, and uses `time.perf_counter()` for elapsed time. No event contains the original filename or waveform.

- [ ] **Step 6: Run analyzer tests**

Run: `.\.venv\Scripts\python -m pytest tests/test_analyzer.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/analyzer.py tests/test_analyzer.py
git commit -m "feat: add segmented emotion analysis"
```

## Task 6: FastAPI Application and Streaming Boundary

**Files:**
- Create: `app/main.py`
- Create: `tests/test_api.py`
- Create: `tests/test_privacy.py`

- [ ] **Step 1: Write failing health and analysis API tests**

```python
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_does_not_load_model(fake_services) -> None:
    client = TestClient(create_app(services=fake_services))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_status"] == "not_loaded"
    assert fake_services.runtime.load_count == 0


def test_analyze_stream_finishes_with_result(fake_services, wav_bytes) -> None:
    client = TestClient(create_app(services=fake_services))
    with client.stream("POST", "/api/analyze", files={"audio": ("voice.wav", wav_bytes, "audio/wav")}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert events[-1]["type"] == "result"
```

Add tests for model-load status, size rejection, unsupported suffix, all-silent error event, one-analysis-at-a-time `ANALYSIS_BUSY`, static index serving, safe CORS absence, and generic 500 handling.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_api.py -v`

Expected: FAIL because `app.main` does not exist.

- [ ] **Step 3: Implement the application factory**

`create_app(settings=None, services=None)` constructs services only when not injected. Mount `/static` from the package path and return `index.html` at `/`. `GET /api/health` reads runtime state without loading. `POST /api/model/load` explicitly loads and returns safe status. Export `app = create_app()` at module scope for the documented Uvicorn command.

`POST /api/analyze` accepts one `UploadFile`, acquires a bounded semaphore without waiting, performs limited read and decode, and returns `StreamingResponse` over analyzer events serialized with `model_dump_json() + "\n"`. A wrapper catches `AppError` during iteration and yields exactly one `ErrorEvent`; it logs only event name, error code, device, segment count, and elapsed bucket.

Add exception handlers that return `{"error":{"code":...,"message":...}}` for errors before streaming. Do not add permissive CORS middleware because the page and API share one local origin.

- [ ] **Step 4: Write and satisfy privacy tests**

At test runtime, assemble a synthetic sensitive-looking filename from separate name, phone-prefix, and repeated-digit fragments, then force decode failure. Assert that the assembled filename, phone-shaped value, temporary path, byte content, traceback, and prediction values appear in neither logs nor response. Patch `Path.unlink` observation and assert temporary cleanup on timeout and model failure.

- [ ] **Step 5: Run API and privacy tests**

Run: `.\.venv\Scripts\python -m pytest tests/test_api.py tests/test_privacy.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api.py tests/test_privacy.py tests/fakes.py
git commit -m "feat: expose local streaming analysis API"
```

## Task 7: Semantic Chinese Workbench and Responsive Visual System

**Files:**
- Create: `app/static/index.html`
- Create: `app/static/styles.css`
- Create: `app/static/icons/upload.svg`
- Create: `app/static/icons/mic.svg`
- Create: `app/static/icons/pause.svg`
- Create: `app/static/icons/play.svg`
- Create: `app/static/icons/square.svg`
- Create: `app/static/icons/trash-2.svg`
- Create: `app/static/icons/activity.svg`
- Create: `app/static/icons/alert-triangle.svg`
- Create: `tests/test_frontend.py`

- [ ] **Step 1: Write failing static contract tests**

Read `index.html` with `html.parser` or string assertions. Require one `h1` containing `声析`, upload/record segmented buttons, file input accept list, audio player, waveform canvas, record controls, analyze button, clear icon button with `aria-label`, model/device status, result summary, four probability rows, timeline, `aria-live` status, and only relative local stylesheet/script/icon URLs.

- [ ] **Step 2: Run the contract tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_frontend.py -v`

Expected: FAIL because static files do not exist.

- [ ] **Step 3: Implement the semantic page shell**

Create a compact header, two-column `<main>`, upload/record mode control, stable audio workspace, one primary analyze command, icon-only clear command with tooltip and `aria-label`, result summary, four fixed probability rows, and timeline detail region. Keep user-visible text task-oriented; put model limitations in the footer disclosure link rather than explanatory feature copy.

- [ ] **Step 4: Implement the visual system**

Use CSS custom properties:

```css
:root {
  --paper: #f7f6f2;
  --surface: #ffffff;
  --ink: #1f2328;
  --muted: #667085;
  --line: #d9dde3;
  --vermilion: #c63c2f;
  --neutral: #68788a;
  --happy: #39845a;
  --anger: #c63c2f;
  --sad: #5264a7;
  --warning: #a66b18;
  --radius: 8px;
}
```

Desktop uses `grid-template-columns: minmax(0, 1.05fr) minmax(360px, .95fr)`; under 860 px use one column. Define stable dimensions for mode controls, record controls, waveform, probability tracks, timeline rows, loading area, and icon buttons. Use a system Chinese font stack, zero letter spacing, no gradients, no decorative blobs, no nested cards, and no horizontal page scrolling.

- [ ] **Step 5: Add local Lucide assets**

Vendor the exact upstream Lucide SVG files for the named icons, preserve their MIT attribution in `THIRD_PARTY_NOTICES.md` in Task 10, and reference them with relative URLs. Do not hand-draw substitutes or call a CDN.

- [ ] **Step 6: Run static contract tests**

Run: `.\.venv\Scripts\python -m pytest tests/test_frontend.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/static tests/test_frontend.py
git commit -m "feat: build responsive Chinese workbench"
```

## Task 8: Upload, Browser WAV Recording, Player, and Waveform

**Files:**
- Create: `app/static/app.js`
- Modify: `app/static/index.html`
- Modify: `tests/test_frontend.py`
- Create: `tests/e2e/test_workbench.py`

- [ ] **Step 1: Write failing Playwright interaction tests**

Start the app with injected fake services. In Playwright, select a generated WAV fixture and assert filename, formatted size, duration, enabled analyze button, player source, and nonblank waveform pixels. Switch to recording mode with a browser-injected fake `MediaStream`, click record/pause/resume/stop, and assert the timer state, WAV Blob MIME type, enabled analysis, and object URL cleanup after clear.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/e2e/test_workbench.py -k "upload or record" -v`

Expected: FAIL because `app.js` behavior does not exist.

- [ ] **Step 3: Implement one explicit UI state machine**

Use states `empty`, `ready`, `recording`, `paused`, `loading_model`, `analyzing`, `success`, and `error`. A single `renderState()` updates disabled controls, status text, visible panels, and busy attributes. Mode switching is blocked during recording or analysis.

- [ ] **Step 4: Implement upload validation and audio lifecycle**

Validate allowed suffix, 50 MiB size, and decodable browser audio metadata before enabling analysis. Retain the `File` only in memory, use `URL.createObjectURL`, revoke the previous URL on replacement and clear, and never display the full filename in a log or API response.

- [ ] **Step 5: Implement PCM recording and WAV encoding**

Use `getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}})`, Web Audio sample capture, and a pure `encodeWav(float32Chunks, sampleRate)` function that writes RIFF/WAVE PCM16 headers and samples. Stop every media track on stop, clear, mode switch, and page unload. Enforce five minutes in the timer and stop automatically at the limit.

- [ ] **Step 6: Implement Canvas waveform**

Decode selected audio through `AudioContext.decodeAudioData`, downsample peaks to canvas width, respect device pixel ratio, draw a neutral baseline and ink waveform, and redraw via `ResizeObserver`. Keep dimensions stable and show no waveform for invalid audio.

- [ ] **Step 7: Run interaction tests**

Run: `.\.venv\Scripts\python -m pytest tests/e2e/test_workbench.py -k "upload or record" -v`

Expected: PASS in Chromium.

- [ ] **Step 8: Commit**

```bash
git add app/static/app.js app/static/index.html tests/test_frontend.py tests/e2e/test_workbench.py
git commit -m "feat: add private upload and browser recording"
```

## Task 9: Streamed Analysis Results and Timeline Interaction

**Files:**
- Modify: `app/static/app.js`
- Modify: `app/static/index.html`
- Modify: `app/static/styles.css`
- Modify: `tests/e2e/test_workbench.py`

- [ ] **Step 1: Write failing streamed-result tests**

Mock `/api/analyze` with chunk-split NDJSON so JSON lines cross network chunk boundaries. Assert status and progress events update `正在分析第 3/12 段`, result events render the dominant Chinese label, exact four percentage bars, reliability badge, elapsed time, silent segments, and timeline buttons. Assert an error event restores controls and places the Chinese message beside the audio workspace.

- [ ] **Step 2: Run the result tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/e2e/test_workbench.py -k "stream or timeline or error" -v`

Expected: FAIL because result streaming is not implemented.

- [ ] **Step 3: Implement robust NDJSON parsing**

Use `response.body.getReader()`, streaming `TextDecoder`, and a retained text buffer. Parse only complete newline-terminated records; parse the final nonempty buffer at stream end. Dispatch by fixed event type and reject unknown or malformed events with one stable local error.

- [ ] **Step 4: Render results without unsafe HTML**

Use `textContent`, `style.width`, and `document.createElement`; do not insert server values with `innerHTML`. Map fixed emotion IDs to Chinese labels and CSS variables. Low reliability displays `更倾向于…` and the returned fixed reason codes mapped to concise Chinese text.

- [ ] **Step 5: Implement timeline and player synchronization**

Each segment is a fixed-height button with start/end time, color, probability, reliability marker, and silent styling. Clicking sets `audio.currentTime`; `timeupdate` selects the active segment without rebuilding the list. Keyboard activation and focus indicators match pointer behavior.

- [ ] **Step 6: Run result and regression tests**

Run: `.\.venv\Scripts\python -m pytest tests/e2e/test_workbench.py -v`

Expected: PASS.

Run: `.\.venv\Scripts\python -m pytest -q`

Expected: all non-smoke tests PASS.

- [ ] **Step 7: Commit**

```bash
git add app/static/app.js app/static/index.html app/static/styles.css tests/e2e/test_workbench.py
git commit -m "feat: visualize streamed emotion timeline"
```

## Task 10: Scripts, User Documentation, and Open-Source Policy

**Files:**
- Create: `scripts/download_model.py`
- Create: `scripts/smoke_test_model.py`
- Create: `README.md`
- Create: `README_zh-CN.md`
- Create: `docs/architecture.md`
- Create: `docs/api.md`
- Create: `docs/model-limitations.md`
- Create: `docs/privacy.md`
- Create: `LICENSE`
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SECURITY.md`
- Create: `THIRD_PARTY_NOTICES.md`

- [ ] **Step 1: Write documentation contract tests**

Extend `tests/test_privacy.py` to require every listed file, verify both READMEs include install/start/test/privacy/model-limitations/troubleshooting sections, verify the Chinese README is linked first from the English README and vice versa, and assert no documentation contains a real email address, phone number, Windows user path, token-shaped string, or private audio reference.

- [ ] **Step 2: Run the documentation test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_privacy.py -k documentation -v`

Expected: FAIL with missing documentation files.

- [ ] **Step 3: Implement model warm-up and smoke scripts**

`download_model.py` loads `EmotionModelRuntime` and prints only model ID, final state, and device. `smoke_test_model.py` generates a two-second 220 Hz WAV in memory, runs the real pipeline, prints the fixed emotion IDs and probabilities, and removes any temporary file in `finally`. Neither script accepts or prints personal filenames.

- [ ] **Step 4: Write bilingual user documentation**

Document Python versions, virtual environment creation for PowerShell/bash, dependency installation, `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`, first-run 1.1 GB model download, offline behavior after caching, CPU/CUDA selection, upload formats and limits, recording permission, tests, real-model smoke test, privacy lifecycle, limitations, and troubleshooting. Use synthetic examples only.

- [ ] **Step 5: Write architecture, API, limitation, and privacy docs**

Record the exact endpoints, NDJSON event examples, public error codes, six-to-four formula, reliability thresholds, CASIA 1200-sample model-card caveat, acted-speech/domain-shift limitation, temporary file cleanup, no remote inference, no logs of predictions, and prohibited high-risk uses.

- [ ] **Step 6: Add community and license files**

Use the canonical MIT license text with year 2026 and `Project Contributors`, Contributor Covenant without a private contact address, a security policy directing reports through GitHub private vulnerability reporting, contribution commands, and notices for the Apache-2.0 model, Lucide MIT icons, imageio-ffmpeg, FFmpeg, and all direct runtime dependencies.

- [ ] **Step 7: Run documentation and privacy tests**

Run: `.\.venv\Scripts\python -m pytest tests/test_privacy.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts README.md README_zh-CN.md docs LICENSE CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md THIRD_PARTY_NOTICES.md tests/test_privacy.py
git commit -m "docs: complete open-source documentation"
```

## Task 11: CI, Privacy Scanner, and Full Automated Verification

**Files:**
- Create: `scripts/privacy_scan.py`
- Create: `.github/workflows/tests.yml`
- Modify: `pyproject.toml`
- Modify: `tests/test_privacy.py`

- [ ] **Step 1: Write failing scanner tests**

Use a temporary Git-like file list with clean content and fixtures that assemble a private-key header, bearer token, Chinese phone-shaped value, personal email, Windows home path, POSIX home path, audio extension, and checkpoint extension from separate string fragments at test runtime. Assert the scanner reports path plus rule ID without echoing the matched secret. Assert documentation URLs and `maintainer@example.invalid` are allowed only where explicitly expected.

- [ ] **Step 2: Run scanner tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_privacy.py -k scanner -v`

Expected: FAIL because `scripts.privacy_scan` does not exist.

- [ ] **Step 3: Implement privacy scanning**

Scan `git ls-files -z`, reject tracked audio/model/environment files, decode text safely, and apply named patterns. Print only `relative/path: RULE_ID`; never print the matched substring. Support `--staged` using `git diff --cached --name-only -z` and a repository-wide default.

- [ ] **Step 4: Create GitHub Actions workflow**

On pushes and pull requests, use Python 3.11 CPU, install `requirements-dev.txt`, install Chromium for Playwright, run Ruff, mypy, privacy scan, pytest with coverage, and upload no audio or model artifacts. Cache pip only; do not cache Hugging Face weights because regular tests must not access them.

- [ ] **Step 5: Run the complete local gate**

Run: `.\.venv\Scripts\python -m ruff format --check .`

Run: `.\.venv\Scripts\python -m ruff check .`

Run: `.\.venv\Scripts\python -m mypy app scripts`

Run: `.\.venv\Scripts\python scripts/privacy_scan.py`

Run: `.\.venv\Scripts\python -m pytest --cov=app --cov-report=term-missing`

Expected: every command exits 0; coverage includes audio, model, analyzer, API, and frontend contracts; no test downloads the real model.

- [ ] **Step 6: Commit**

```bash
git add scripts/privacy_scan.py .github/workflows/tests.yml pyproject.toml tests/test_privacy.py
git commit -m "ci: enforce tests and privacy checks"
```

## Task 12: Real-Model, Visual, Release, and GitHub Verification

**Files:**
- Modify only files required by failures found during verification.

- [ ] **Step 1: Use the verification skill before any completion claim**

Invoke `superpowers:verification-before-completion` and follow its evidence requirements.

- [ ] **Step 2: Download and smoke-test the real model**

Run: `.\.venv\Scripts\python scripts/download_model.py`

Expected: model reaches `loaded`, reports the selected local device, and emits no local path.

Run: `.\.venv\Scripts\python scripts/smoke_test_model.py`

Expected: one valid four-class probability distribution and no retained temporary audio.

- [ ] **Step 3: Start the local server**

Run: `.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`

Expected: health endpoint is reachable at `http://127.0.0.1:8000/api/health` and the workbench at `http://127.0.0.1:8000/`.

- [ ] **Step 4: Perform visual and interaction QA**

Use Playwright screenshots at 1440x900, 1024x768, 390x844, and 360x800. Confirm nonblank waveform canvas pixels, stable loading/result heights, no overlap, no horizontal scroll, no clipped Chinese text, touch-size controls, correct upload and simulated recording flows, timeline/player synchronization, and no failed asset requests or console errors.

- [ ] **Step 5: Re-run the full release gate**

Run the five commands from Task 11 Step 5, then run `git status --short`, `git diff --check`, and `git log --format=fuller --all`.

Expected: all checks exit 0; worktree is clean after any corrective commit; history uses only `Project Maintainer <maintainer@example.invalid>`; no sensitive strings or generated artifacts are tracked.

- [ ] **Step 6: Create the public GitHub repository and push**

Confirm `gh auth status` is authenticated to the intended account. Create public repository `pytorch-call-emotion-recognition` with description `Local-first Mandarin call emotion recognition with PyTorch and HuBERT`, set source to the current directory, set remote `origin`, and push `main`. Do not enable issues, discussions, pages, or deployments beyond GitHub defaults unless separately requested.

- [ ] **Step 7: Verify the public repository**

Check the remote repository visibility, default branch, README rendering, license detection, Actions run, tracked-file list, and absence of releases/packages/artifacts containing audio or model weights. Record the public URL in the final handoff.

- [ ] **Step 8: Tag the verified initial release**

After CI passes, create annotated tag `v0.1.0` with message `Initial open-source release` and push the tag. Do not create a binary GitHub Release because model weights and audio artifacts must remain outside the repository.
