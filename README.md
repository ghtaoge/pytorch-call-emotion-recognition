# 声析：PyTorch 通话情绪识别

简体中文 | [English](README_en.md)

一个本地优先、面向普通话的 PyTorch 语音情绪识别教学案例。上传通话音频或使用浏览器录音后，页面会展示整体情绪、四类概率和分段时间轴。

> 本项目用于技术学习和交互演示。模型不能替代人工判断，不应用于医疗诊断、员工绩效、风控、自动处罚等高风险场景。

## 项目简介

**声析** 是一个完全在本地运行的普通话通话语音情绪识别系统，基于 PyTorch 和 Hugging Face Transformers 实现。项目名称"声析"寓意"以声析情"——通过声音来感知情绪。

项目核心采用 **HuBERT（Hidden-Unit BERT）** 预训练模型作为特征提取器。HuBERT 是 Facebook 提出的自监督语音表征学习模型，与 wav2vec 2.0 类似但训练策略不同——它先通过离线聚类生成离散标签作为"隐藏单元"，再以 BERT 式掩码预测目标进行预训练，在语音情感识别等下游任务上表现出色。本项目使用的检查点 `xmj2002/hubert-base-ch-speech-emotion-recognition` 专门适配普通话语音，在大约 6 类表演语音语料上微调完成。

**四分类投影策略** 是本项目的关键设计。原始模型输出六个类别：愤怒、恐惧、开心、中性、悲伤、惊讶。但通话场景中最有实际意义的是四类——中性、开心、愤怒、悲伤。项目并非简单丢弃恐惧和惊讶，而是将两类概率单独计算为 `excluded_probability`，并据此评估结果可靠性：

- 当被排除类别概率超过 35% 时，标记为"超出四分类范围"
- 当最高概率低于 45% 时，标记为"最高概率不足"
- 当最高与次高概率差距小于 12% 时，标记为"类别区分不足"
- 当有声段占比低于 30% 时，标记为"语音覆盖率不足"

任何一个可靠性标记触发，整体结论就会标注为"谨慎参考"，而非给出虚假的高置信结果。这种诚实降级机制避免了模型在不适用的场景下产生误导性的"确定性"。

## 功能

- 普通话 HuBERT 预训练模型，本地 PyTorch 推理，不调用任何远程 API
- 上传 WAV、MP3、FLAC、OGG、M4A、WebM 格式音频
- 浏览器麦克风录音、播放器和波形预览
- 中性、开心、愤怒、悲伤四类概率分布
- 6 秒窗口、5 秒步长的情绪分段时间轴
- 诚实降级：超出四分类范围、概率不足或类别模糊时标注"谨慎参考"
- CPU、CUDA、MPS 自动选择；CUDA 显存不足时自动回退 CPU
- 音频不保存、不上传云端、不建立历史记录
- 中文详细代码注释、单元测试和 GitHub Actions 持续集成

## 技术架构

### 处理流程

音频从浏览器上传或录音开始，经过以下步骤完成分析：

```text
浏览器上传 / 录音
       |
       v
FastAPI 接收 ──→ 大小校验（≤ 50 MB）──→ 临时文件解码（FFmpeg）
                                               |
                                               v
                                       16 kHz 单声道波形归一化
                                               |
                                               v
                                       6 秒窗口 / 5 秒步长分段 ──→ 静音过滤
                                               |
                                               v
                                       HuBERT 六分类 PyTorch 推理
                                               |
                                               v
                                       六→四分类投影 + 可靠性评估
                                               |
                                               v
                                       按时长与能量加权聚合整体结论
                                               |
                                               v
                                       NDJSON 流式进度与结果 ──→ 页面呈现
```

### 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| 音频处理 | `app/audio.py` | 文件大小限制、FFmpeg 解码、16 kHz 重采样、多声道合并、静音检测和重叠窗口分段 |
| 模型推理 | `app/model.py` | HuBERT 分类结构复现、权重延迟加载、设备自动选择、PyTorch 推理与 CUDA 回退 |
| 情绪分析 | `app/analyzer.py` | 六→四概率投影、可靠性评估、按时长与 RMS 能量加权聚合 |
| 环境配置 | `app/config.py` | 参数验证、`.env` 读取、模型 ID、窗口时长、最大音频限制等 |
| 数据模型 | `app/schemas.py` | Pydantic 严格模式契约，保证 API 输出不含多余字段 |
| 错误处理 | `app/errors.py` | 统一异常类，仅返回安全公开信息，不含路径或堆栈 |
| HTTP 接口 | `app/main.py` | FastAPI 路由、单任务并发锁、NDJSON 流式传输 |
| 前端工作台 | `app/static/` | 中文单页界面：录音、上传、波形预览、实时进度和情绪时间轴 |

### 关键设计原则

- **延迟加载**：服务启动不下载模型；首次分析或调用 `/api/model/load` 时才加载约 1.1 GB 权重
- **线程安全**：模型加载使用双重检查锁，分析接口使用并发锁保证同一时间只有一个推理任务
- **临时文件安全**：压缩音频解码使用随机命名临时文件，无论成功或失败都在 `finally` 中立即删除
- **内存防护**：上传流分块读取，不依赖 HTTP 声明的 Content-Length；CUDA 显存不足时自动回退 CPU 重试
- **诚实降级**：可靠性机制从不隐藏模型的局限性，在不适用的场景下明确标注"谨慎参考"

## API 接口

所有接口仅在本机 `127.0.0.1` 上可用，无需认证。

### `GET /`

中文前端工作台页面。

### `GET /api/health`

返回服务状态、模型加载状态和推理设备，不触发模型下载。

```json
{"status": "ok", "model_status": "loaded", "device": "cuda"}
```

### `POST /api/model/load`

显式加载模型到内存。首次调用可能下载约 1.1 GB 权重，后续调用检查已加载后直接返回状态。

### `POST /api/analyze`

核心分析接口。字段 `audio` 为 multipart 上传文件。响应类型为 `application/x-ndjson`，每行一个 JSON 事件，流式返回进度和最终结果：

```json
{"type":"status","current":0,"total":8,"message":"模型准备中"}
{"type":"progress","current":2,"total":8,"message":"正在分析第 2/8 段"}
{"type":"progress","current":8,"total":8,"message":"正在分析第 8/8 段"}
{"type":"result","result":{"dominant_emotion":"neutral","probabilities":{"neutral":0.62,"happy":0.11,"anger":0.15,"sad":0.12},"reliability":{"level":"high","reasons":[]},"excluded_probability":0.08,"voiced_ratio":0.91,"duration_seconds":42.5,"device":"cuda","elapsed_ms":3200,"segments":[...]}}
```

**错误码**：`EMPTY_FILE`、`FILE_TOO_LARGE`、`UNSUPPORTED_FORMAT`、`DECODE_FAILED`、`DECODE_TIMEOUT`、`AUDIO_TOO_LONG`、`NO_VOICE`、`MODEL_LOAD_FAILED`、`INFERENCE_FAILED`、`ANALYSIS_BUSY`。响应不含文件名、路径、音频内容或堆栈信息。

详见 [API 文档](docs/api.md)。

## 环境要求

- Python 3.11、3.12 或 3.13
- 首次下载模型约需 1.1 GB 磁盘空间和可访问 Hugging Face 的网络
- 建议至少 8 GB 内存；使用 CUDA 可明显缩短分析时间

## 安装

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r requirements.txt
```

## 启动

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。首次分析会下载模型；也可以提前执行：

```powershell
.\.venv\Scripts\python scripts\download_model.py
```

## 测试

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check app tests scripts
.\.venv\Scripts\python scripts\privacy_scan.py
```

常规测试使用假模型和程序生成波形，不下载权重。真实模型冒烟测试：

```powershell
.\.venv\Scripts\python scripts\smoke_test_model.py
.\.venv\Scripts\python scripts\smoke_test_api.py
```

## 隐私

浏览器只把音频发送到当前计算机上的 FastAPI 服务。服务不调用远程推理 API，不保存音频或分析结果。压缩音频解码时使用随机临时文件，并在成功或失败后立即删除。首次模型下载是分析流程之外唯一需要联网的步骤。服务没有数据库，刷新页面不会保留结果。详见 [隐私说明](docs/privacy.md)。

## 模型与四分类

默认模型原生输出六类：愤怒、恐惧、开心、中性、悲伤、惊讶。本项目只展示中性、开心、愤怒、悲伤四类，并单独计算被排除的恐惧与惊讶概率。如果排除概率较高、最高概率较低或前两名接近，页面会标记"谨慎参考"，而不是给出虚假的高置信结论。

训练数据是规模有限的表演语音，真实电话、方言、噪声、多人重叠和跨设备录音会产生明显域偏移。详见 [模型限制](docs/model-limitations.md)。

## 常见问题

**首次分析很慢**：模型约 1.1 GB，需要下载并加载到内存。后续运行使用本机缓存。

**提示未检测到人声**：提高录音音量、减少背景噪声，或上传包含连续说话的片段。

**M4A 无法读取**：确认文件未损坏；项目通过随依赖提供的 FFmpeg 组件解码。

**CUDA 显存不足**：服务会自动回退到 CPU 重试一次，耗时会增加。

**录音按钮不可用**：允许浏览器访问麦克风，并使用 `localhost` 或 `127.0.0.1` 打开页面。

## 项目结构

```text
app/audio.py       音频限制、解码、重采样和分段
app/model.py       HuBERT 结构、权重加载和 PyTorch 推理
app/analyzer.py    四分类投影、可靠性和整体聚合
app/config.py      环境配置和参数验证
app/schemas.py     API 数据模型和合同验证
app/errors.py      统一错误处理
app/main.py        FastAPI 与 NDJSON 流式接口
app/static/        中文前端工作台 (index.html, styles.css, app.js)
tests/             无网络自动化测试
docs/              架构、API、隐私和模型限制
scripts/           模型预下载、冒烟测试和隐私扫描
```

## 开源协作

项目采用 [MIT License](LICENSE)。提交变更前请阅读 [贡献指南](CONTRIBUTING.md)、[行为准则](CODE_OF_CONDUCT.md) 和 [安全策略](SECURITY.md)。第三方许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
