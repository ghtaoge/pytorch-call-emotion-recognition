# 声析：PyTorch 通话情绪识别

[English](README.md) | 简体中文

一个本地优先、面向普通话的 PyTorch 语音情绪识别教学案例。上传通话音频或使用浏览器录音后，页面会展示整体情绪、四类概率和分段时间轴。

> 本项目用于技术学习和交互演示。模型不能替代人工判断，不应用于医疗诊断、员工绩效、风控、自动处罚等高风险场景。

## 功能

- 普通话 HuBERT 预训练模型，本地 PyTorch 推理
- 上传 WAV、MP3、FLAC、OGG、M4A、WebM
- 浏览器麦克风录音、播放器和波形预览
- 中性、开心、愤怒、悲伤四类分布
- 6 秒窗口、5 秒步长的情绪时间轴
- CPU、CUDA、MPS 自动选择
- 音频不保存、不上传云端、不建立历史记录
- 中文详细代码注释、单元测试和 GitHub Actions

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

浏览器只把音频发送到当前计算机上的 FastAPI 服务。服务不调用远程推理 API，不保存音频或分析结果。压缩音频解码时使用随机临时文件，并在成功或失败后立即删除。首次模型下载是分析流程之外唯一需要联网的步骤。详见 [隐私说明](docs/privacy.md)。

## 模型与四分类

默认模型原生输出愤怒、恐惧、开心、中性、悲伤、惊讶六类。本项目只展示中性、开心、愤怒、悲伤，并单独计算被排除的恐惧与惊讶概率。如果排除概率较高、最高概率较低或前两名接近，页面会标记“谨慎参考”，而不是给出虚假的高置信结论。

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
app/main.py        FastAPI 与 NDJSON 流式接口
app/static/        中文前端工作台
tests/             无网络自动化测试
docs/              架构、API、隐私和模型限制
scripts/           模型预下载、冒烟测试和隐私扫描
```

## 开源协作

项目采用 [MIT License](LICENSE)。提交变更前请阅读 [贡献指南](CONTRIBUTING.md)、[行为准则](CODE_OF_CONDUCT.md) 和 [安全策略](SECURITY.md)。第三方许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
