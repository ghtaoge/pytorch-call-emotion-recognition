# Shengxi: PyTorch Call Emotion Recognition

English | [简体中文](README_zh-CN.md)

A local-first PyTorch example for Mandarin speech emotion recognition. Upload a call recording or record in the browser to see an overall four-emotion distribution and a segmented timeline.

> This project is for technical education and demonstration. Do not use it for medical diagnosis, employment decisions, risk scoring, punishment, or other high-impact decisions.

## Features

- Local PyTorch inference with a Mandarin HuBERT checkpoint
- Upload and browser recording workflows
- Neutral, happy, anger, and sad probabilities
- Six-second segments with a five-second hop
- Automatic CUDA, MPS, or CPU selection
- No accounts, database, telemetry, remote inference, or result history
- Chinese teaching comments, tests, and GitHub Actions

## Requirements

- Python 3.11 through 3.13
- About 1.1 GB of disk space and network access for the first model download
- 8 GB RAM recommended

## Install and Run

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux: ./.venv/bin/python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. To warm the model cache first, run `python scripts/download_model.py`.

## Test

```bash
python -m pytest -q
python -m ruff check app tests scripts
python scripts/privacy_scan.py
```

Routine tests use generated waveforms and fake model outputs. `python scripts/smoke_test_model.py` checks the real model and `python scripts/smoke_test_api.py` checks the complete decode/API path.

## Privacy and Limitations

Audio is sent only to the FastAPI process on your computer. It is not persisted or sent to a remote inference API. Temporary decode files use random names and are deleted in `finally` cleanup. See [privacy](docs/privacy.md) and [model limitations](docs/model-limitations.md).

The checkpoint was trained on a small acted-speech corpus and natively predicts six emotions. This app transparently projects four target classes and marks results as low reliability when excluded classes or ambiguity are high.

## Troubleshooting

- First analysis is slow: wait for the model download and load to finish.
- No voice detected: use a clearer, louder speech segment.
- Microphone unavailable: allow browser permission and use localhost.
- CUDA out of memory: the runtime retries once on CPU.

## Open Source

MIT licensed. Read [Contributing](CONTRIBUTING.md), [Code of Conduct](CODE_OF_CONDUCT.md), [Security](SECURITY.md), and [Third-party notices](THIRD_PARTY_NOTICES.md).
