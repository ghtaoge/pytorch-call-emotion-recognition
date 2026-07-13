from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, HubertModel, HubertPreTrainedModel, Wav2Vec2FeatureExtractor

from app.config import Settings
from app.errors import AppError

RAW_LABELS = ("anger", "fear", "happy", "neutral", "sad", "surprise")


class HubertClassificationHead(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(float(getattr(config, "classifier_dropout", 0.1)))
        self.out_proj = nn.Linear(hidden_size, int(getattr(config, "num_class", 6)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.dropout(torch.tanh(self.dense(values))))


class HubertForSpeechClassification(HubertPreTrainedModel):
    """复现权重模型卡中的均值池化分类结构，不启用远程自定义代码。"""

    def __init__(self, config: Any) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.hubert = HubertModel(config)  # type: ignore[arg-type]
        self.classifier = HubertClassificationHead(config)
        self.post_init()

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.hubert(input_values).last_hidden_state
        return self.classifier(hidden_states.mean(dim=1))


class EmotionModelRuntime:
    """线程安全地延迟加载模型；页面启动不会立即下载约 1.1 GB 权重。"""

    def __init__(self, settings: Settings, device: str | None = None) -> None:
        self.settings = settings
        self.device = device or self._select_device()
        self.status = "not_loaded"
        self._processor: Wav2Vec2FeatureExtractor | None = None
        self._model: HubertForSpeechClassification | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self.status = "loading"
            try:
                try:
                    processor, model = self._load_components(local_files_only=True)
                except OSError:
                    # 缓存不完整时才访问模型站点，避免离线运行被多次网络重试拖慢。
                    processor, model = self._load_components(local_files_only=False)
                model.eval().to(self.device)
                self._processor = processor
                self._model = model
                self.status = "loaded"
            except Exception as exc:
                self.status = "error"
                raise AppError(
                    "MODEL_LOAD_FAILED", "模型加载失败，首次运行请检查网络后重试", 503
                ) from exc

    def _load_components(
        self, *, local_files_only: bool
    ) -> tuple[Wav2Vec2FeatureExtractor, HubertForSpeechClassification]:
        config = AutoConfig.from_pretrained(
            self.settings.model_id, local_files_only=local_files_only
        )
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.settings.model_id, local_files_only=local_files_only
        )
        model = HubertForSpeechClassification.from_pretrained(
            self.settings.model_id,
            config=config,
            local_files_only=local_files_only,
        )
        return processor, model

    def predict(self, waveform: np.ndarray) -> np.ndarray:
        """返回固定顺序的六类概率；调用方不能依赖模型配置里的自由文本标签。"""
        self.load()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(
            waveform,
            sampling_rate=16_000,
            padding="max_length",
            truncation=True,
            max_length=round(self.settings.window_seconds * 16_000),
            return_tensors="pt",
        ).input_values.to(self.device)
        try:
            with torch.inference_mode():
                logits = self._model(inputs)
                probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        except torch.OutOfMemoryError as exc:
            if self.device != "cuda":
                raise AppError("INFERENCE_FAILED", "设备内存不足，无法完成分析", 503) from exc
            torch.cuda.empty_cache()
            self.device = "cpu"
            self._model.to("cpu")
            return self.predict(waveform)
        except Exception as exc:
            raise AppError("INFERENCE_FAILED", "情绪分析失败，请重试", 500) from exc
        if probabilities.shape != (6,) or not np.isfinite(probabilities).all():
            raise AppError("INVALID_MODEL_OUTPUT", "模型输出格式异常", 500)
        return probabilities.astype(np.float64)
