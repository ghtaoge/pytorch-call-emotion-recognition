"""
app.model — 模型加载与推理运行时

本模块实现了基于 HuBERT 的语音情绪分类模型的加载、推理与错误恢复机制。

核心架构：
1. HubertClassificationHead — 分类头，将 HuBERT 编码器的隐藏状态映射为六类情绪概率
   结构：Dense → Tanh → Dropout → Linear，采用均值池化聚合帧级特征
2. HubertForSpeechClassification — 完整分类模型，组合 HuBERT 编码器与分类头
   前向流程：原始波形 → HuBERT 编码 → 均值池化 → 分类头 → logits
3. EmotionModelRuntime — 运行时管理器，封装模型的生命周期与推理调用
   关键特性：
   - 延迟加载：服务启动时不立即下载约 1.1 GB 权重，首次推理时才触发加载
   - 双重检查锁定（DCLP）：保证多线程环境下只加载一次模型
   - 设备自动选择：优先 CUDA > MPS > CPU
   - CUDA OOM 自动降级：推理时若 GPU 内存不足，自动回退到 CPU 并重试
   - 本地优先加载：优先从缓存加载，缓存不完整才访问网络，避免离线环境被重试拖慢

线程安全设计：
- EmotionModelRuntime._lock 是 threading.Lock，用于双重检查锁定模式
- load() 方法使用 DCLP（外层检查 + 内层加锁检查），避免不必要的锁竞争
- predict() 方法本身不加锁，因为模型加载完成后 _model 对象是不可变的引用

注意：本模块复现了模型卡中的均值池化分类结构，不依赖远程自定义代码，
保证模型加载行为完全可控，不受 HuggingFace 仓库代码变更的影响。
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, HubertModel, HubertPreTrainedModel, Wav2Vec2FeatureExtractor

from app.config import Settings
from app.errors import AppError

# 原始六类情绪标签，固定顺序 — 对应模型输出的 logits 位置索引
# 注意：调用方不能依赖模型配置中的自由文本标签（可能被仓库作者修改），
# 必须使用此硬编码顺序来解析模型输出
RAW_LABELS = ("anger", "fear", "happy", "neutral", "sad", "surprise")


class HubertClassificationHead(nn.Module):
    """
    HuBERT 分类头 — 将编码器的隐藏状态映射为情绪类别 logits

    结构：Linear(hidden_size → hidden_size) → Tanh → Dropout → Linear(hidden_size → num_class)

    设计说明：
    - Tanh 激活：对中间表示做非线性变换，相比 ReLU 更适合情感特征的全局归一化
    - Dropout：使用 classifier_dropout（默认 0.1），训练时防止过拟合，
      推理时自动关闭（model.eval() 后 dropout 不生效）
    - 输入为均值池化后的帧级特征（维度 = hidden_size），而非逐帧特征，
      均值池化保证了时间维度上的全局语义聚合
    """

    def __init__(self, config: Any) -> None:
        """
        初始化分类头

        参数：
            config — 模型配置对象，包含以下关键字段：
                - hidden_size: 编码器隐藏层维度（HuBERT-Base 为 768）
                - classifier_dropout: 分类头 dropout 比率，默认 0.1
                - num_class: 分类类别数，默认 6（六类情绪）
        """
        super().__init__()
        hidden_size = int(config.hidden_size)
        # 第一层线性变换：将隐藏状态投影到同维度空间，用于特征重组
        self.dense = nn.Linear(hidden_size, hidden_size)
        # Dropout 层：训练时随机丢弃 10% 的神经元，推理时自动关闭
        # getattr 用于兼容不同版本的 config 对象（某些版本可能缺少 classifier_dropout 字段）
        self.dropout = nn.Dropout(float(getattr(config, "classifier_dropout", 0.1)))
        # 输出投影层：将重组后的特征映射为六类情绪的 logits
        self.out_proj = nn.Linear(hidden_size, int(getattr(config, "num_class", 6)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        前向传播：Dense → Tanh → Dropout → OutProj

        参数：
            values — 均值池化后的隐藏状态，形状 (batch_size, hidden_size)

        返回：
            logits — 各类别的未归一化得分，形状 (batch_size, num_class)
            调用方需自行使用 softmax 转换为概率
        """
        return self.out_proj(self.dropout(torch.tanh(self.dense(values))))


class HubertForSpeechClassification(HubertPreTrainedModel):
    """
    HuBERT 语音情绪分类模型 — 编码器 + 均值池化 + 分类头

    复现权重模型卡中的均值池化分类结构，不启用远程自定义代码。
    此类继承 HubertPreTrainedModel 以获得 from_pretrained() 等标准加载能力，
    但前向逻辑完全由本类自行实现，不依赖 HuggingFace Hub 上的远程代码。

    前向流程：
    1. 原始波形 → Wav2Vec2FeatureExtractor 预处理（归一化、填充、截断）
    2. 预处理结果 → HuBERT 编码器 → last_hidden_state (batch, frames, hidden_size)
    3. 均值池化：对帧维度求平均 → (batch, hidden_size)
    4. 分类头：hidden_size → logits (batch, num_class)
    """

    def __init__(self, config: Any) -> None:
        """
        初始化分类模型

        参数：
            config — AutoConfig 加载的模型配置对象
        """
        super().__init__(config)  # type: ignore[arg-type]
        # HuBERT 编码器：将原始波形编码为帧级隐藏状态
        self.hubert = HubertModel(config)  # type: ignore[arg-type]
        # 分类头：将均值池化后的隐藏状态映射为情绪类别 logits
        self.classifier = HubertClassificationHead(config)
        # post_init() 执行权重初始化与安全检查（Pydantic 预训练模型的标准流程）
        self.post_init()

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播：编码 → 均值池化 → 分类

        参数：
            input_values — 特征提取器预处理后的波形张量，形状 (batch, sequence_length)

        返回：
            logits — 各类别的未归一化得分，形状 (batch, num_class)

        算法步骤：
        1. HuBERT 编码器提取帧级隐藏状态 → (batch, frames, hidden_size)
        2. 对帧维度（dim=1）求均值 → (batch, hidden_size)，实现全局语义聚合
        3. 分类头将聚合特征映射为六类 logits → (batch, 6)

        注意：均值池化比最大池化更适合语音情绪任务，
        因为情绪特征通常均匀分布在整个片段中而非集中在少数帧。
        """
        hidden_states = self.hubert(input_values).last_hidden_state
        # 对帧维度求均值：将变长帧序列压缩为固定维度向量
        return self.classifier(hidden_states.mean(dim=1))


class EmotionModelRuntime:
    """
    模型运行时管理器 — 封装模型生命周期与推理调用

    线程安全地延迟加载模型；页面启动不会立即下载约 1.1 GB 权重。

    线程安全设计：
    - 使用 threading.Lock + 双重检查锁定模式（DCLP）保证多线程环境下只加载一次
    - load() 方法：外层检查 _model is not None（快速路径，无需加锁）
      → 内层加锁再次检查 _model is not None（防止多线程同时通过外层检查）
    - predict() 方法本身不加锁：模型加载完成后 _model 和 _processor 是不可变引用，
      推理过程只读不写，天然线程安全（PyTorch model.eval() + inference_mode）

    延迟加载策略：
    - 服务启动时仅创建 EmotionModelRuntime 实例，不加载模型权重
    - 首次调用 predict() 时触发 load()
    - load() 优先尝试 local_files_only=True（纯本地加载，零网络延迟）
      若本地缓存不完整则回退到 local_files_only=False（允许下载）

    设备选择与错误恢复：
    - 设备优先级：CUDA（GPU 加速） > MPS（Apple Silicon） > CPU（兜底）
    - CUDA OOM 自动降级：推理时若 GPU 内存不足，
      清空 CUDA 缓存 → 将模型迁移到 CPU → 使用 CPU 重试推理
    - 降级是永久性的：device 字段被修改为 "cpu"，后续推理均使用 CPU
    """

    def __init__(self, settings: Settings, device: str | None = None) -> None:
        """
        初始化运行时管理器

        参数：
            settings — 全局配置对象，提供 model_id 和 window_seconds 等参数
            device — 可选设备覆盖，若未指定则自动选择最优设备

        注意：初始化时不加载模型，仅记录配置和设备选择结果。
        """
        self.settings = settings
        # 设备选择：优先使用调用方指定的设备，否则自动选择
        self.device = device or self._select_device()
        # 模型状态标识，供 API 健康检查端点读取
        # 可能值："not_loaded" / "loading" / "loaded" / "error"
        self.status = "not_loaded"
        # 特征提取器：延迟加载，推理时使用
        self._processor: Wav2Vec2FeatureExtractor | None = None
        # 分类模型：延迟加载，推理时使用
        self._model: HubertForSpeechClassification | None = None
        # 线程锁：用于双重检查锁定模式，保证 load() 只执行一次
        self._lock = threading.Lock()

    @staticmethod
    def _select_device() -> str:
        """
        自动选择最优推理设备

        设备优先级：
        1. "cuda" — NVIDIA GPU，推理速度最快，但内存有限
        2. "mps" — Apple Metal Performance Shaders，M1/M2 芯片的 GPU 加速
        3. "cpu" — 兜底方案，速度最慢但内存充足且始终可用

        返回：
            设备标识字符串："cuda" / "mps" / "cpu"
        """
        # 优先 CUDA：NVIDIA GPU 提供 10-50 倍推理加速
        if torch.cuda.is_available():
            return "cuda"
        # 次选 MPS：Apple Silicon GPU，速度介于 CUDA 和 CPU 之间
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        # 兜底 CPU：始终可用，但推理速度较慢
        return "cpu"

    def load(self) -> None:
        """
        加载模型权重与特征提取器

        使用双重检查锁定模式（Double-Checked Locking Pattern）保证线程安全：

        1. 外层检查 _model is not None — 快速路径，模型已加载则直接返回
           大多数调用（模型已加载后）走此路径，无需获取锁，零性能开销
        2. 内层加锁 + 再次检查 _model is not None — 防止多线程竞争
           仅首次加载时走此路径，获取锁后再次确认，避免重复加载
        3. 状态管理 — loading → loaded/error，供健康检查端点读取

        本地优先策略：
        - 首次尝试 local_files_only=True（纯本地加载，零网络延迟）
        - 若抛出 OSError（缓存不完整），回退到 local_files_only=False
          允许从 HuggingFace Hub 下载缺失文件
        - 此策略保证离线环境不会被多次网络重试拖慢

        异常处理：
        - 任何加载异常都将状态设为 "error" 并抛出 AppError(503)
        - 调用方可根据 status 字段判断是否需要重试
        """
        # 快速路径：模型已加载，无需获取锁
        if self._model is not None:
            return
        # 加锁路径：保证多线程环境下只加载一次
        with self._lock:
            # 内层检查：获取锁后再次确认，防止竞争
            if self._model is not None:
                return
            # 标记加载状态，供健康检查端点读取
            self.status = "loading"
            try:
                # 本地优先策略：先尝试纯本地加载
                try:
                    processor, model = self._load_components(local_files_only=True)
                except OSError:
                    # 缓存不完整时才访问模型站点，避免离线运行被多次网络重试拖慢。
                    processor, model = self._load_components(local_files_only=False)
                # 切换到推理模式：关闭 dropout 等训练专用层
                model.eval().to(self.device)
                self._processor = processor
                self._model = model
                self.status = "loaded"
            except Exception as exc:
                # 加载失败：标记错误状态，抛出 503（服务暂时不可用）
                self.status = "error"
                raise AppError(
                    "MODEL_LOAD_FAILED", "模型加载失败，首次运行请检查网络后重试", 503
                ) from exc

    def _load_components(
        self, *, local_files_only: bool
    ) -> tuple[Wav2Vec2FeatureExtractor, HubertForSpeechClassification]:
        """
        加载模型组件：配置、特征提取器与分类模型

        参数：
            local_files_only — 是否仅从本地缓存加载：
                True: 不访问网络，适合离线环境或缓存已完整的场景
                False: 允许从 HuggingFace Hub 下载缺失文件

        返回：
            (processor, model) 元组：
            - processor: 特征提取器，用于波形预处理（归一化、填充、截断）
            - model: 分类模型，包含 HuBERT 编码器与分类头

        注意：三个组件使用相同的 local_files_only 参数，
        保证要么全部从本地加载，要么全部允许网络下载，
        避免部分组件走本地、部分走网络的不一致状态。
        """
        # 加载模型配置：定义隐藏层维度、分类数等结构参数
        config = AutoConfig.from_pretrained(
            self.settings.model_id, local_files_only=local_files_only
        )
        # 加载特征提取器：用于将原始波形转换为模型输入格式
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.settings.model_id, local_files_only=local_files_only
        )
        # 加载分类模型：包含 HuBERT 编码器 + 自定义分类头
        # 使用自定义的 HubertForSpeechClassification 类，不依赖远程代码
        model = HubertForSpeechClassification.from_pretrained(
            self.settings.model_id,
            config=config,
            local_files_only=local_files_only,
        )
        return processor, model

    def predict(self, waveform: np.ndarray) -> np.ndarray:
        """
        执行单段音频的情绪推理

        返回固定顺序的六类概率；调用方不能依赖模型配置里的自由文本标签。

        参数：
            waveform — 单声道 16kHz 浮点波形，形状 (sample_count,)
                      通常为滑动窗口切分后的片段（约 6 秒 ≈ 96000 采样点）

        返回：
            probabilities — 六类情绪概率数组，形状 (6,), dtype=float64
            顺序固定为 [anger, fear, happy, neutral, sad, surprise]

        算法步骤：
        1. 触发模型加载（若尚未加载）
        2. 特征提取器预处理：归一化 → 填充至 max_length → 截断 → 转为 PyTorch 张量
        3. 模型推理：inference_mode 下执行前向传播，关闭梯度计算以节省内存
        4. Softmax 归一化：将 logits 转换为概率分布
        5. 转移到 CPU 并转为 numpy 数组

        错误恢复 — CUDA OOM 自动降级：
        1. 捕获 torch.OutOfMemoryError
        2. 若当前设备不是 CUDA，直接抛出 AppError（CPU 不应出现 OOM）
        3. 清空 CUDA 缓存（torch.cuda.empty_cache()）释放碎片内存
        4. 将 device 永久修改为 "cpu"
        5. 将模型迁移到 CPU（self._model.to("cpu")）
        6. 使用 CPU 重试推理（递归调用 self.predict()）
        7. 降级是永久性的，后续所有推理均使用 CPU

        输出校验：
        - 检查概率数组形状为 (6,)，防止模型结构异常
        - 检查所有值为有限数（np.isfinite），防止 NaN/Inf 进入下游逻辑
        """
        # 延迟加载：首次调用时触发模型加载
        self.load()
        assert self._processor is not None and self._model is not None
        # 特征提取器预处理：
        # - sampling_rate=16000: 匹配模型预训练时的采样率
        # - padding="max_length": 填充至固定长度，保证所有片段输入形状一致
        # - truncation=True: 超长片段截断，避免超出模型最大输入长度
        # - max_length: 等于 window_seconds * 16000，即窗口长度对应的采样点数
        inputs = self._processor(
            waveform,
            sampling_rate=16_000,
            padding="max_length",
            truncation=True,
            max_length=round(self.settings.window_seconds * 16_000),
            return_tensors="pt",
        ).input_values.to(self.device)
        try:
            # inference_mode: 关闭梯度计算与自动求导，节省约 50% GPU 内存
            with torch.inference_mode():
                logits = self._model(inputs)
                # softmax 归一化：将 logits 转换为概率分布（各类概率之和 = 1）
                probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        except torch.OutOfMemoryError as exc:
            # CUDA OOM 自动降级逻辑：
            # 1. 若非 CUDA 设备出现 OOM（理论上不应发生），直接报错
            if self.device != "cuda":
                raise AppError("INFERENCE_FAILED", "设备内存不足，无法完成分析", 503) from exc
            # 2. 清空 CUDA 缓存，释放碎片内存
            torch.cuda.empty_cache()
            # 3. 永久降级到 CPU
            self.device = "cpu"
            self._model.to("cpu")
            # 4. 使用 CPU 重试推理
            return self.predict(waveform)
        except Exception as exc:
            # 其他推理异常：统一抛出 AppError(500)
            raise AppError("INFERENCE_FAILED", "情绪分析失败，请重试", 500) from exc
        # 输出校验：防止模型结构异常导致下游逻辑出错
        if probabilities.shape != (6,) or not np.isfinite(probabilities).all():
            raise AppError("INVALID_MODEL_OUTPUT", "模型输出格式异常", 500)
        # 转为 float64：下游加权聚合使用 float64 精度，避免多次累加的浮点误差
        return probabilities.astype(np.float64)
