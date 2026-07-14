"""
app.audio — 音频解码、重采样与分段切分模块

本模块负责将上传的音频文件从各种格式解码为统一的单声道 16kHz 浮点波形，
并将其按滑动窗口切分为适合模型推理的片段。

核心流程：
1. read_limited_stream: 从上传流中分块读取数据，防止恶意大文件耗尽内存
2. decode_audio: 使用 FFmpeg subprocess 将各种音频格式解码为 PCM 浮点波形
3. normalize_waveform: 统一声道数、采样率与数值范围，使输入符合 HuBERT 预训练约定
4. segment_waveform: 按重叠滑动窗口切分长音频，尾段不补零

关键设计决策：
- 使用 FFmpeg subprocess 而非 Python 音频库：FFmpeg 支持几乎所有音频格式，
  且解码速度远快于纯 Python 实现。通过 imageio_ffmpeg 获取 FFmpeg 二进制路径，
  避免要求用户自行安装 FFmpeg
- 临时文件策略：将上传数据写入临时文件供 FFmpeg 输入，解码完成后立即删除，
  保证含原音频的随机临时文件必定清理（finally 块 + suppress(FileNotFoundError))
- 重采样算法：使用 scipy.signal.resample_poly（多相滤波器），比简单线性插值
  产生更少的频谱混叠，保证重采样后的波形质量
- 静音检测：基于 RMS 阈值（0.01 ≈ -40 dB），低于此值的片段视为静音并跳过推理
- 分段策略：滑动窗口（默认 6 秒） + 步长（默认 5 秒） = 1 秒重叠，
  重叠区提供上下文过渡信息；尾段不补零，留给特征处理器统一填充
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import imageio_ffmpeg
import numpy as np
from scipy.signal import resample_poly

from app.config import Settings
from app.errors import AppError

# 支持的音频文件扩展名集合 — 对应 FFmpeg 可解码的常见格式
SUPPORTED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
# 目标采样率：16 kHz — HuBERT 预训练时使用的采样率，必须匹配
TARGET_SAMPLE_RATE = 16_000
# 流式读取的块大小：1 MB — 平衡内存占用与 I/O 效率
READ_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """
    解码后的音频数据 — 统一为单声道 16kHz 浮点波形

    frozen=True + slots=True 保证不可变性与内存效率：
    - frozen=True: 实例创建后不可修改字段，防止波形数据被意外篡改
    - slots=True: 使用 __slots__ 代替 __dict__，减少约 40% 内存占用

    字段说明：
    - waveform    : 单声道 16kHz 浮点波形，dtype=np.float32，值域 [-1.0, 1.0]
    - sample_rate : 采样率，固定为 TARGET_SAMPLE_RATE (16000)
    """
    waveform: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        """
        音频时长（秒） — 基于波形长度与采样率计算

        返回：
            时长（秒），精度取决于采样率（16kHz 时精度为 1/16000 秒 ≈ 62.5 微秒）
        """
        return len(self.waveform) / self.sample_rate


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """
    音频片段 — 滑动窗口切分后的单段数据

    字段说明：
    - index          : 片段序号（从 0 开始），用于结果排序与展示
    - start_seconds  : 片段起始时间（秒），用于前端展示时间轴
    - end_seconds    : 片段结束时间（秒），注意尾段可能短于窗口长度
    - waveform       : 片段波形数据（原始波形切片，非独立拷贝）
    - sample_count   : 片段采样点数，等于 len(waveform)
    - rms            : 根均方（Root Mean Square）能量值，用于静音检测与加权聚合
    - is_silent      : 是否为静音片段（rms < silence_rms_threshold）

    注意：waveform 是原始波形数组的切片视图，不是独立拷贝。
    这是为了避免大音频文件在分段时产生大量内存拷贝。
    若需要独立拷贝，调用方应自行 waveform.copy()。
    """
    index: int
    start_seconds: float
    end_seconds: float
    waveform: np.ndarray
    sample_count: int
    rms: float
    is_silent: bool


def read_limited_stream(stream: BinaryIO, max_bytes: int) -> bytes:
    """
    分块读取上传内容，避免客户端声明错误大小时一次占满内存

    此函数不依赖 UploadFile.content_length（可能不可靠或缺失），
    而是在读取过程中实时累加字节计数，一旦超过 max_bytes 立即抛出异常。

    算法步骤：
    1. 以 READ_CHUNK_SIZE (1 MB) 为单位逐块读取
    2. 实时累加已读取的字节总数
    3. 若总量超过 max_bytes，抛出 AppError("FILE_TOO_LARGE", 413)
    4. 若读取完毕后无数据（空文件），抛出 AppError("EMPTY_FILE", 400)
    5. 将所有块拼接为完整字节串返回

    参数：
        stream — 上传文件的二进制流（UploadFile.file 或类似对象）
        max_bytes — 文件大小上限（字节），通常为 50 MB

    返回：
        bytes — 完整的文件内容

    异常：
        AppError("FILE_TOO_LARGE", 413) — 文件超过大小上限
        AppError("EMPTY_FILE", 400) — 文件为空
    """
    chunks: list[bytes] = []
    total = 0
    # 逐块读取：每次最多读取 READ_CHUNK_SIZE (1 MB)
    while chunk := stream.read(READ_CHUNK_SIZE):
        total += len(chunk)
        # 实时检查大小上限，防止恶意大文件耗尽内存
        if total > max_bytes:
            raise AppError("FILE_TOO_LARGE", "音频文件不能超过 50 MB", 413)
        chunks.append(chunk)
    # 空文件检查：防止上传空文件导致下游解码异常
    if not chunks:
        raise AppError("EMPTY_FILE", "请选择有效的音频文件", 400)
    return b"".join(chunks)


def normalize_waveform(
    waveform: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """
    统一声道、采样率与数值范围，使输入符合 HuBERT 预训练约定

    处理步骤：
    1. 类型转换：强制转为 float32（模型推理的标准浮点类型）
    2. 有效性校验：拒绝空数组、零/负采样率、NaN/Inf 值
    3. 声道合并：多声道音频取均值合并为单声道
       - 二维数组时，较小的维度通常是声道（[声道, 采样] 或 [采样, 声道]）
       - 此判定基于经验观察：音频库（librosa, soundfile 等）通常返回 [声道, 采样] 格式
    4. 重采样：若源采样率与目标采样率不同，使用多相滤波器重采样
       - resample_poly 比 FFT 重采样更高效，比线性插值更精确
       - 使用 gcd 计算上/下采样因子，保证整数精度
    5. 幅值裁剪：将值域限制在 [-1.0, 1.0]，防止解码异常导致的极端值
    6. 连续数组：np.ascontiguousarray 保证内存布局连续，提升推理速度

    参数：
        waveform — FFmpeg 解码后的原始波形数据（可能为多声道）
        source_rate — 原始采样率（Hz）
        target_rate — 目标采样率（Hz），默认为 16000（HuBERT 预训练约定）

    返回：
        np.ndarray — 单声道目标采样率浮点波形，dtype=np.float32，值域 [-1.0, 1.0]

    异常：
        AppError("INVALID_AUDIO", 422) — 音频内容无效（空/零采样率/非有限值/无法识别声道）
    """
    # 强制转为 float32：模型推理的标准浮点类型，平衡精度与内存占用
    values = np.asarray(waveform, dtype=np.float32)
    # 有效性校验：拒绝空数组、零/负采样率
    if values.size == 0 or source_rate <= 0 or target_rate <= 0:
        raise AppError("INVALID_AUDIO", "音频内容无效", 422)
    # 有限性校验：拒绝 NaN 和 Inf 值（解码异常或数据损坏可能导致）
    if not np.isfinite(values).all():
        raise AppError("INVALID_AUDIO", "音频包含无效采样值", 422)

    # 声道合并逻辑
    if values.ndim == 2:
        # 音频库可能返回 [声道, 采样] 或 [采样, 声道]，较小的维度通常是声道。
        # 此判定基于经验观察：大多数音频库返回的声道数远小于采样点数
        channel_axis = 0 if values.shape[0] <= values.shape[1] else 1
        # 取均值合并为单声道：保证所有声道的信息都被保留，
        # 而非丢弃部分声道（通话场景中多声道通常为双声道立体声，均值即中点）
        values = values.mean(axis=channel_axis)
    elif values.ndim != 1:
        # 非一维/二维数组：无法识别声道结构，拒绝处理
        raise AppError("INVALID_AUDIO", "音频声道结构无法识别", 422)

    # 重采样逻辑：若源采样率与目标采样率不同，使用多相滤波器重采样
    if source_rate != target_rate:
        # 计算最大公约数（gcd），将重采样因子分解为整数上/下采样比率
        # 例：48000 → 16000 时，gcd=16000，up=1, down=3
        divisor = math.gcd(source_rate, target_rate)
        # resample_poly 使用多相滤波器，比 FFT 重采样更高效，
        # 比线性插值更精确（减少频谱混叠）
        values = resample_poly(values, target_rate // divisor, source_rate // divisor)
    # 幅值裁剪：将值域限制在 [-1.0, 1.0]，防止解码异常导致的极端值
    # 连续数组：保证内存布局连续（C-contiguous），提升模型推理速度
    return np.ascontiguousarray(np.clip(values, -1.0, 1.0), dtype=np.float32)


def decode_audio(data: bytes, filename: str | None, settings: Settings) -> DecodedAudio:
    """
    用参数列表调用 FFmpeg，并保证含原音频的随机临时文件必定清理

    此函数是音频解码的核心入口，将上传的二进制数据解码为统一的波形数组。

    算法流程：
    1. 根据文件扩展名判断是否为支持的格式
    2. 将数据写入临时文件（FFmpeg 需要文件路径作为输入）
    3. 构造 FFmpeg 命令行参数，将音频解码为 PCM 浮点单声道 16kHz
    4. 执行 FFmpeg subprocess，设置超时保护
    5. 将输出字节流解析为 numpy 浮点数组
    6. 调用 normalize_waveform 统一声道与采样率
    7. 校验音频时长不超过上限
    8. 在 finally 块中删除临时文件，保证清理必定执行

    FFmpeg 命令行参数说明：
    - "-v error"        : 仅输出错误信息，抑制冗余日志
    - "-nostdin"        : 不从标准输入读取，防止 FFmpeg 意外阻塞等待输入
    - "-i temp_path"    : 输入文件路径（临时文件）
    - "-t max+1"        : 限制解码时长为 max_duration_seconds + 1 秒，
                          多 1 秒的缓冲保证边界帧不被截断
    - "-f f32le"        : 输出格式为 32 位浮点小端序（PCM）
    - "-acodec pcm_f32le": 编解码器为 PCM 32 位浮点小端序
    - "-ac 1"           : 输出单声道（FFmpeg 在解码阶段合并声道，更高效）
    - "-ar 16000"       : 输出采样率 16kHz（FFmpeg 在解码阶段重采样，更高效）
    - "pipe:1"          : 输出到 stdout（避免二次临时文件）

    注意：声道合并与重采样在 FFmpeg 阶段完成而非 normalize_waveform 阶段，
    因为 FFmpeg 的内置重采样器对大多数格式的处理更高效且更精确。
    normalize_waveform 主要处理 FFmpeg 不覆盖的边缘情况。

    参数：
        data — 上传文件的全部二进制内容
        filename — 上传文件的原始文件名（用于扩展名判断），可为 None
        settings — 全局配置对象，提供 max_duration_seconds 等参数

    返回：
        DecodedAudio — 解码后的音频对象，包含单声道 16kHz 波形

    异常：
        AppError("UNSUPPORTED_FORMAT", 415) — 不支持的音频格式
        AppError("DECODE_FAILED", 422) — FFmpeg 解码失败或输出为空
        AppError("AUDIO_TOO_LONG", 413) — 音频时长超过上限
        AppError("DECODE_TIMEOUT", 408) — FFmpeg 执行超时
        AppError("DECODE_FAILED", 500) — FFmpeg 二进制不可用
    """
    # 根据文件扩展名判断是否为支持的格式
    # 若 filename 为 None，默认使用 .wav 扩展名（最常见的未命名格式）
    suffix = Path(filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise AppError("UNSUPPORTED_FORMAT", "支持 WAV、MP3、FLAC、OGG、M4A 和 WebM", 415)

    # 临时文件路径：用于存储上传数据供 FFmpeg 读取
    # delete=False：不自动删除，因为我们需要在写入后读取，
    # 手动在 finally 块中删除以保证清理
    temp_path: str | None = None
    try:
        # 创建临时文件并写入上传数据
        # NamedTemporaryFile 保证文件名唯一，避免并发冲突
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name

        # 构造 FFmpeg 命令行参数
        # 使用 imageio_ffmpeg.get_ffmpeg_exe() 获取 FFmpeg 二进制路径，
        # 避免要求用户自行安装 FFmpeg
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v", "error",             # 仅输出错误信息，抑制冗余日志
            "-nostdin",                # 不从标准输入读取，防止意外阻塞
            "-i", temp_path,           # 输入文件路径
            "-t", str(settings.max_duration_seconds + 1),  # 限制解码时长（多 1 秒缓冲）
            "-f", "f32le",             # 输出格式：32 位浮点小端序 PCM
            "-acodec", "pcm_f32le",    # 编解码器：PCM 32 位浮点小端序
            "-ac", "1",                # 输出声道数：单声道（FFmpeg 阶段合并）
            "-ar", str(TARGET_SAMPLE_RATE),  # 输出采样率：16kHz（FFmpeg 阶段重采样）
            "pipe:1",                  # 输出到 stdout，避免二次临时文件
        ]
        # 执行 FFmpeg subprocess
        # - check=False：不自动抛异常，我们手动检查 returncode
        # - capture_output=True：捕获 stdout 和 stderr
        # - timeout：超时保护，取 max(30秒, 1.5倍音频时长)
        #   30秒底限保证极短音频也有足够的解码时间
        #   1.5倍音频时长保证长音频不被过早终止
        # - shell=False：参数列表调用，防止 shell 注入风险
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=max(30.0, settings.max_duration_seconds * 1.5),
            shell=False,
        )
        # 检查 FFmpeg 执行结果
        # - returncode != 0：FFmpeg 解码失败（文件损坏、格式不匹配等）
        # - stdout 为空：解码成功但无输出（极短音频或空文件）
        if completed.returncode != 0 or not completed.stdout:
            raise AppError("DECODE_FAILED", "无法读取该音频，请检查文件是否损坏", 422)
        # 将 FFmpeg 输出的字节流解析为 numpy 浮点数组
        # dtype="<f4"：小端序 32 位浮点（与 FFmpeg 输出格式 f32le 匹配）
        # .copy()：将字节缓冲区转为独立数组，避免引用原始 stdout 缓冲区
        waveform = np.frombuffer(completed.stdout, dtype="<f4").copy()
        # 调用 normalize_waveform 统一声道与采样率
        # 注意：source_rate 传入 TARGET_SAMPLE_RATE (16000)，
        # 因为 FFmpeg 已在解码阶段完成重采样，normalize_waveform 无需再重采样
        waveform = normalize_waveform(waveform, TARGET_SAMPLE_RATE)
        # 构建 DecodedAudio 对象
        audio = DecodedAudio(waveform=waveform, sample_rate=TARGET_SAMPLE_RATE)
        # 校验音频时长不超过上限
        if audio.duration_seconds > settings.max_duration_seconds:
            raise AppError("AUDIO_TOO_LONG", "音频时长不能超过 5 分钟", 413)
        return audio
    except subprocess.TimeoutExpired as exc:
        # FFmpeg 执行超时：可能是音频过长或 FFmpeg 卡在损坏帧
        raise AppError("DECODE_TIMEOUT", "音频解码超时，请缩短后重试", 408) from exc
    except OSError as exc:
        # FFmpeg 二进制不可用：imageio_ffmpeg 未正确安装或系统不支持
        raise AppError("DECODE_FAILED", "音频解码组件不可用", 500) from exc
    finally:
        # 保证含原音频的随机临时文件必定清理
        # suppress(FileNotFoundError)：若临时文件已被其他进程删除，不抛异常
        if temp_path:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)


def segment_waveform(waveform: np.ndarray, settings: Settings) -> list[AudioSegment]:
    """
    按重叠窗口切分长音频；尾段不补零，留给特征处理器统一填充

    此函数将完整波形按滑动窗口切分为多个片段，供模型逐段推理。

    切分算法：
    1. 计算窗口大小（window_seconds × sample_rate）和步长（hop_seconds × sample_rate）
    2. 以步长为间隔遍历波形，每个位置取窗口长度的切片
    3. 尾段处理：若最后一段不足窗口长度，直接取剩余波形（不补零）
       补零由特征提取器的 padding="max_length" 参数完成，保证补零方式与模型一致
    4. 对每个片段计算 RMS 能量，用于静音检测与加权聚合
    5. 终止条件：当切片为空或窗口范围已覆盖波形末尾时停止

    重叠设计：
    - 窗间重叠 = window_seconds - hop_seconds（默认 1 秒）
    - 重叠区提供上下文过渡信息，避免窗口边界处的情绪漏检
    - 重叠区的信息通过加权聚合自动去重（权重归一化后总和为 1）

    参数：
        waveform — 完整波形数组，单声道 16kHz 浮点
        settings — 全局配置对象，提供 window_seconds 和 hop_seconds

    返回：
        list[AudioSegment] — 按时间顺序排列的音频片段列表

    注意：每个片段的 waveform 是原始波形的切片视图，不是独立拷贝。
    这是为了避免大音频文件在分段时产生大量内存拷贝。
    """
    # 目标采样率：16 kHz（与波形数据一致）
    sample_rate = TARGET_SAMPLE_RATE
    # 窗口大小（采样点数）：max(1, ...) 保证至少 1 个采样点
    window = max(1, round(settings.window_seconds * sample_rate))
    # 步长（采样点数）：max(1, ...) 保证至少 1 个采样点
    hop = max(1, round(settings.hop_seconds * sample_rate))
    # 片段列表
    segments: list[AudioSegment] = []
    # 以步长为间隔遍历波形
    for index, start in enumerate(range(0, len(waveform), hop)):
        # 取窗口长度的切片
        values = waveform[start : start + window]
        # 空切片检查：若波形已遍历完毕，停止
        if values.size == 0:
            break
        # 计算 RMS（根均方）能量值：
        # RMS = sqrt(mean(values^2))，反映片段的平均能量水平
        # dtype=np.float64：使用双精度计算，避免短片段的精度损失
        rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        # 构建片段对象
        segments.append(
            AudioSegment(
                index=index,
                # 时间转换：采样点位置 / 采样率 = 秒数
                start_seconds=start / sample_rate,
                # 注意：end_seconds 基于 len(values) 计算，
                # 尾段可能短于 window_seconds，时间范围反映实际长度
                end_seconds=(start + len(values)) / sample_rate,
                # waveform 是原始波形的切片视图，非独立拷贝
                waveform=values,
                # 采样点数：等于 len(values)
                sample_count=len(values),
                # RMS 能量值：用于静音检测与加权聚合
                rms=rms,
                # 静音判定：RMS < silence_rms_threshold (0.01 ≈ -40 dB) 视为静音
                # 静音片段跳过模型推理，节省计算资源
                is_silent=rms < settings.silence_rms_threshold,
            )
        )
        # 终止条件：当窗口范围已覆盖波形末尾时停止
        # 此条件保证尾段（短于窗口长度）被包含，但不再产生更多空片段
        if start + window >= len(waveform):
            break
    return segments
