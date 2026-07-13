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

SUPPORTED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
TARGET_SAMPLE_RATE = 16_000
READ_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    waveform: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.waveform) / self.sample_rate


@dataclass(frozen=True, slots=True)
class AudioSegment:
    index: int
    start_seconds: float
    end_seconds: float
    waveform: np.ndarray
    sample_count: int
    rms: float
    is_silent: bool


def read_limited_stream(stream: BinaryIO, max_bytes: int) -> bytes:
    """分块读取上传内容，避免客户端声明错误大小时一次占满内存。"""
    chunks: list[bytes] = []
    total = 0
    while chunk := stream.read(READ_CHUNK_SIZE):
        total += len(chunk)
        if total > max_bytes:
            raise AppError("FILE_TOO_LARGE", "音频文件不能超过 50 MB", 413)
        chunks.append(chunk)
    if not chunks:
        raise AppError("EMPTY_FILE", "请选择有效的音频文件", 400)
    return b"".join(chunks)


def normalize_waveform(
    waveform: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """统一声道、采样率与数值范围，使输入符合 HuBERT 预训练约定。"""
    values = np.asarray(waveform, dtype=np.float32)
    if values.size == 0 or source_rate <= 0 or target_rate <= 0:
        raise AppError("INVALID_AUDIO", "音频内容无效", 422)
    if not np.isfinite(values).all():
        raise AppError("INVALID_AUDIO", "音频包含无效采样值", 422)

    if values.ndim == 2:
        # 音频库可能返回 [声道, 采样] 或 [采样, 声道]，较小的维度通常是声道。
        channel_axis = 0 if values.shape[0] <= values.shape[1] else 1
        values = values.mean(axis=channel_axis)
    elif values.ndim != 1:
        raise AppError("INVALID_AUDIO", "音频声道结构无法识别", 422)

    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        values = resample_poly(values, target_rate // divisor, source_rate // divisor)
    return np.ascontiguousarray(np.clip(values, -1.0, 1.0), dtype=np.float32)


def decode_audio(data: bytes, filename: str | None, settings: Settings) -> DecodedAudio:
    """用参数列表调用 FFmpeg，并保证含原音频的随机临时文件必定清理。"""
    suffix = Path(filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise AppError("UNSUPPORTED_FORMAT", "支持 WAV、MP3、FLAC、OGG、M4A 和 WebM", 415)

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-nostdin",
            "-i",
            temp_path,
            "-t",
            str(settings.max_duration_seconds + 1),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "pipe:1",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=max(30.0, settings.max_duration_seconds * 1.5),
            shell=False,
        )
        if completed.returncode != 0 or not completed.stdout:
            raise AppError("DECODE_FAILED", "无法读取该音频，请检查文件是否损坏", 422)
        waveform = np.frombuffer(completed.stdout, dtype="<f4").copy()
        waveform = normalize_waveform(waveform, TARGET_SAMPLE_RATE)
        audio = DecodedAudio(waveform=waveform, sample_rate=TARGET_SAMPLE_RATE)
        if audio.duration_seconds > settings.max_duration_seconds:
            raise AppError("AUDIO_TOO_LONG", "音频时长不能超过 5 分钟", 413)
        return audio
    except subprocess.TimeoutExpired as exc:
        raise AppError("DECODE_TIMEOUT", "音频解码超时，请缩短后重试", 408) from exc
    except OSError as exc:
        raise AppError("DECODE_FAILED", "音频解码组件不可用", 500) from exc
    finally:
        if temp_path:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)


def segment_waveform(waveform: np.ndarray, settings: Settings) -> list[AudioSegment]:
    """按重叠窗口切分长音频；尾段不补零，留给特征处理器统一填充。"""
    sample_rate = TARGET_SAMPLE_RATE
    window = max(1, round(settings.window_seconds * sample_rate))
    hop = max(1, round(settings.hop_seconds * sample_rate))
    segments: list[AudioSegment] = []
    for index, start in enumerate(range(0, len(waveform), hop)):
        values = waveform[start : start + window]
        if values.size == 0:
            break
        rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        segments.append(
            AudioSegment(
                index=index,
                start_seconds=start / sample_rate,
                end_seconds=(start + len(values)) / sample_rate,
                waveform=values,
                sample_count=len(values),
                rms=rms,
                is_silent=rms < settings.silence_rms_threshold,
            )
        )
        if start + window >= len(waveform):
            break
    return segments
