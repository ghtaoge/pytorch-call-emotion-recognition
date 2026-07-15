"""
模型端冒烟测试脚本

用途：直接验证情感识别模型的推理流程是否正常工作，不经过 HTTP API 层。
     使用合成的正弦波音频作为输入，测试从波形数据到情感概率输出的完整链路。

使用方式：
    python scripts/smoke_test_model.py

测试流程：
    1. 加载项目配置并初始化模型运行时
    2. 生成一段 2 秒的 220Hz 正弦波音频（16kHz 采样率，float32 格式）
    3. 将波形送入模型执行推理，获取原始输出 logits
    4. 通过 project_probabilities 将 logits 转换为情感概率分布
    5. 打印最终的概率分布结果

与 smoke_test_api.py 的区别：
    本脚本跳过 HTTP 层与 WAV 编解码，直接在模型层进行验证，
    适合排查模型加载或推理本身的问题。
"""

import sys
from pathlib import Path

import numpy as np

# 将项目根目录加入 sys.path，以便直接导入 app 包下的模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analyzer import project_probabilities  # 将模型原始 logits 投射为情感概率分布
from app.config import get_settings  # 获取项目配置（含模型ID、设备等参数）
from app.model import EmotionModelRuntime  # 情感模型运行时封装，负责加载与推理


def main() -> None:
    """冒烟测试主函数：加载模型 -> 合成音频 -> 推理 -> 输出概率。"""

    # === 第一步：初始化配置与模型运行时 ===
    # 从环境变量或默认配置文件中读取项目设置
    settings = get_settings()
    # 根据配置创建模型运行时实例
    runtime = EmotionModelRuntime(settings)

    # === 第二步：生成合成音频信号 ===
    # 创建 2 秒的时间轴，采样率 16kHz（共 32000 个采样点）
    seconds = np.arange(2 * 16_000, dtype=np.float32) / 16_000
    # 生成 220Hz 正弦波，振幅 0.1，作为模拟语音输入
    # 注意：此处保持 float32 格式，无需转为 int16（模型直接接受浮点波形）
    waveform = 0.1 * np.sin(2 * np.pi * 220 * seconds)

    # === 第三步：执行模型推理 ===
    # 将波形数据送入模型，获取原始 logits 输出
    result = project_probabilities(runtime.predict(waveform))

    # === 第四步：输出结果 ===
    # 将概率分布以 JSON 格式打印，便于人工检查或脚本解析
    print(result.probabilities.model_dump_json())


if __name__ == "__main__":
    main()
