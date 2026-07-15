"""
API 端冒烟测试脚本

用途：对情感识别 API 的完整端到端流程进行快速验证。
     使用合成的正弦波音频模拟真实请求，验证从音频上传到
     流式响应接收的整条链路是否正常工作。

使用方式：
    python scripts/smoke_test_api.py

测试流程：
    1. 生成一段 2 秒的 220Hz 正弦波合成音频（16kHz 采样率）
    2. 将合成音频编码为 WAV 格式
    3. 通过 FastAPI TestClient 以 POST 方式发送至 /api/analyze 接口
    4. 收集流式响应中的所有事件（SSE 格式）
    5. 验证 HTTP 状态码为 200，且最终事件类型为 "result"
    6. 打印最终结果中的情感概率分布

若任一断言失败，脚本将抛出 AssertionError 表示链路异常。
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient  # FastAPI 内置测试客户端，无需启动真实服务器
from scipy.io.wavfile import write  # scipy 的 WAV 写入函数，用于生成合规的音频文件

# 将项目根目录加入 sys.path，以便直接导入 app 包下的模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app  # FastAPI 应用工厂函数，创建应用实例


def main() -> None:
    """冒烟测试主函数：合成音频 -> 调用 API -> 校验结果。"""

    # === 第一步：生成合成音频信号 ===
    # 创建 2 秒的时间轴，采样率 16kHz（共 32000 个采样点）
    seconds = np.arange(2 * 16_000, dtype=np.float32) / 16_000
    # 生成 220Hz 正弦波，振幅 0.1，模拟低音量的语音输入
    # 乘以 32767 并转为 int16，将浮点波形映射到 16-bit PCM 范围
    waveform = (0.1 * np.sin(2 * np.pi * 220 * seconds) * 32767).astype(np.int16)

    # === 第二步：编码为 WAV 格式 ===
    buffer = io.BytesIO()  # 创建内存缓冲区，避免写临时文件
    write(buffer, 16_000, waveform)  # 将波形数据写入缓冲区，指定采样率 16kHz

    # === 第三步：通过 TestClient 发送请求并接收流式响应 ===
    # 使用 FastAPI TestClient 模拟 HTTP 请求，无需启动真实服务器进程
    # 以流式（stream）方式 POST 音频文件至 /api/analyze 接口
    with (
        TestClient(create_app()) as client,
        client.stream(
            "POST",
            "/api/analyze",
            files={"audio": ("synthetic.wav", buffer.getvalue(), "audio/wav")},
        ) as response,
    ):
        # === 第四步：解析 SSE 流式事件 ===
        # 逐行读取流式响应，过滤空行，将每行 JSON 解析为事件对象
        events = [json.loads(line) for line in response.iter_lines() if line]

    # === 第五步：断言验证 ===
    # 验证 HTTP 响应状态码为 200（请求成功）
    assert response.status_code == 200
    # 验证最后一个事件类型为 "result"（表示模型已完成推理并返回结果）
    assert events[-1]["type"] == "result"

    # === 第六步：输出结果 ===
    # 打印最终结果中的情感概率分布（ensure_ascii=False 保留中文标签）
    print(json.dumps(events[-1]["result"]["probabilities"], ensure_ascii=False))


if __name__ == "__main__":
    main()
