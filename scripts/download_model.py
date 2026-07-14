"""
模型下载脚本

用途：在部署或开发前，预先下载并加载情感识别模型权重。
     确保模型文件缓存在本地，避免首次请求时的长等待时间。

使用方式：
    python scripts/download_model.py

脚本会读取项目配置中的模型标识（model_id），初始化运行时引擎并加载权重，
最后打印模型的状态和运行设备信息，以便确认下载是否成功。
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，以便直接导入 app 包下的模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings          # 获取项目配置（含模型ID、设备等参数）
from app.model import EmotionModelRuntime     # 情感模型运行时封装，负责加载与推理


def main() -> None:
    """主函数：加载模型并输出状态信息。"""
    # 从环境变量或默认配置文件中读取项目设置
    settings = get_settings()

    # 根据配置初始化模型运行时实例
    runtime = EmotionModelRuntime(settings)

    # 执行模型加载：下载权重文件并移至目标设备
    runtime.load()

    # 打印加载结果，包括模型标识、当前状态和运行设备
    print(f"model={settings.model_id} status={runtime.status} device={runtime.device}")


if __name__ == "__main__":
    main()
