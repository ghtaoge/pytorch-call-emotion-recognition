"""
test_privacy —— 仓库隐私合规与文档完整性验证

本模块检查仓库是否满足开源合规要求：
- 不包含私密音频文件或模型权重（防止泄露用户数据或大体积文件）
- 包含所有必需的开源文档（README、LICENSE、行为准则等）

这些测试确保仓库在公开发布时不会意外包含敏感数据。
"""

from pathlib import Path


def test_repository_contains_no_private_audio_or_model_weights() -> None:
    """验证仓库中不包含私密音频文件或模型权重文件。

    检查 git 跟踪的所有文件，排除以下扩展名：
    - 音频文件：.wav、.mp3、.m4a、.flac、.ogg、.webm（可能包含用户语音数据）
    - 模型权重：.bin、.pt、.pth、.safetensors（体积过大，不适合放入仓库）

    如果发现任何此类文件，测试失败，提醒开发者移除或使用 Git LFS。
    """
    # 定义应被禁止的文件扩展名集合
    blocked = {
        ".wav",  # WAV 音频文件
        ".mp3",  # MP3 音频文件
        ".m4a",  # M4A 音频文件
        ".flac",  # FLAC 音频文件
        ".ogg",  # OGG 音频文件
        ".webm",  # WebM 音频/视频文件
        ".bin",  # 二进制模型权重
        ".pt",  # PyTorch 模型权重
        ".pth",  # PyTorch 模型权重
        ".safetensors",  # safetensors 模型权重
    }
    # 获取 git 跟踪的所有文件列表
    tracked = [
        Path(path)
        for path in __import__("subprocess")
        .check_output(["git", "ls-files"], text=True)
        .splitlines()
    ]
    # 确保没有跟踪文件属于禁止的扩展名类型
    assert not [path for path in tracked if path.suffix.lower() in blocked]


def test_required_open_source_documents_exist() -> None:
    """验证仓库包含所有必需的开源文档。

    检查以下文档是否存在于仓库根目录：
    - README.md：项目说明（英文）
    - README_zh-CN.md：项目说明（中文）
    - LICENSE：开源许可证
    - CONTRIBUTING.md：贡献指南
    - CODE_OF_CONDUCT.md：行为准则
    - SECURITY.md：安全政策
    - THIRD_PARTY_NOTICES.md：第三方依赖声明

    缺少任何文档都会导致测试失败。
    """
    for name in (
        "README.md",  # 英文项目说明
        "README_zh-CN.md",  # 中文项目说明
        "LICENSE",  # 开源许可证
        "CONTRIBUTING.md",  # 贡献指南
        "CODE_OF_CONDUCT.md",  # 行为准则
        "SECURITY.md",  # 安全政策
        "THIRD_PARTY_NOTICES.md",  # 第三方依赖声明
    ):
        assert Path(name).is_file()  # 每个文档都必须存在
