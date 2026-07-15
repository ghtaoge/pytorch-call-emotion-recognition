"""
隐私安全扫描脚本

用途：扫描 Git 仓库中被跟踪的文件，检测可能泄露隐私或敏感信息的文件与内容。
     用于 CI/CD 流水线或本地开发时的安全自检，防止将敏感数据提交到版本库。

使用方式：
    python scripts/privacy_scan.py

扫描规则：
    1. 禁止二进制文件 —— 音频、模型权重等不应进入 Git 仓库
    2. 禁止私钥 —— 检测 PEM 格式私钥标识
    3. 禁止 Bearer Token —— 检测长字符串格式的认证令牌
    4. 禁止本地路径泄露 —— 检测 Windows 与 POSIX 的用户主目录路径

若发现任何问题，脚本返回退出码 1；无问题则返回 0。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# 禁止提交的二进制文件扩展名集合
# 包括音频文件、二进制数据文件、以及各类模型权重文件格式
BLOCKED_SUFFIXES = {
    ".wav",  # WAV 音频文件
    ".mp3",  # MP3 音频文件
    ".m4a",  # M4A 音频文件
    ".flac",  # FLAC 无损音频文件
    ".ogg",  # OGG 音频文件
    ".webm",  # WebM 音视频文件
    ".bin",  # 通用二进制文件
    ".pt",  # PyTorch 模型权重（旧格式）
    ".pth",  # PyTorch 模型权重（旧格式别名）
    ".safetensors",  # HuggingFace safetensors 格式权重
}

# 正则表达式规则集合：用于在文本文件中检测敏感信息
PATTERNS = {
    # 检测 PEM 格式私钥的起始行（如 RSA、EC、DSA 等各类私钥）
    "PRIVATE_KEY": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    # 检测 HTTP Bearer Token，匹配长度 >= 20 的令牌字符串
    "BEARER_TOKEN": re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
    # 检测 Windows 用户主目录路径泄露；规则文本使用拼接，避免扫描器匹配自身。
    "WINDOWS_HOME": re.compile(r"[A-Za-z]:\\" + "Users" + r"\\[^\\\s]+"),
    # 检测 POSIX 用户主目录路径泄露，同样避免写出连续的目标路径。
    "POSIX_HOME": re.compile("/" + "home" + r"/[^/\s]+"),
}


def tracked_files() -> list[Path]:
    """获取 Git 仓库中所有被跟踪的文件路径列表。

    使用 `git ls-files -z` 命令，以 NULL 字符分隔输出，
    确保含空格或特殊字符的路径也能正确解析。
    """
    output = subprocess.check_output(["git", "ls-files", "-z"])
    # 将原始字节按 NULL 字符拆分，解码为 UTF-8 字符串并转为 Path 对象
    return [Path(item.decode("utf-8")) for item in output.split(b"\0") if item]


def main() -> int:
    """执行隐私扫描主流程。

    对每个被 Git 跟踪的文件依次执行：
      1. 二进制文件后缀检测 —— 发现则标记为 BINARY_PRIVATE_ARTIFACT
      2. 文本内容正则扫描 —— 逐条匹配敏感信息规则
    最后汇总所有发现并输出报告。

    返回值：
        1 —— 存在至少一个隐私风险发现（应阻止提交）
        0 —— 未发现任何隐私风险（通过检查）
    """
    findings: list[tuple[Path, str]] = []  # 收集所有扫描发现：(文件路径, 规则名称)

    for path in tracked_files():
        # 第一步：检查文件后缀是否属于禁止提交的二进制类型
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            findings.append((path, "BINARY_PRIVATE_ARTIFACT"))
            continue  # 二进制文件无需再做内容扫描，跳过

        # 第二步：尝试以 UTF-8 编码读取文件内容
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # 无法读取的文件（非文本编码或权限问题）跳过
            continue

        # 第三步：对文本内容逐条应用正则规则，检测敏感信息
        for rule, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append((path, rule))

    # 输出所有发现，格式为 "文件路径: 规则名称"
    for path, rule in findings:
        print(f"{path.as_posix()}: {rule}")

    # 有发现则返回 1（表示存在风险），无发现则返回 0（表示安全）
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
