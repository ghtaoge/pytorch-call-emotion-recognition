from pathlib import Path


def test_repository_contains_no_private_audio_or_model_weights() -> None:
    blocked = {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".webm",
        ".bin",
        ".pt",
        ".pth",
        ".safetensors",
    }
    tracked = [
        Path(path)
        for path in __import__("subprocess")
        .check_output(["git", "ls-files"], text=True)
        .splitlines()
    ]
    assert not [path for path in tracked if path.suffix.lower() in blocked]


def test_required_open_source_documents_exist() -> None:
    for name in (
        "README.md",
        "README_zh-CN.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        assert Path(name).is_file()
