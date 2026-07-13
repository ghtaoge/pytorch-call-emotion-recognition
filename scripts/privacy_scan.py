from __future__ import annotations

import re
import subprocess
from pathlib import Path

BLOCKED_SUFFIXES = {
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
PATTERNS = {
    "PRIVATE_KEY": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "BEARER_TOKEN": re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
    "WINDOWS_HOME": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+"),
    "POSIX_HOME": re.compile("/" + "home" + r"/[^/\s]+"),
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(item.decode("utf-8")) for item in output.split(b"\0") if item]


def main() -> int:
    findings: list[tuple[Path, str]] = []
    for path in tracked_files():
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            findings.append((path, "BINARY_PRIVATE_ARTIFACT"))
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for rule, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append((path, rule))
    for path, rule in findings:
        print(f"{path.as_posix()}: {rule}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
