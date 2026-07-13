import io
import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from scipy.io.wavfile import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app


def main() -> None:
    seconds = np.arange(2 * 16_000, dtype=np.float32) / 16_000
    waveform = (0.1 * np.sin(2 * np.pi * 220 * seconds) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    write(buffer, 16_000, waveform)
    with (
        TestClient(create_app()) as client,
        client.stream(
            "POST",
            "/api/analyze",
            files={"audio": ("synthetic.wav", buffer.getvalue(), "audio/wav")},
        ) as response,
    ):
        events = [json.loads(line) for line in response.iter_lines() if line]
    assert response.status_code == 200
    assert events[-1]["type"] == "result"
    print(json.dumps(events[-1]["result"]["probabilities"], ensure_ascii=False))


if __name__ == "__main__":
    main()
