import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analyzer import project_probabilities
from app.config import get_settings
from app.model import EmotionModelRuntime


def main() -> None:
    settings = get_settings()
    runtime = EmotionModelRuntime(settings)
    seconds = np.arange(2 * 16_000, dtype=np.float32) / 16_000
    waveform = 0.1 * np.sin(2 * np.pi * 220 * seconds)
    result = project_probabilities(runtime.predict(waveform))
    print(result.probabilities.model_dump_json())


if __name__ == "__main__":
    main()
