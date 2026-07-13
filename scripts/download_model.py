import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.model import EmotionModelRuntime


def main() -> None:
    settings = get_settings()
    runtime = EmotionModelRuntime(settings)
    runtime.load()
    print(f"model={settings.model_id} status={runtime.status} device={runtime.device}")


if __name__ == "__main__":
    main()
