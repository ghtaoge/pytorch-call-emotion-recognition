import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

ENVIRONMENT_KEYS = (
    "MODEL_ID",
    "MAX_BYTES",
    "MAX_DURATION_SECONDS",
    "WINDOW_SECONDS",
    "HOP_SECONDS",
    "SILENCE_RMS_THRESHOLD",
    "HOST",
    "PORT",
)


def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_settings_use_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_environment(monkeypatch)

    settings = Settings()

    assert settings.model_id == "xmj2002/hubert-base-ch-speech-emotion-recognition"
    assert settings.max_bytes == 50 * 1024 * 1024
    assert settings.max_duration_seconds == 300.0
    assert settings.window_seconds == 6.0
    assert settings.hop_seconds == 5.0
    assert settings.silence_rms_threshold == 0.01
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000


def test_settings_reject_hop_larger_than_window() -> None:
    with pytest.raises(ValidationError):
        Settings(window_seconds=5.0, hop_seconds=6.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_bytes", 0),
        ("max_duration_seconds", 0),
        ("window_seconds", 0),
        ("hop_seconds", 0),
        ("port", 0),
        ("port", 65_536),
        ("silence_rms_threshold", -0.01),
        ("silence_rms_threshold", 1.01),
        ("model_id", "  "),
        ("host", ""),
    ],
)
def test_settings_reject_invalid_limits(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})  # type: ignore[arg-type]


def test_settings_honor_uppercase_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SILENCE_RMS_THRESHOLD", "0.25")

    assert Settings().silence_rms_threshold == 0.25


def test_get_settings_caches_until_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("SILENCE_RMS_THRESHOLD", "0.2")

    first = get_settings()
    monkeypatch.setenv("SILENCE_RMS_THRESHOLD", "0.3")
    cached = get_settings()
    get_settings.cache_clear()
    refreshed = get_settings()
    get_settings.cache_clear()

    assert cached is first
    assert cached.silence_rms_threshold == 0.2
    assert refreshed is not first
    assert refreshed.silence_rms_threshold == 0.3


def test_settings_are_frozen_and_ignore_extra_input() -> None:
    settings = Settings(unused_value="ignored")  # type: ignore[call-arg]

    assert not hasattr(settings, "unused_value")
    with pytest.raises(ValidationError):
        settings.port = 9000
