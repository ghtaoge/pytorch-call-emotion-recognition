from functools import lru_cache
from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveFloat = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NormalizedFloat = Annotated[float, Field(ge=0, le=1)]
Port = Annotated[int, Field(ge=1, le=65_535)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    model_id: str = "xmj2002/hubert-base-ch-speech-emotion-recognition"
    max_bytes: PositiveInt = 50 * 1024 * 1024
    max_duration_seconds: PositiveFloat = 300.0
    window_seconds: PositiveFloat = 6.0
    hop_seconds: PositiveFloat = 5.0
    silence_rms_threshold: NormalizedFloat = 0.01
    host: str = "127.0.0.1"
    port: Port = 8000

    @field_validator("model_id", "host")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_hop_size(self) -> Self:
        if self.hop_seconds > self.window_seconds:
            raise ValueError("hop_seconds must not exceed window_seconds")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
