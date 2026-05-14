from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


class SettingsError(RuntimeError):
    """Raised when the application configuration is invalid."""


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()

    if not value:
        return value

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]

    return value


def load_env_file(env_path: Path = ENV_PATH, override: bool = False) -> None:
    """Load key-value pairs from a local .env file into process env vars."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        env_key = key.strip()

        if not env_key:
            continue

        if not override and env_key in os.environ:
            continue

        os.environ[env_key] = _parse_env_value(raw_value)


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_publishable_key: str
    supabase_service_role_key: str
    api_football_key: str
    api_football_base_url: str
    api_football_host: str
    api_football_timeout_seconds: float
    the_odds_api_key: str
    the_odds_api_base_url: str
    the_odds_api_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file()

        settings = cls(
            supabase_url=_get_required_env("SUPABASE_URL"),
            supabase_publishable_key=_get_required_env("SUPABASE_PUBLISHABLE_KEY"),
            supabase_service_role_key=_get_required_env("SUPABASE_SERVICE_ROLE_KEY"),
            api_football_key=_get_required_env("API_FOOTBALL_KEY"),
            api_football_base_url=_get_env(
                "API_FOOTBALL_BASE_URL",
                "https://api-football-v1.p.rapidapi.com/v3",
            ),
            api_football_host=_get_env(
                "API_FOOTBALL_HOST",
                "api-football-v1.p.rapidapi.com",
            ),
            api_football_timeout_seconds=_get_float_env(
                "API_FOOTBALL_TIMEOUT_SECONDS",
                30.0,
            ),
            the_odds_api_key=_get_required_env("THE_ODDS_API_KEY"),
            the_odds_api_base_url=_get_env(
                "THE_ODDS_API_BASE_URL",
                "https://api.the-odds-api.com/v4",
            ),
            the_odds_api_timeout_seconds=_get_float_env(
                "THE_ODDS_API_TIMEOUT_SECONDS",
                30.0,
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.supabase_url.startswith(("http://", "https://")):
            raise SettingsError(
                "SUPABASE_URL must start with http:// or https://."
            )

        if ".supabase.co" not in self.supabase_url and ".supabase.in" not in self.supabase_url:
            raise SettingsError(
                "SUPABASE_URL does not look like a valid Supabase project URL."
            )

        if not self.api_football_base_url.startswith(("http://", "https://")):
            raise SettingsError(
                "API_FOOTBALL_BASE_URL must start with http:// or https://."
            )

        if self.api_football_timeout_seconds <= 0:
            raise SettingsError(
                "API_FOOTBALL_TIMEOUT_SECONDS must be greater than 0."
            )

        if not self.the_odds_api_base_url.startswith(("http://", "https://")):
            raise SettingsError(
                "THE_ODDS_API_BASE_URL must start with http:// or https://."
            )

        if self.the_odds_api_timeout_seconds <= 0:
            raise SettingsError(
                "THE_ODDS_API_TIMEOUT_SECONDS must be greater than 0."
            )


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise SettingsError(
            f"Missing required environment variable: {name}. "
            f"Set it in {ENV_PATH} or in the process environment."
        )

    return value


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()

    if not raw_value:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise SettingsError(
            f"Environment variable {name} must be a valid float."
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
