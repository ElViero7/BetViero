from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
ENV_PATHS = (
    BASE_DIR / ".env",
    PROJECT_DIR / ".env",
    PROJECT_DIR / "lol" / ".env",
)


class TennisConfigError(RuntimeError):
    """Raised when tennis configuration is missing or invalid."""


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_env_files(override: bool = False) -> None:
    """Load Supabase credentials from tennis, project-root, or LoL env files."""
    for env_path in ENV_PATHS:
        if not env_path.exists():
            continue

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


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        searched = ", ".join(str(path) for path in ENV_PATHS)
        raise TennisConfigError(
            f"Missing required environment variable: {name}. Searched: {searched}"
        )
    return value


@dataclass(frozen=True)
class TennisDatabaseConfig:
    supabase_url: str
    supabase_service_role_key: str
    supabase_schema: str = "tennis"

    @classmethod
    def from_env(cls) -> "TennisDatabaseConfig":
        load_env_files()
        config = cls(
            supabase_url=_get_required_env("SUPABASE_URL").rstrip("/"),
            supabase_service_role_key=_get_required_env("SUPABASE_SERVICE_ROLE_KEY"),
            supabase_schema=os.getenv("TENNIS_SUPABASE_SCHEMA", "tennis").strip() or "tennis",
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.supabase_url.startswith(("http://", "https://")):
            raise TennisConfigError("SUPABASE_URL must start with http:// or https://.")
        if not self.supabase_schema:
            raise TennisConfigError("TENNIS_SUPABASE_SCHEMA cannot be empty.")


@lru_cache(maxsize=1)
def get_tennis_database_config() -> TennisDatabaseConfig:
    return TennisDatabaseConfig.from_env()


@dataclass(frozen=True)
class TennisOddsApiConfig:
    api_key: str
    base_url: str
    sport_key: str
    regions: str
    bookmakers: str
    markets: str
    odds_format: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "TennisOddsApiConfig":
        load_env_files()
        api_key = os.getenv("THE_ODDS_API_KEY", "").strip() or os.getenv("ODDS_API_KEY", "").strip()
        if not api_key:
            searched = ", ".join(str(path) for path in ENV_PATHS)
            raise TennisConfigError(
                "Missing required environment variable: THE_ODDS_API_KEY. "
                f"Searched: {searched}"
            )

        config = cls(
            api_key=api_key,
            base_url=os.getenv("THE_ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4").rstrip("/"),
            sport_key=os.getenv("TENNIS_ODDS_SPORT_KEY", "tennis_atp").strip() or "tennis_atp",
            regions=os.getenv("TENNIS_ODDS_REGIONS", "eu,uk,us").strip() or "eu,uk,us",
            bookmakers=os.getenv("TENNIS_ODDS_BOOKMAKERS", "pinnacle").strip(),
            markets=os.getenv("TENNIS_ODDS_MARKETS", "totals,spreads").strip() or "totals,spreads",
            odds_format=os.getenv("TENNIS_ODDS_FORMAT", "decimal").strip() or "decimal",
            timeout_seconds=float(os.getenv("THE_ODDS_API_TIMEOUT_SECONDS", "30")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise TennisConfigError("THE_ODDS_API_BASE_URL must start with http:// or https://.")
        if self.odds_format not in {"decimal", "american"}:
            raise TennisConfigError("TENNIS_ODDS_FORMAT must be 'decimal' or 'american'.")
        if self.timeout_seconds <= 0:
            raise TennisConfigError("THE_ODDS_API_TIMEOUT_SECONDS must be greater than 0.")


@lru_cache(maxsize=1)
def get_tennis_odds_api_config() -> TennisOddsApiConfig:
    return TennisOddsApiConfig.from_env()
