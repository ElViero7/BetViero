from __future__ import annotations

from functools import lru_cache
from typing import Any

from supabase import Client, create_client

from src.config import Settings, get_settings


class DatabaseConnectionError(RuntimeError):
    """Raised when the Supabase client cannot be initialized or used."""


def _build_supabase_client(settings: Settings) -> Client:
    try:
        return create_client(
            supabase_url=settings.supabase_url,
            supabase_key=settings.supabase_service_role_key,
        )
    except Exception as exc:
        raise DatabaseConnectionError(
            "Failed to initialize Supabase client."
        ) from exc


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    settings = get_settings()
    return _build_supabase_client(settings)


def check_supabase_connection() -> dict[str, Any]:
    """
    Execute a lightweight query to validate credentials and database reachability.
    """
    client = get_supabase_client()

    try:
        response = (
            client.table("matches")
            .select("id", count="exact")
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise DatabaseConnectionError(
            "Supabase connectivity check failed while querying matches."
        ) from exc

    return {
        "ok": True,
        "table": "matches",
        "row_count": getattr(response, "count", None),
    }
