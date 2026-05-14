from __future__ import annotations

from typing import Any

import pandas as pd
import requests

try:
    from config import TennisDatabaseConfig, get_tennis_database_config
except ModuleNotFoundError:
    from tennis.src.config import TennisDatabaseConfig, get_tennis_database_config


class TennisDatabaseError(RuntimeError):
    """Raised when a tennis Supabase operation fails."""


def _supabase_headers(
    config: TennisDatabaseConfig,
    *,
    write: bool = False,
    prefer: str | None = None,
) -> dict[str, str]:
    headers = {
        "apikey": config.supabase_service_role_key,
        "Authorization": f"Bearer {config.supabase_service_role_key}",
        "Accept-Profile": config.supabase_schema,
    }
    if write:
        headers["Content-Type"] = "application/json"
        headers["Content-Profile"] = config.supabase_schema
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _table_url(config: TennisDatabaseConfig, table_name: str) -> str:
    return f"{config.supabase_url}/rest/v1/{table_name}"


def fetch_tennis_table(
    table_name: str,
    *,
    select: str = "*",
    order: str | None = None,
    limit: int | None = None,
    filters: dict[str, str] | None = None,
    config: TennisDatabaseConfig | None = None,
) -> pd.DataFrame:
    """Fetch rows from a table in the tennis schema."""
    active_config = config or get_tennis_database_config()
    params: dict[str, str] = {"select": select}
    if order:
        params["order"] = order
    if limit is not None:
        params["limit"] = str(limit)
    if filters:
        params.update(filters)

    response = requests.get(
        _table_url(active_config, table_name),
        headers=_supabase_headers(active_config),
        params=params,
        timeout=30,
    )
    if response.status_code == 404:
        return pd.DataFrame()
    if response.status_code >= 400:
        raise TennisDatabaseError(
            f"Supabase fetch failed for tennis.{table_name} "
            f"({response.status_code}): {response.text}"
        )
    return pd.DataFrame(response.json())


def insert_tennis_rows(
    table_name: str,
    rows: dict[str, Any] | list[dict[str, Any]],
    *,
    on_conflict: str | None = None,
    config: TennisDatabaseConfig | None = None,
) -> list[dict[str, Any]]:
    """Insert rows into a table in the tennis schema and return inserted rows."""
    active_config = config or get_tennis_database_config()
    params = {"on_conflict": on_conflict} if on_conflict else None
    response = requests.post(
        _table_url(active_config, table_name),
        headers=_supabase_headers(active_config, write=True, prefer="return=representation"),
        params=params,
        json=rows,
        timeout=30,
    )
    if response.status_code >= 400:
        raise TennisDatabaseError(
            f"Supabase insert failed for tennis.{table_name} "
            f"({response.status_code}): {response.text}"
        )
    return response.json()


def upsert_tennis_rows(
    table_name: str,
    rows: dict[str, Any] | list[dict[str, Any]],
    *,
    on_conflict: str,
    config: TennisDatabaseConfig | None = None,
) -> list[dict[str, Any]]:
    """Upsert rows into a table in the tennis schema and return affected rows."""
    active_config = config or get_tennis_database_config()
    response = requests.post(
        _table_url(active_config, table_name),
        headers=_supabase_headers(
            active_config,
            write=True,
            prefer="resolution=merge-duplicates,return=representation",
        ),
        params={"on_conflict": on_conflict},
        json=rows,
        timeout=30,
    )
    if response.status_code >= 400:
        raise TennisDatabaseError(
            f"Supabase upsert failed for tennis.{table_name} "
            f"({response.status_code}): {response.text}"
        )
    return response.json()


def update_tennis_row(
    table_name: str,
    row_id: str,
    payload: dict[str, Any],
    *,
    config: TennisDatabaseConfig | None = None,
) -> None:
    """Update one row by id in a table in the tennis schema."""
    active_config = config or get_tennis_database_config()
    response = requests.patch(
        _table_url(active_config, table_name),
        headers=_supabase_headers(active_config, write=True, prefer="return=minimal"),
        params={"id": f"eq.{row_id}"},
        json=payload,
        timeout=30,
    )
    if response.status_code >= 400:
        raise TennisDatabaseError(
            f"Supabase update failed for tennis.{table_name}/{row_id} "
            f"({response.status_code}): {response.text}"
        )


def check_tennis_connection(config: TennisDatabaseConfig | None = None) -> dict[str, Any]:
    """Run a lightweight tennis-schema query to validate credentials and schema exposure."""
    frame = fetch_tennis_table(
        "matches_history",
        select="id",
        order="created_at.desc",
        limit=1,
        config=config,
    )
    return {
        "ok": True,
        "schema": (config or get_tennis_database_config()).supabase_schema,
        "table": "matches_history",
        "sample_rows": len(frame),
    }
