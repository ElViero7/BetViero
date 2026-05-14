from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import gdown
import pandas as pd
import requests
from dotenv import load_dotenv
from gdown.parse_url import parse_url

from .base import LoLPipelineError


@dataclass(frozen=True)
class DashboardConfig:
    supabase_url: str
    supabase_service_role_key: str
    supabase_schema: str
    predictions_output_path: Path
    raw_data_dir: Path
    oracle_2026_drive_url: str


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise LoLPipelineError(f"Missing required environment variable: {name}")
    return value


def load_dashboard_config() -> DashboardConfig:
    load_dotenv()
    return DashboardConfig(
        supabase_url=require_env("SUPABASE_URL").rstrip("/"),
        supabase_service_role_key=require_env("SUPABASE_SERVICE_ROLE_KEY"),
        supabase_schema=os.getenv("SUPABASE_SCHEMA", "public").strip() or "public",
        predictions_output_path=Path(
            os.getenv(
                "PREDICTIONS_INPUT_PATH",
                "data/predictions/upcoming_multimarket_predictions.csv",
            )
        ),
        raw_data_dir=Path(os.getenv("ORACLES_ELIXIR_RAW_DIR", "data/raw/oracles_elixir")),
        oracle_2026_drive_url=os.getenv("ORACLES_ELIXIR_2026_DRIVE_URL", "").strip(),
    )


def _supabase_headers(config: DashboardConfig) -> dict[str, str]:
    return {
        "apikey": config.supabase_service_role_key,
        "Authorization": f"Bearer {config.supabase_service_role_key}",
        "Accept-Profile": config.supabase_schema,
    }


def fetch_table(config: DashboardConfig, table_name: str) -> pd.DataFrame:
    url = f"{config.supabase_url}/rest/v1/{table_name}"
    params = {"select": "*", "order": "created_at.desc", "limit": "5000"}
    response = requests.get(url, headers=_supabase_headers(config), params=params, timeout=30)
    if response.status_code == 404:
        return pd.DataFrame()
    if response.status_code >= 400:
        raise LoLPipelineError(
            f"Supabase fetch failed for {table_name} ({response.status_code}): {response.text}"
        )
    payload = response.json()
    return pd.DataFrame(payload)


def update_table_row(
    config: DashboardConfig,
    table_name: str,
    row_id: str,
    payload: dict[str, Any],
) -> None:
    url = f"{config.supabase_url}/rest/v1/{table_name}"
    headers = {
        "apikey": config.supabase_service_role_key,
        "Authorization": f"Bearer {config.supabase_service_role_key}",
        "Content-Type": "application/json",
        "Accept-Profile": config.supabase_schema,
        "Content-Profile": config.supabase_schema,
        "Prefer": "return=minimal",
    }
    response = requests.patch(
        url,
        headers=headers,
        params={"id": f"eq.{row_id}"},
        json=payload,
        timeout=30,
    )
    if response.status_code >= 400:
        raise LoLPipelineError(
            f"Supabase update failed for {table_name}/{row_id} ({response.status_code}): {response.text}"
        )


def download_oracle_drive_folder(
    drive_url: str,
    output_dir: Path,
    skip_existing: bool = True,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download_folder(
        url=drive_url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
        skip_download=skip_existing,
    )
    if not downloaded:
        raise LoLPipelineError("No files were downloaded from the Oracle Drive folder.")
    return downloaded


def download_oracle_2026_csv(
    drive_url: str,
    output_path: Path,
) -> Path:
    drive_reference = drive_url.strip()
    if not drive_reference:
        raise LoLPipelineError(
            "Falta `ORACLES_ELIXIR_2026_DRIVE_URL` en el entorno. Si Google Drive da rate-limit, descarga el CSV 2026 a mano y sustituyelo en `data/raw/oracles_elixir`."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_id = None
    if drive_reference.startswith(("http://", "https://")):
        file_id, _ = parse_url(drive_reference, warning=False)
    elif re.fullmatch(r"[-\w]{20,}", drive_reference):
        file_id = drive_reference

    download_kwargs: dict[str, Any] = {
        "output": str(output_path),
        "quiet": False,
        "fuzzy": True,
        "use_cookies": False,
    }
    if file_id:
        download_kwargs["id"] = file_id
    else:
        download_kwargs["url"] = drive_reference

    downloaded = gdown.download(
        **download_kwargs,
    )
    if (not downloaded or not output_path.exists()) and file_id:
        downloaded = gdown.download(
            url=drive_reference,
            output=str(output_path),
            quiet=False,
            fuzzy=True,
            use_cookies=False,
        )
    if not downloaded or not output_path.exists():
        raise LoLPipelineError(
            "No pude descargar el CSV 2026 desde Drive. Revisa que `ORACLES_ELIXIR_2026_DRIVE_URL` sea un enlace de archivo de Google Drive o un file ID, no una carpeta."
        )
    if output_path.stat().st_size == 0:
        raise LoLPipelineError("La descarga de Oracle 2026 produjo un archivo vacio.")
    return output_path


def load_predictions_snapshot(config: DashboardConfig) -> pd.DataFrame:
    if not config.predictions_output_path.exists():
        return pd.DataFrame()
    return pd.read_csv(config.predictions_output_path)


def normalize_name_key(text: str) -> str:
    normalized = "".join(char for char in str(text).lower() if char.isalnum())
    normalized = normalized.replace("redforce", "redforce")
    normalized = normalized.replace("force", "")
    normalized = normalized.replace("kia", "kia")
    return normalized


def build_oracle_results_frame(raw_data_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(raw_data_dir.glob("2026*.csv"))
    if not csv_paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        frames.append(
            pd.read_csv(
                path,
                low_memory=False,
                usecols=[
                    "gameid",
                    "date",
                    "league",
                    "teamname",
                    "position",
                    "datacompleteness",
                    "side",
                    "result",
                    "firstblood",
                    "firsttower",
                    "kills",
                    "game",
                ],
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame.columns = [column.lower().strip() for column in frame.columns]
    frame["position"] = frame["position"].astype(str).str.strip().str.lower()
    frame["datacompleteness"] = frame["datacompleteness"].astype(str).str.strip().str.lower()
    frame["league"] = frame["league"].astype(str).str.strip()
    frame["teamname"] = frame["teamname"].astype(str).str.strip()
    frame["side"] = frame["side"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["game"] = pd.to_numeric(frame["game"], errors="coerce")
    frame["result"] = pd.to_numeric(frame["result"], errors="coerce").fillna(0.0)
    frame["firstblood"] = pd.to_numeric(frame["firstblood"], errors="coerce").fillna(0.0)
    frame["firsttower"] = pd.to_numeric(frame["firsttower"], errors="coerce").fillna(0.0)
    frame["kills"] = pd.to_numeric(frame["kills"], errors="coerce")

    eligible = frame.loc[
        (frame["position"] == "team")
        & (
            (frame["datacompleteness"] == "complete")
            | ((frame["league"] == "LPL") & (frame["datacompleteness"] == "partial"))
        )
    ].copy()

    rows: list[dict[str, Any]] = []
    for _, group in eligible.groupby("gameid"):
        if len(group) != 2:
            continue
        group = group.sort_values("side").reset_index(drop=True)
        team_names = group["teamname"].tolist()
        winner_rows = group.loc[group["result"] == 1, "teamname"].tolist()
        firstblood_rows = group.loc[group["firstblood"] == 1, "teamname"].tolist()
        firsttower_rows = group.loc[group["firsttower"] == 1, "teamname"].tolist()
        rows.append(
            {
                "game_id": str(group["gameid"].iloc[0]),
                "match_date": group["date"].iloc[0],
                "league": group["league"].iloc[0],
                "game": group["game"].iloc[0],
                "team_a": team_names[0],
                "team_b": team_names[1],
                "team_key_a": normalize_name_key(team_names[0]),
                "team_key_b": normalize_name_key(team_names[1]),
                "pair_key": "||".join(sorted([normalize_name_key(team_names[0]), normalize_name_key(team_names[1])])),
                "winner": winner_rows[0] if winner_rows else None,
                "firstblood_winner": firstblood_rows[0] if firstblood_rows else None,
                "firsttower_winner": firsttower_rows[0] if firsttower_rows else None,
                "total_kills": float(group["kills"].fillna(0.0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)


def settle_legacy_bets(frame: pd.DataFrame, results_frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    settled = frame.copy()
    settled["match_name"] = settled["match_name"].astype(str)
    settled["created_at"] = pd.to_datetime(settled["created_at"], utc=True, errors="coerce")
    settled["picked_team"] = settled.get("picked_team").fillna(settled.get("model_selected_team"))
    settled["my_odd"] = pd.to_numeric(settled["my_odd"], errors="coerce")
    settled["model_fair_odd"] = pd.to_numeric(settled["model_fair_odd"], errors="coerce")
    settled = settled.loc[settled["my_odd"] > settled["model_fair_odd"]].copy()
    if results_frame.empty:
        settled["settled_status"] = settled.get("status", "pending")
        settled["settled_league"] = pd.NA
        settled["settled_match_date"] = pd.NaT
        return settled

    status_values: list[str] = []
    settled_leagues: list[str | None] = []
    settled_dates: list[pd.Timestamp | None] = []
    for _, row in settled.iterrows():
        team_parts = [part.strip() for part in str(row["match_name"]).split(" vs ")]
        if len(team_parts) != 2:
            status_values.append(str(row.get("status", "pending")).lower())
            settled_leagues.append(None)
            settled_dates.append(None)
            continue
        pair_key = "||".join(sorted([normalize_name_key(team_parts[0]), normalize_name_key(team_parts[1])]))
        created_at = row["created_at"]
        candidates = results_frame.loc[results_frame["pair_key"] == pair_key].copy()
        if pd.notna(created_at):
            candidates = candidates.loc[candidates["match_date"] >= created_at]
        candidates = candidates.sort_values("match_date")
        if candidates.empty:
            status_values.append(str(row.get("status", "pending")).lower())
            settled_leagues.append(None)
            settled_dates.append(None)
            continue
        match_row = candidates.iloc[0]
        winner = match_row["firstblood_winner"]
        picked_team = str(row["picked_team"]).strip()
        if not winner or pd.isna(winner):
            status_values.append("void")
        elif normalize_name_key(winner) == normalize_name_key(picked_team):
            status_values.append("won")
        else:
            status_values.append("lost")
        settled_leagues.append(str(match_row["league"]))
        settled_dates.append(match_row["match_date"])

    settled["settled_status"] = status_values
    settled["settled_league"] = settled_leagues
    settled["settled_match_date"] = pd.to_datetime(settled_dates, utc=True, errors="coerce")
    return settled


def _find_settlement_row(
    match_name: str,
    created_at: Any,
    results_frame: pd.DataFrame,
) -> pd.Series | None:
    team_parts = [part.strip() for part in str(match_name).split(" vs ")]
    if len(team_parts) != 2:
        return None
    pair_key = "||".join(sorted([normalize_name_key(team_parts[0]), normalize_name_key(team_parts[1])]))
    candidates = results_frame.loc[results_frame["pair_key"] == pair_key].copy()
    created_at_ts = pd.to_datetime(created_at, utc=True, errors="coerce")
    if pd.notna(created_at_ts):
        candidates = candidates.loc[candidates["match_date"] >= created_at_ts]
    candidates = candidates.sort_values("match_date")
    if candidates.empty:
        return None
    return candidates.iloc[0]


def settle_picks(frame: pd.DataFrame, results_frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    settled = frame.copy()
    if results_frame.empty:
        settled["settled_status"] = settled.get("status", "pending")
        settled["settled_league"] = settled.get("league")
        settled["settled_match_date"] = pd.NaT
        settled["settled_actual_value"] = pd.NA
        return settled

    settled["created_at"] = pd.to_datetime(settled["created_at"], utc=True, errors="coerce")
    settled["line"] = pd.to_numeric(settled.get("line"), errors="coerce")
    settled["predicted_value"] = pd.to_numeric(settled.get("predicted_value"), errors="coerce")

    status_values: list[str] = []
    settled_leagues: list[str | None] = []
    settled_dates: list[pd.Timestamp | None] = []
    settled_actual_values: list[float | str | None] = []

    for _, row in settled.iterrows():
        match_row = _find_settlement_row(str(row.get("match_name", "")), row.get("created_at"), results_frame)
        if match_row is None:
            status_values.append(str(row.get("status", "pending")).lower())
            settled_leagues.append(row.get("league"))
            settled_dates.append(None)
            settled_actual_values.append(None)
            continue

        market = str(row.get("market", "")).strip().lower()
        selection = str(row.get("selection", "")).strip()
        status = str(row.get("status", "pending")).lower()
        actual_value: float | str | None = None

        if market == "winner":
            actual_value = match_row.get("winner")
            if not actual_value or pd.isna(actual_value):
                status = "void"
            elif normalize_name_key(str(actual_value)) == normalize_name_key(selection):
                status = "won"
            else:
                status = "lost"
        elif market == "firstblood":
            actual_value = match_row.get("firstblood_winner")
            if not actual_value or pd.isna(actual_value):
                status = "void"
            elif normalize_name_key(str(actual_value)) == normalize_name_key(selection):
                status = "won"
            else:
                status = "lost"
        elif market == "firsttower":
            actual_value = match_row.get("firsttower_winner")
            if not actual_value or pd.isna(actual_value):
                status = "void"
            elif normalize_name_key(str(actual_value)) == normalize_name_key(selection):
                status = "won"
            else:
                status = "lost"
        elif market == "kills":
            actual_total = pd.to_numeric(match_row.get("total_kills"), errors="coerce")
            line_value = pd.to_numeric(row.get("line"), errors="coerce")
            actual_value = None if pd.isna(actual_total) else float(actual_total)
            if pd.isna(actual_total) or pd.isna(line_value):
                status = "pending"
            elif selection.lower() == "over":
                status = "won" if float(actual_total) > float(line_value) else "lost" if float(actual_total) < float(line_value) else "void"
            elif selection.lower() == "under":
                status = "won" if float(actual_total) < float(line_value) else "lost" if float(actual_total) > float(line_value) else "void"
            else:
                status = "pending"

        status_values.append(status)
        settled_leagues.append(str(match_row.get("league")) if pd.notna(match_row.get("league")) else row.get("league"))
        settled_dates.append(match_row.get("match_date"))
        settled_actual_values.append(actual_value)

    settled["settled_status"] = status_values
    settled["settled_league"] = settled_leagues
    settled["settled_match_date"] = pd.to_datetime(settled_dates, utc=True, errors="coerce")
    settled["settled_actual_value"] = settled_actual_values
    return settled


def normalize_legacy_bets(frame: pd.DataFrame, results_frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "source_table",
                "bet_type",
                "record_id",
                "match_name",
                "league",
                "market",
                "selection",
                "line",
                "my_odd",
                "model_fair_odd",
                "model_probability",
                "predicted_value",
                "implied_probability",
                "edge",
                "ev",
                "recommended_stake",
                "bankroll_snapshot",
                "status",
                "created_at",
                "updated_at",
                "event_date",
                "pnl",
            ]
        )

    settled = settle_legacy_bets(frame, results_frame)
    fallback_league = (
        settled["league"]
        if "league" in settled.columns
        else pd.Series(pd.NA, index=settled.index, dtype="object")
    )
    fallback_selection = (
        settled["picked_team"]
        if "picked_team" in settled.columns
        else pd.Series(pd.NA, index=settled.index, dtype="object")
    )
    fallback_model_selection = (
        settled["model_selected_team"]
        if "model_selected_team" in settled.columns
        else pd.Series(pd.NA, index=settled.index, dtype="object")
    )
    normalized = pd.DataFrame(
        {
            "source_table": "bets",
            "bet_type": "real",
            "record_id": settled.get("id"),
            "match_name": settled.get("match_name"),
            "league": settled["settled_league"].fillna(fallback_league),
            "market": settled.get("market", "First Blood"),
            "selection": fallback_selection.fillna(fallback_model_selection),
            "line": pd.NA,
            "my_odd": pd.to_numeric(settled.get("my_odd"), errors="coerce"),
            "model_fair_odd": pd.to_numeric(settled.get("model_fair_odd"), errors="coerce"),
            "model_probability": pd.to_numeric(1.0 / settled.get("model_fair_odd"), errors="coerce"),
            "predicted_value": pd.NA,
            "implied_probability": pd.to_numeric(1.0 / settled.get("my_odd"), errors="coerce"),
            "edge": pd.to_numeric((1.0 / settled.get("model_fair_odd")) - (1.0 / settled.get("my_odd")), errors="coerce"),
            "ev": pd.to_numeric((1.0 / settled.get("model_fair_odd")) * settled.get("my_odd") - 1.0, errors="coerce"),
            "recommended_stake": 1.0,
            "bankroll_snapshot": pd.NA,
            "status": settled.get("settled_status").fillna(settled.get("status")).fillna("pending"),
            "created_at": pd.to_datetime(settled.get("created_at"), utc=True, errors="coerce"),
            "updated_at": pd.to_datetime(settled.get("updated_at"), utc=True, errors="coerce"),
            "event_date": pd.to_datetime(settled.get("settled_match_date"), utc=True, errors="coerce"),
        }
    )
    normalized["pnl"] = normalized.apply(calculate_pick_pnl, axis=1)
    return normalized


def normalize_picks(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "source_table",
                "bet_type",
                "record_id",
                "match_name",
                "league",
                "market",
                "selection",
                "line",
                "my_odd",
                "model_fair_odd",
                "model_probability",
                "predicted_value",
                "implied_probability",
                "edge",
                "ev",
                "recommended_stake",
                "bankroll_snapshot",
                "status",
                "created_at",
                "updated_at",
                "event_date",
                "pnl",
            ]
        )

    normalized = pd.DataFrame(
        {
            "source_table": "picks",
            "bet_type": frame.get("bet_type", pd.Series("real", index=frame.index)).fillna("real"),
            "record_id": frame.get("id"),
            "match_name": frame.get("match_name"),
            "league": frame.get("league"),
            "market": frame.get("market"),
            "selection": frame.get("selection"),
            "line": pd.to_numeric(frame.get("line"), errors="coerce"),
            "my_odd": pd.to_numeric(frame.get("my_odd"), errors="coerce"),
            "model_fair_odd": pd.to_numeric(frame.get("model_fair_odd"), errors="coerce"),
            "model_probability": pd.to_numeric(frame.get("model_probability"), errors="coerce"),
            "predicted_value": pd.to_numeric(frame.get("predicted_value"), errors="coerce"),
            "implied_probability": pd.to_numeric(frame.get("implied_probability"), errors="coerce"),
            "edge": pd.to_numeric(frame.get("edge"), errors="coerce"),
            "ev": pd.to_numeric(frame.get("ev"), errors="coerce"),
            "recommended_stake": pd.to_numeric(frame.get("recommended_stake"), errors="coerce"),
            "bankroll_snapshot": pd.to_numeric(frame.get("bankroll_snapshot"), errors="coerce"),
            "status": frame.get("status").fillna("pending"),
            "created_at": pd.to_datetime(frame.get("created_at"), utc=True, errors="coerce"),
            "updated_at": pd.to_datetime(frame.get("updated_at"), utc=True, errors="coerce"),
            "event_date": pd.NaT,
        }
    )
    normalized["pnl"] = normalized.apply(calculate_pick_pnl, axis=1)
    return normalized


def normalize_settled_picks(frame: pd.DataFrame, results_frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return normalize_picks(frame)

    settled = settle_picks(frame, results_frame)
    normalized = pd.DataFrame(
        {
            "source_table": "picks",
            "bet_type": settled.get("bet_type", pd.Series("real", index=settled.index)).fillna("real"),
            "record_id": settled.get("id"),
            "match_name": settled.get("match_name"),
            "league": settled.get("settled_league").fillna(settled.get("league")),
            "market": settled.get("market"),
            "selection": settled.get("selection"),
            "line": pd.to_numeric(settled.get("line"), errors="coerce"),
            "my_odd": pd.to_numeric(settled.get("my_odd"), errors="coerce"),
            "model_fair_odd": pd.to_numeric(settled.get("model_fair_odd"), errors="coerce"),
            "model_probability": pd.to_numeric(settled.get("model_probability"), errors="coerce"),
            "predicted_value": pd.to_numeric(settled.get("predicted_value"), errors="coerce"),
            "implied_probability": pd.to_numeric(settled.get("implied_probability"), errors="coerce"),
            "edge": pd.to_numeric(settled.get("edge"), errors="coerce"),
            "ev": pd.to_numeric(settled.get("ev"), errors="coerce"),
            "recommended_stake": pd.to_numeric(settled.get("recommended_stake"), errors="coerce"),
            "bankroll_snapshot": pd.to_numeric(settled.get("bankroll_snapshot"), errors="coerce"),
            "status": settled.get("settled_status").fillna(settled.get("status")).fillna("pending"),
            "created_at": pd.to_datetime(settled.get("created_at"), utc=True, errors="coerce"),
            "updated_at": pd.to_datetime(settled.get("updated_at"), utc=True, errors="coerce"),
            "event_date": pd.to_datetime(settled.get("settled_match_date"), utc=True, errors="coerce"),
            "actual_value": settled.get("settled_actual_value"),
        }
    )
    normalized["pnl"] = normalized.apply(calculate_pick_pnl, axis=1)
    return normalized


def calculate_pick_pnl(row: pd.Series) -> float | None:
    stake = row.get("recommended_stake")
    odd = row.get("my_odd")
    status = str(row.get("status", "")).lower()
    if pd.isna(stake) or pd.isna(odd):
        return None
    if status == "won":
        return float(stake) * (float(odd) - 1.0)
    if status == "lost":
        return -float(stake)
    if status == "void":
        return 0.0
    return None


def build_unified_records(
    bets_frame: pd.DataFrame,
    picks_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
) -> pd.DataFrame:
    unified = pd.concat(
        [normalize_legacy_bets(bets_frame, results_frame), normalize_settled_picks(picks_frame, results_frame)],
        ignore_index=True,
    )
    if unified.empty:
        return unified
    unified["market"] = unified["market"].astype(str).str.strip()
    unified["league"] = unified["league"].astype(str).replace({"nan": pd.NA})
    unified["status"] = unified["status"].astype(str).str.strip().str.lower()
    unified = unified.sort_values("created_at", ascending=False, na_position="last").reset_index(drop=True)
    return unified


def compute_summary_metrics(records: pd.DataFrame) -> dict[str, Any]:
    if records.empty:
        return {
            "total_records": 0,
            "pending_records": 0,
            "settled_records": 0,
            "won_records": 0,
            "lost_records": 0,
            "void_records": 0,
            "total_stake": 0.0,
            "total_pnl": 0.0,
            "roi": None,
            "avg_edge": None,
            "avg_ev": None,
            "current_bankroll": None,
        }

    pending_mask = records["status"].eq("pending")
    settled_mask = records["status"].isin(["won", "lost", "void"])
    total_stake = float(records.loc[settled_mask, "recommended_stake"].fillna(0.0).sum())
    total_pnl = float(records.loc[settled_mask, "pnl"].fillna(0.0).sum())

    return {
        "total_records": int(len(records)),
        "pending_records": int(pending_mask.sum()),
        "settled_records": int(settled_mask.sum()),
        "won_records": int(records["status"].eq("won").sum()),
        "lost_records": int(records["status"].eq("lost").sum()),
        "void_records": int(records["status"].eq("void").sum()),
        "total_stake": total_stake,
        "total_pnl": total_pnl,
        "roi": (total_pnl / total_stake) if total_stake > 0 else None,
        "avg_edge": (
            float(records["edge"].dropna().mean()) if records["edge"].notna().any() else None
        ),
        "avg_ev": float(records["ev"].dropna().mean()) if records["ev"].notna().any() else None,
        "current_bankroll": None,
    }


def compute_current_bankroll(records: pd.DataFrame, initial_bankroll: float) -> float:
    if records.empty:
        return float(initial_bankroll)
    settled_mask = records["status"].isin(["won", "lost", "void"])
    settled_pnl = float(records.loc[settled_mask, "pnl"].fillna(0.0).sum())
    return float(initial_bankroll) + settled_pnl


def apply_filters(
    records: pd.DataFrame,
    source_filter: list[str],
    league_filter: list[str],
    market_filter: list[str],
    status_filter: list[str],
) -> pd.DataFrame:
    filtered = records.copy()
    if source_filter:
        filtered = filtered.loc[filtered["source_table"].isin(source_filter)]
    if league_filter:
        filtered = filtered.loc[filtered["league"].isin(league_filter)]
    if market_filter:
        filtered = filtered.loc[filtered["market"].isin(market_filter)]
    if status_filter:
        filtered = filtered.loc[filtered["status"].isin(status_filter)]
    return filtered.reset_index(drop=True)
