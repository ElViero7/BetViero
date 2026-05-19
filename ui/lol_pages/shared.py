from __future__ import annotations

from datetime import date
import logging
import os
from pathlib import Path
import re

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from lol.src.base import LoLPipelineError
from lol.src.dashboard import (
    build_oracle_results_frame,
    fetch_table,
    load_dashboard_config,
    load_predictions_snapshot,
    settle_legacy_bets,
    settle_picks,
    update_table_row,
)
from lol.src.fair_odds_bot import BotConfig, FairOddsEngine, build_alert_message
from lol.src.vision_parser import apply_ocr_team_corrections, clean_ocr_team_text


logger = logging.getLogger(__name__)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOL_DIR = PROJECT_ROOT / "lol"

DISABLED_LOCAL_PROXY_VALUES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
}
for proxy_env_name in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    if os.environ.get(proxy_env_name) in DISABLED_LOCAL_PROXY_VALUES:
        os.environ.pop(proxy_env_name, None)

os.environ.setdefault("PREDICTIONS_INPUT_PATH", str(LOL_DIR / "data" / "predictions" / "upcoming_multimarket_predictions.csv"))
os.environ.setdefault("ORACLES_ELIXIR_RAW_DIR", str(LOL_DIR / "data" / "raw" / "oracles_elixir"))
os.environ.setdefault("MODEL_ARTIFACT_PATH", str(LOL_DIR / "models" / "modelo_multimarket.pkl"))
os.environ.setdefault(
    "ORACLES_ELIXIR_CANONICAL_PARQUET_PATH",
    str(LOL_DIR / "data" / "processed" / "oracles_elixir_team_rows_training.parquet"),
)
os.environ.setdefault("TEAM_NAME_OVERRIDES_PATH", str(LOL_DIR / "data" / "config" / "team_name_overrides.json"))

@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_dashboard_config()
    bets_frame = fetch_table(config, "bets")
    picks_frame = fetch_table(config, "picks")
    predictions_frame = load_predictions_snapshot(config)
    results_frame = build_oracle_results_frame(config.raw_data_dir)
    return bets_frame, picks_frame, predictions_frame, results_frame

@st.cache_resource(show_spinner=False)
def get_fair_odds_engine(_config: BotConfig, _model_mtime: float = 0.0) -> FairOddsEngine:
    return FairOddsEngine(_config)

def get_lol_model_mtime() -> float:
    model_path = LOL_DIR / "models" / "modelo_multimarket.pkl"
    return model_path.stat().st_mtime if model_path.exists() else 0.0

def get_lol_manual_options(bot_config: BotConfig) -> dict[str, list[str]]:
    frame = get_fair_odds_engine(bot_config, get_lol_model_mtime()).feature_store.team_frame
    if frame.empty:
        return {}
    options: dict[str, list[str]] = {}
    for league, league_frame in frame.groupby("league", dropna=True):
        teams = sorted(league_frame["team_name"].dropna().astype(str).unique().tolist())
        if teams:
            options[str(league)] = teams
    return dict(sorted(options.items()))

def build_manual_source_match_id(league: str, team_a: str, team_b: str, scheduled_at: str) -> str:
    raw_key = f"{league}::{team_a}::{team_b}::{scheduled_at}"
    slug = re.sub(r"[^a-z0-9]+", "-", raw_key.lower()).strip("-")
    return f"manual::{slug}"

def send_alerts_to_telegram(config: BotConfig, frame: pd.DataFrame) -> int:
    sent = 0
    for league_name, league_frame in frame.groupby("league", sort=False):
        requests.post(
            f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
            data={"chat_id": config.telegram_chat_id, "text": f"{league_name}\n"},
            timeout=30,
        )
        for _, row in league_frame.iterrows():
            requests.post(
                f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
                data={"chat_id": config.telegram_chat_id, "text": build_alert_message(row)},
                timeout=30,
            )
            sent += 1
    return sent

def save_predictions_snapshot(frame: pd.DataFrame, output_path: Path, merge: bool = True) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_to_save = frame.copy()
    if not frame_to_save.empty and "source_match_id" in frame_to_save.columns:
        manual_mask = frame_to_save["source_match_id"].astype(str).str.startswith("manual::")
        if manual_mask.any() and "snapshot_source" not in frame_to_save.columns:
            frame_to_save["snapshot_source"] = "manual_remesa"
    if merge and output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, frame_to_save], ignore_index=True)
        combined = combined.drop_duplicates(subset=["match_name"], keep="last")
    else:
        combined = frame_to_save
    combined = combined.sort_values(["league", "scheduled_at", "match_name"], na_position="last").reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    load_dashboard_data.clear()
    return combined

def filter_upcoming_predictions(predictions_frame: pd.DataFrame) -> pd.DataFrame:
    if predictions_frame.empty or "scheduled_at" not in predictions_frame.columns:
        return predictions_frame
    output = predictions_frame.copy()
    output["scheduled_at"] = pd.to_datetime(output["scheduled_at"], utc=True, errors="coerce")
    today_utc = pd.Timestamp.now(tz="UTC").normalize()
    output = output.loc[output["scheduled_at"].isna() | (output["scheduled_at"] >= today_utc)].copy()
    return output.sort_values(["scheduled_at", "league", "match_name"], na_position="last").reset_index(drop=True)

def filter_prediction_view(predictions_frame: pd.DataFrame) -> pd.DataFrame:
    if predictions_frame.empty:
        return predictions_frame
    if "source_match_id" in predictions_frame.columns:
        manual_mask = predictions_frame["source_match_id"].astype(str).str.startswith("manual::")
        snapshot_is_manual = False
        if "snapshot_source" in predictions_frame.columns:
            snapshot_is_manual = predictions_frame["snapshot_source"].astype(str).eq("manual_remesa").any()
        if snapshot_is_manual or manual_mask.all():
            output = predictions_frame.loc[manual_mask].copy()
            if "scheduled_at" in output.columns:
                output["scheduled_at"] = pd.to_datetime(output["scheduled_at"], utc=True, errors="coerce")
                output = output.sort_values(["scheduled_at", "league", "match_name"], na_position="last")
            return output.reset_index(drop=True)
    return filter_upcoming_predictions(predictions_frame)

def enrich_event_dates(records: pd.DataFrame, predictions_frame: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    enriched = records.copy()
    if "event_date" not in enriched.columns:
        enriched["event_date"] = pd.NaT
    enriched["event_date"] = pd.to_datetime(enriched["event_date"], utc=True, errors="coerce")
    if predictions_frame.empty or "match_name" not in predictions_frame.columns or "scheduled_at" not in predictions_frame.columns:
        return enriched
    schedule_map = (
        predictions_frame[["match_name", "scheduled_at"]]
        .dropna(subset=["match_name"])
        .drop_duplicates(subset=["match_name"], keep="last")
        .copy()
    )
    schedule_map["scheduled_at"] = pd.to_datetime(schedule_map["scheduled_at"], utc=True, errors="coerce")
    merged = enriched.merge(schedule_map, on="match_name", how="left")
    merged["event_date"] = merged["event_date"].fillna(merged["scheduled_at"])
    merged = merged.drop(columns=["scheduled_at"])
    return merged

def parse_manual_matches_text(text: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            league_part, match_part = [part.strip() for part in line.split("|", 1)]
            if " vs " not in match_part:
                raise LoLPipelineError(f"Linea invalida: {line}")
            team_a, team_b = [part.strip() for part in match_part.split(" vs ", 1)]
            rows.append({"league": league_part, "team_a": team_a, "team_b": team_b})
        else:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                raise LoLPipelineError(f"Linea invalida: {line}")
            rows.append({"league": parts[0], "team_a": parts[1], "team_b": parts[2]})
    if not rows:
        raise LoLPipelineError("No se detectaron partidos manuales.")
    return pd.DataFrame(rows)

def build_valid_team_list(predictions_frame: pd.DataFrame) -> list[str]:
    if predictions_frame.empty:
        return []
    team_values: list[str] = []
    for column in ["team_a", "team_b"]:
        if column in predictions_frame.columns:
            team_values.extend(predictions_frame[column].dropna().astype(str).tolist())
    cleaned_teams = [apply_ocr_team_corrections(team_name) for team_name in team_values]
    cleaned_teams = [team_name for team_name in cleaned_teams if team_name]
    return sorted(set(cleaned_teams))

def find_prediction_match_name(predictions_frame: pd.DataFrame, team_a: str, team_b: str) -> str | None:
    if predictions_frame.empty or not {"team_a", "team_b", "match_name"}.issubset(predictions_frame.columns):
        return None
    target_pair = {clean_ocr_team_text(team_a).lower(), clean_ocr_team_text(team_b).lower()}
    for _, row in predictions_frame.iterrows():
        row_pair = {
            clean_ocr_team_text(str(row["team_a"])).lower(),
            clean_ocr_team_text(str(row["team_b"])).lower(),
        }
        if row_pair == target_pair:
            return str(row["match_name"])
    return None

def _sync_picks_and_bets(dashboard_config) -> None:
    col_legacy, col_picks = st.columns(2)
    if col_legacy.button("Sincronizar legacy bets", use_container_width=True):
        try:
            bets_frame = fetch_table(dashboard_config, "bets")
            if not bets_frame.empty:
                bets_frame = bets_frame.loc[
                    bets_frame.get("status").fillna("pending").astype(str).str.lower().eq("pending")
                ].copy()
            settled = settle_legacy_bets(bets_frame, build_oracle_results_frame(dashboard_config.raw_data_dir))
            updated = 0
            for _, row in settled.iterrows():
                update_table_row(
                    dashboard_config,
                    "bets",
                    str(row["id"]),
                    {"status": str(row["settled_status"])},
                )
                updated += 1
            load_dashboard_data.clear()
            st.success(f"Legacy bets sincronizadas: {updated}")
        except Exception as exc:
            st.error(str(exc))

    if col_picks.button("Sincronizar picks", use_container_width=True):
        try:
            picks_frame = fetch_table(dashboard_config, "picks")
            if not picks_frame.empty:
                picks_frame = picks_frame.loc[
                    picks_frame.get("status").fillna("pending").astype(str).str.lower().eq("pending")
                ].copy()
            settled = settle_picks(picks_frame, build_oracle_results_frame(dashboard_config.raw_data_dir))
            updated = 0
            for _, row in settled.iterrows():
                update_table_row(
                    dashboard_config,
                    "picks",
                    str(row["id"]),
                    {"status": str(row["settled_status"])},
                )
                updated += 1
            load_dashboard_data.clear()
            st.success(f"Picks sincronizados: {updated}")
        except Exception as exc:
            st.error(str(exc))
