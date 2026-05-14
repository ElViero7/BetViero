from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base import LoLPipelineError


REQUIRED_PREDICTION_COLUMNS = {
    "blue_team",
    "red_team",
    "blue_firstblood_prob",
    "red_firstblood_prob",
}


@dataclass(frozen=True)
class ListenerConfig:
    odds_api_key: str
    odds_api_base_url: str
    odds_sport_key: str
    odds_market_key: str
    odds_region: str
    odds_bookmakers: str
    odds_commence_lookahead_hours: int
    predictions_input_path: Path
    odds_snapshots_output_path: Path
    ev_signals_output_path: Path
    listener_state_path: Path
    team_name_overrides_path: Path
    value_edge_threshold: float
    min_ev_threshold: float
    default_stake_units: float
    listener_poll_minutes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll The Odds API for First Blood prices and flag EV+ paper trading signals."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one polling cycle and exit.",
    )
    parser.add_argument(
        "--predictions-input",
        type=Path,
        default=None,
        help="Override the predictions input path from .env.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise LoLPipelineError(f"Missing required environment variable: {name}")
    return value


def load_config(args: argparse.Namespace) -> ListenerConfig:
    load_dotenv()
    predictions_input_path = (
        args.predictions_input
        if args.predictions_input is not None
        else Path(os.getenv("PREDICTIONS_INPUT_PATH", "data/predictions/upcoming_firstblood_predictions.csv"))
    )
    return ListenerConfig(
        odds_api_key=require_env("ODDS_API_KEY"),
        odds_api_base_url=os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4").rstrip("/"),
        odds_sport_key=require_env("ODDS_SPORT_KEY"),
        odds_market_key=require_env("ODDS_MARKET_KEY"),
        odds_region=os.getenv("ODDS_REGION", "eu").strip(),
        odds_bookmakers=os.getenv("ODDS_BOOKMAKERS", "").strip(),
        odds_commence_lookahead_hours=int(os.getenv("ODDS_COMMENCE_LOOKAHEAD_HOURS", "36")),
        predictions_input_path=predictions_input_path,
        odds_snapshots_output_path=Path(
            os.getenv("ODDS_SNAPSHOTS_OUTPUT_PATH", "data/paper_trading/odds_snapshots.jsonl")
        ),
        ev_signals_output_path=Path(
            os.getenv("EV_SIGNALS_OUTPUT_PATH", "data/paper_trading/ev_signals.jsonl")
        ),
        listener_state_path=Path(
            os.getenv("LISTENER_STATE_PATH", "data/paper_trading/listener_state.json")
        ),
        team_name_overrides_path=Path(
            os.getenv("TEAM_NAME_OVERRIDES_PATH", "data/config/team_name_overrides.json")
        ),
        value_edge_threshold=float(os.getenv("VALUE_EDGE_THRESHOLD", "0.025")),
        min_ev_threshold=float(os.getenv("MIN_EV_THRESHOLD", "0.02")),
        default_stake_units=float(os.getenv("DEFAULT_STAKE_UNITS", "1.0")),
        listener_poll_minutes=int(os.getenv("LISTENER_POLL_MINUTES", "30")),
    )


def normalize_team_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", name.lower())
    return cleaned.strip()


def load_name_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {"odds_api": {}, "predictions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def canonicalize_team_name(name: str, provider: str, overrides: dict[str, dict[str, str]]) -> str:
    provider_map = overrides.get(provider, {})
    normalized_name = normalize_team_name(name)
    return normalize_team_name(provider_map.get(normalized_name, name))


def load_predictions(path: Path, overrides: dict[str, dict[str, str]]) -> pd.DataFrame:
    if not path.exists():
        raise LoLPipelineError(f"Predictions input file not found: {path}")

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = duckdb.read_parquet(str(path)).df()
    else:
        raise LoLPipelineError(f"Unsupported predictions input format: {path.suffix}")

    missing_columns = REQUIRED_PREDICTION_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise LoLPipelineError(
            f"Predictions file is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    prepared = frame.copy()
    prepared["blue_team"] = prepared["blue_team"].astype(str).str.strip()
    prepared["red_team"] = prepared["red_team"].astype(str).str.strip()
    prepared["blue_firstblood_prob"] = pd.to_numeric(
        prepared["blue_firstblood_prob"], errors="coerce"
    )
    prepared["red_firstblood_prob"] = pd.to_numeric(
        prepared["red_firstblood_prob"], errors="coerce"
    )
    prepared["prediction_match_key"] = prepared.apply(
        lambda row: build_match_key(
            canonicalize_team_name(row["blue_team"], "predictions", overrides),
            canonicalize_team_name(row["red_team"], "predictions", overrides),
        ),
        axis=1,
    )
    if "odds_event_id" in prepared.columns:
        prepared["odds_event_id"] = prepared["odds_event_id"].astype(str).str.strip()
    else:
        prepared["odds_event_id"] = ""
    return prepared


def build_match_key(team_a: str, team_b: str) -> str:
    return "|".join(sorted([team_a, team_b]))


def request_json(url: str, params: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json(), dict(response.headers)


def fetch_events(config: ListenerConfig) -> tuple[list[dict[str, Any]], dict[str, str]]:
    now_utc = datetime.now(UTC)
    commence_from = (now_utc - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    url = f"{config.odds_api_base_url}/sports/{config.odds_sport_key}/events"
    params = {
        "apiKey": config.odds_api_key,
        "dateFormat": "iso",
        "commenceTimeFrom": commence_from,
    }
    payload, headers = request_json(url, params)
    return list(payload), headers


def fetch_event_odds(config: ListenerConfig, event_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    url = f"{config.odds_api_base_url}/sports/{config.odds_sport_key}/events/{event_id}/odds"
    params: dict[str, Any] = {
        "apiKey": config.odds_api_key,
        "regions": config.odds_region,
        "markets": config.odds_market_key,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    if config.odds_bookmakers:
        params["bookmakers"] = config.odds_bookmakers
    payload, headers = request_json(url, params)
    return dict(payload), headers


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"seen_signal_keys": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")


def match_predictions_to_events(
    predictions: pd.DataFrame,
    events: list[dict[str, Any]],
    overrides: dict[str, dict[str, str]],
) -> list[tuple[pd.Series, dict[str, Any]]]:
    matches: list[tuple[pd.Series, dict[str, Any]]] = []

    predictions_by_event_id = {
        str(row["odds_event_id"]).strip(): row
        for _, row in predictions.loc[predictions["odds_event_id"] != ""].iterrows()
    }
    predictions_by_match_key = {
        row["prediction_match_key"]: row for _, row in predictions.iterrows()
    }

    for event in events:
        event_id = str(event.get("id", "")).strip()
        if event_id and event_id in predictions_by_event_id:
            matches.append((predictions_by_event_id[event_id], event))
            continue

        home_team = canonicalize_team_name(str(event.get("home_team", "")), "odds_api", overrides)
        away_team = canonicalize_team_name(str(event.get("away_team", "")), "odds_api", overrides)
        event_match_key = build_match_key(home_team, away_team)
        prediction_row = predictions_by_match_key.get(event_match_key)
        if prediction_row is not None:
            matches.append((prediction_row, event))

    return matches


def compute_kelly_fraction(probability: float, odds_decimal: float) -> float:
    b = odds_decimal - 1.0
    if b <= 0:
        return 0.0
    kelly = ((probability * odds_decimal) - 1.0) / b
    return max(0.0, kelly)


def choose_side_probability(prediction: pd.Series, outcome_name: str, overrides: dict[str, dict[str, str]]) -> tuple[str | None, float | None]:
    outcome_key = canonicalize_team_name(outcome_name, "odds_api", overrides)
    blue_key = canonicalize_team_name(str(prediction["blue_team"]), "predictions", overrides)
    red_key = canonicalize_team_name(str(prediction["red_team"]), "predictions", overrides)

    if outcome_key == blue_key:
        return "blue", float(prediction["blue_firstblood_prob"])
    if outcome_key == red_key:
        return "red", float(prediction["red_firstblood_prob"])
    return None, None


def build_snapshot_record(
    event: dict[str, Any],
    bookmaker: dict[str, Any],
    market: dict[str, Any],
    outcome: dict[str, Any],
    side: str,
    fair_probability: float,
) -> dict[str, Any]:
    price = float(outcome["price"])
    implied_probability = 1.0 / price
    edge = fair_probability - implied_probability
    ev = (fair_probability * price) - 1.0
    kelly_fraction = compute_kelly_fraction(fair_probability, price)
    return {
        "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "odds_event_id": event.get("id"),
        "commence_time": event.get("commence_time"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "bookmaker_key": bookmaker.get("key"),
        "bookmaker_title": bookmaker.get("title"),
        "market_key": market.get("key"),
        "side": side,
        "outcome_name": outcome.get("name"),
        "decimal_odds": price,
        "fair_probability": fair_probability,
        "implied_probability": implied_probability,
        "edge": edge,
        "ev": ev,
        "kelly_fraction": kelly_fraction,
    }


def build_signal_key(snapshot: dict[str, Any]) -> str:
    return "|".join(
        [
            str(snapshot["odds_event_id"]),
            str(snapshot["bookmaker_key"]),
            str(snapshot["market_key"]),
            str(snapshot["side"]),
            f"{snapshot['decimal_odds']:.4f}",
        ]
    )


def run_cycle(config: ListenerConfig) -> dict[str, Any]:
    overrides = load_name_overrides(config.team_name_overrides_path)
    predictions = load_predictions(config.predictions_input_path, overrides)
    state = load_state(config.listener_state_path)
    seen_signal_keys = set(state.get("seen_signal_keys", []))

    events, event_headers = fetch_events(config)
    matched_pairs = match_predictions_to_events(predictions, events, overrides)

    snapshot_records: list[dict[str, Any]] = []
    signal_records: list[dict[str, Any]] = []
    quota_headers: dict[str, str] = event_headers.copy()

    for prediction_row, event in matched_pairs:
        commence_time_raw = str(event.get("commence_time", ""))
        if commence_time_raw:
            commence_time = datetime.fromisoformat(commence_time_raw.replace("Z", "+00:00"))
            if commence_time > datetime.now(UTC) + timedelta(hours=config.odds_commence_lookahead_hours):
                continue

        try:
            event_odds, odds_headers = fetch_event_odds(config, str(event["id"]))
        except requests.HTTPError as exc:
            print(f"Skipping event {event.get('id')} due to HTTP error: {exc}")
            continue

        quota_headers = odds_headers
        bookmakers = event_odds.get("bookmakers", [])
        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                if market.get("key") != config.odds_market_key:
                    continue
                for outcome in market.get("outcomes", []):
                    side, fair_probability = choose_side_probability(
                        prediction=prediction_row,
                        outcome_name=str(outcome.get("name", "")),
                        overrides=overrides,
                    )
                    if side is None or fair_probability is None:
                        continue
                    snapshot = build_snapshot_record(
                        event=event,
                        bookmaker=bookmaker,
                        market=market,
                        outcome=outcome,
                        side=side,
                        fair_probability=fair_probability,
                    )
                    snapshot_records.append(snapshot)

                    if (
                        snapshot["edge"] >= config.value_edge_threshold
                        and snapshot["ev"] >= config.min_ev_threshold
                    ):
                        signal_key = build_signal_key(snapshot)
                        if signal_key in seen_signal_keys:
                            continue
                        seen_signal_keys.add(signal_key)
                        signal = {
                            **snapshot,
                            "stake_units": config.default_stake_units,
                            "status": "open",
                        }
                        signal_records.append(signal)

    append_jsonl(config.odds_snapshots_output_path, snapshot_records)
    append_jsonl(config.ev_signals_output_path, signal_records)
    save_state(
        config.listener_state_path,
        {"seen_signal_keys": sorted(seen_signal_keys)},
    )

    return {
        "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "predictions_loaded": int(len(predictions)),
        "events_seen": int(len(events)),
        "matched_events": int(len(matched_pairs)),
        "snapshots_written": int(len(snapshot_records)),
        "signals_written": int(len(signal_records)),
        "quota_used_last": quota_headers.get("x-requests-last"),
        "quota_remaining": quota_headers.get("x-requests-remaining"),
        "quota_used_total": quota_headers.get("x-requests-used"),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args)

    while True:
        summary = run_cycle(config)
        print(json.dumps(summary, indent=2))
        if args.once:
            return
        time.sleep(config.listener_poll_minutes * 60)


if __name__ == "__main__":
    main()
