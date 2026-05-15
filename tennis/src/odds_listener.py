from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from db import TennisDatabaseError, fetch_tennis_table, upsert_tennis_rows
except ModuleNotFoundError:
    from tennis.src.db import TennisDatabaseError, fetch_tennis_table, upsert_tennis_rows


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "tennis_market_odds.csv"
THE_ODDS_API_BASE_URL = os.getenv("THE_ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4").rstrip("/")
SPORT_KEY = os.getenv("TENNIS_ODDS_SPORT_KEY", "tennis_atp")
REGIONS = os.getenv("TENNIS_ODDS_REGIONS", "eu,uk")
MARKETS = "h2h,totals,spreads"
UPCOMING_ODDS_TABLE = "upcoming_odds"
UPCOMING_ODDS_CONFLICT_KEY = "quote_key"


class TennisOddsListenerError(RuntimeError):
    """Raised when tennis market odds cannot be fetched or normalized."""


def get_mock_odds() -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return [
        {
            "id": "mock_alcaraz_sinner",
            "sport_key": "tennis_atp",
            "commence_time": now,
            "home_team": "C Alcaraz",
            "away_team": "Jannik Sinner",
            "bookmakers": [
                {
                    "key": "mockbook",
                    "title": "MockBook",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "C Alcaraz", "price": 1.75},
                                {"name": "Jannik Sinner", "price": 2.10},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.91, "point": 22.5},
                                {"name": "Under", "price": 1.91, "point": 22.5},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "C Alcaraz", "price": 1.87, "point": -2.5},
                                {"name": "Jannik Sinner", "price": 1.95, "point": 2.5},
                            ],
                        },
                    ],
                }
            ],
        },
        {
            "id": "mock_djokovic_medvedev",
            "sport_key": "tennis_atp",
            "commence_time": now,
            "home_team": "Djokovic N.",
            "away_team": "D Medvedev",
            "bookmakers": [
                {
                    "key": "mockbook",
                    "title": "MockBook",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Djokovic N.", "price": 1.83},
                                {"name": "D Medvedev", "price": 2.00},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.88, "point": 23.5},
                                {"name": "Under", "price": 1.94, "point": 23.5},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Djokovic N.", "price": 1.90, "point": -1.5},
                                {"name": "D Medvedev", "price": 1.90, "point": 1.5},
                            ],
                        },
                    ],
                }
            ],
        },
    ]


def fetch_odds_from_api(api_key: str) -> list[dict[str, Any]]:
    return fetch_odds_for_sport(api_key, SPORT_KEY)


def fetch_odds_for_sport(api_key: str, sport_key: str) -> list[dict[str, Any]]:
    url = f"{THE_ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk",
        "markets": MARKETS,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    response = requests.get(url, params=params, timeout=30)
    if response.status_code >= 400:
        raise TennisOddsListenerError(f"The-Odds-API error {response.status_code}: {response.text}")
    return response.json()


def discover_active_tennis_sports(api_key: str) -> list[str]:
    response = requests.get(
        f"{THE_ODDS_API_BASE_URL}/sports",
        params={"apiKey": api_key, "all": "false"},
        timeout=30,
    )
    if response.status_code >= 400:
        raise TennisOddsListenerError(f"The-Odds-API sports error {response.status_code}: {response.text}")
    sports = response.json()
    atp_keys = [
        str(sport["key"])
        for sport in sports
        if str(sport.get("key", "")).startswith("tennis_atp")
    ]
    if atp_keys:
        return sorted(set(atp_keys))

    tennis_keys = [
        str(sport["key"])
        for sport in sports
        if str(sport.get("key", "")).startswith("tennis_")
    ]
    return sorted(set(tennis_keys))


def fetch_market_odds() -> list[dict[str, Any]]:
    api_key = os.getenv("THE_ODDS_API_KEY", "").strip() or os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        print("THE_ODDS_API_KEY not found. Using mock tennis odds.")
        return get_mock_odds()
    try:
        return fetch_odds_from_api(api_key)
    except TennisOddsListenerError as exc:
        if "UNKNOWN_SPORT" not in str(exc):
            raise

    active_sports = discover_active_tennis_sports(api_key)
    if not active_sports:
        print("No active tennis sport keys found. Using mock tennis odds.")
        return get_mock_odds()

    events: list[dict[str, Any]] = []
    for sport_key in active_sports:
        try:
            sport_events = fetch_odds_for_sport(api_key, sport_key)
        except TennisOddsListenerError as exc:
            print(f"Skipping {sport_key}: {exc}")
            continue
        events.extend(sport_events)
    if not events:
        print("No tennis odds returned by active sport keys. Using mock tennis odds.")
        return get_mock_odds()
    return events


def normalize_market_odds(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        player_a = str(event.get("home_team") or "")
        player_b = str(event.get("away_team") or "")
        for bookmaker in event.get("bookmakers", []) or []:
            bookmaker_key = bookmaker.get("key")
            bookmaker_title = bookmaker.get("title")
            for market in bookmaker.get("markets", []) or []:
                market_key = str(market.get("key", ""))
                if market_key not in {"h2h", "totals", "spreads"}:
                    continue
                for outcome in market.get("outcomes", []) or []:
                    line = outcome.get("point")
                    if market_key == "totals" and line is not None and float(line) < 15:
                        continue
                    rows.append(
                        {
                            "event_id": event.get("id"),
                            "sport_key": event.get("sport_key"),
                            "commence_time": event.get("commence_time"),
                            "jugador_a_api": player_a,
                            "jugador_b_api": player_b,
                            "bookmaker_key": bookmaker_key,
                            "bookmaker_title": bookmaker_title,
                            "casa_apuestas": bookmaker_title,
                            "market": market_key,
                            "selection": outcome.get("name"),
                            "line": line,
                            "odds": float(outcome.get("price")),
                        }
                    )
    return pd.DataFrame(rows)


def build_quote_key(row: pd.Series | dict[str, Any]) -> str:
    line = row.get("line")
    line_key = "null" if pd.isna(line) else f"{float(line):.4f}"
    return "|".join(
        [
            "the_odds_api",
            str(row.get("event_id") or ""),
            str(row.get("bookmaker_key") or ""),
            str(row.get("market") or ""),
            str(row.get("selection") or ""),
            line_key,
        ]
    )


def build_upcoming_odds_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    rows: list[dict[str, Any]] = []
    collected_at = datetime.now(timezone.utc).isoformat()
    for _, row in frame.iterrows():
        price = float(row["odds"])
        line = row.get("line")
        rows.append(
            {
                "quote_key": build_quote_key(row),
                "source": "the_odds_api",
                "external_event_id": str(row.get("event_id") or ""),
                "sport_key": row.get("sport_key"),
                "commence_time": row.get("commence_time"),
                "player_1": str(row.get("jugador_a_api") or ""),
                "player_2": str(row.get("jugador_b_api") or ""),
                "bookmaker_key": str(row.get("bookmaker_key") or ""),
                "bookmaker_title": row.get("bookmaker_title"),
                "market_key": str(row.get("market") or ""),
                "selection": str(row.get("selection") or ""),
                "line": None if pd.isna(line) else float(line),
                "price": price,
                "implied_probability": 1.0 / price if price > 0 else None,
                "collected_at": collected_at,
                "raw_payload": {
                    "event_id": row.get("event_id"),
                    "casa_apuestas": row.get("casa_apuestas"),
                    "market": row.get("market"),
                    "selection": row.get("selection"),
                    "line": None if pd.isna(line) else float(line),
                    "odds": price,
                },
            }
        )
    return rows


def fetch_existing_upcoming_prices() -> dict[str, float]:
    existing = fetch_tennis_table(
        UPCOMING_ODDS_TABLE,
        select="quote_key,price",
        filters={"quote_key": "not.is.null"},
    )
    if existing.empty:
        return {}

    prices: dict[str, float] = {}
    for _, row in existing.iterrows():
        quote_key = row.get("quote_key")
        price = row.get("price")
        if pd.isna(quote_key) or pd.isna(price):
            continue
        prices[str(quote_key)] = float(price)
    return prices


def filter_changed_upcoming_odds_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    existing_prices = fetch_existing_upcoming_prices()
    changed_rows: list[dict[str, Any]] = []
    for row in rows:
        quote_key = str(row["quote_key"])
        current_price = float(row["price"])
        existing_price = existing_prices.get(quote_key)
        if existing_price is None or abs(existing_price - current_price) >= 1e-9:
            changed_rows.append(row)
    return changed_rows


def save_upcoming_odds_to_supabase(frame: pd.DataFrame) -> int:
    rows = build_upcoming_odds_rows(frame)
    changed_rows = filter_changed_upcoming_odds_rows(rows)
    if not changed_rows:
        return 0

    upsert_tennis_rows(
        UPCOMING_ODDS_TABLE,
        changed_rows,
        on_conflict=UPCOMING_ODDS_CONFLICT_KEY,
    )
    return len(changed_rows)


def save_market_odds(frame: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def run(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    events = fetch_market_odds()
    frame = normalize_market_odds(events)
    save_market_odds(frame, output_path)
    try:
        saved_rows = save_upcoming_odds_to_supabase(frame)
        print(f"[Supabase] Upcoming odds new/changed rows upserted: {saved_rows}")
    except TennisDatabaseError as exc:
        print(f"[Supabase] Upcoming odds not saved: {exc}")
    return frame


if __name__ == "__main__":
    odds_frame = run()
    print(f"Saved {len(odds_frame)} market odds rows to {OUTPUT_PATH}")
    print(odds_frame.to_string(index=False))
