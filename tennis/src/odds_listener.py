from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "tennis_market_odds.csv"
THE_ODDS_API_BASE_URL = os.getenv("THE_ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4").rstrip("/")
SPORT_KEY = os.getenv("TENNIS_ODDS_SPORT_KEY", "tennis_atp")
REGIONS = os.getenv("TENNIS_ODDS_REGIONS", "eu,uk")
MARKETS = "h2h,totals,spreads"


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


def save_market_odds(frame: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def run(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    events = fetch_market_odds()
    frame = normalize_market_odds(events)
    save_market_odds(frame, output_path)
    return frame


if __name__ == "__main__":
    odds_frame = run()
    print(f"Saved {len(odds_frame)} market odds rows to {OUTPUT_PATH}")
    print(odds_frame.to_string(index=False))
