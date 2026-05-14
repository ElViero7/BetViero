from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests import Response, Session
from requests.exceptions import RequestException

from src.config import get_settings


LOGGER = logging.getLogger(__name__)

FINISHED_STATUSES = {
    "FT",
    "AET",
    "PEN",
}

STAT_KEY_MAP = {
    "Shots on Goal": "shots_on_goal",
    "Shots off Goal": "shots_off_goal",
    "Total Shots": "total_shots",
    "Blocked Shots": "blocked_shots",
    "Shots insidebox": "shots_inside_box",
    "Shots outsidebox": "shots_outside_box",
    "Fouls": "fouls",
    "Corner Kicks": "corner_kicks",
    "Offsides": "offsides",
    "Ball Possession": "ball_possession_pct",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "total_passes",
    "Passes accurate": "passes_accurate",
    "Passes %": "passes_accuracy_pct",
    "expected_goals": "expected_goals",
}


class ApiFootballETLError(RuntimeError):
    """Raised when the API-Football ETL flow fails."""


@dataclass(frozen=True)
class FixtureQuery:
    league_id: int
    season: int
    last_n: int = 100
    timezone: str = "Europe/Madrid"

    def validate(self) -> None:
        if self.league_id <= 0:
            raise ApiFootballETLError("league_id must be a positive integer.")

        if self.season < 2000:
            raise ApiFootballETLError("season must be a valid starting year.")

        if self.last_n <= 0:
            raise ApiFootballETLError("last_n must be greater than 0.")


class ApiFootballETL:
    """
    Extract and normalize historical fixtures and statistics from API-Football.
    """

    def __init__(self, session: Session | None = None, mock_dir: Path | None = None) -> None:
        self._mock_dir = mock_dir
        self._session = session or requests.Session()

        if self._mock_dir is not None:
            self._base_url = ""
            self._timeout = 30.0
            return

        settings = get_settings()
        self._base_url = settings.api_football_base_url.rstrip("/")
        self._timeout = settings.api_football_timeout_seconds
        self._session.headers.update(
            {
                "X-RapidAPI-Key": settings.api_football_key,
                "X-RapidAPI-Host": settings.api_football_host,
            }
        )

    def fetch_last_league_matches(
        self,
        league_id: int,
        season: int,
        last_n: int = 100,
        timezone: str = "Europe/Madrid",
    ) -> pd.DataFrame:
        query = FixtureQuery(
            league_id=league_id,
            season=season,
            last_n=last_n,
            timezone=timezone,
        )
        query.validate()

        payload = self._get(
            endpoint="fixtures",
            params={
                "league": query.league_id,
                "season": query.season,
                "timezone": query.timezone,
            },
        )
        fixtures = self._extract_response_items(payload=payload)
        cleaned_df = self._clean_fixtures(
            fixtures=fixtures,
            query=query,
            finished_only=True,
        )

        if cleaned_df.empty:
            raise ApiFootballETLError(
                "No finished fixtures were returned for the selected league and season."
            )

        return cleaned_df

    def fetch_fixtures_by_date(
        self,
        fixture_date: str,
        league_id: int | None = None,
        season: int | None = None,
        timezone: str = "Europe/Madrid",
    ) -> pd.DataFrame:
        params: dict[str, Any] = {
            "date": fixture_date,
            "timezone": timezone,
        }
        if league_id is not None:
            params["league"] = league_id
        if season is not None:
            params["season"] = season

        payload = self._get(
            endpoint="fixtures",
            params=params,
        )
        fixtures = self._extract_response_items(payload=payload)
        return self._clean_fixtures(
            fixtures=fixtures,
            query=None,
            finished_only=False,
        )

    def fetch_fixture_statistics(self, fixture_id: int) -> pd.DataFrame:
        if fixture_id <= 0:
            raise ApiFootballETLError("fixture_id must be a positive integer.")

        payload = self._get(
            endpoint="fixtures/statistics",
            params={"fixture": fixture_id},
        )
        stats_items = self._extract_response_items(payload=payload)

        if not stats_items:
            return pd.DataFrame()

        return self._clean_fixture_statistics(fixture_id=fixture_id, stats_items=stats_items)

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        mock_payload = self._load_mock_payload(endpoint=endpoint, params=params)
        if mock_payload is not None:
            self._validate_api_payload(payload=mock_payload)
            return mock_payload

        url = f"{self._base_url}/{endpoint}"

        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except RequestException as exc:
            raise ApiFootballETLError(
                f"HTTP request to API-Football failed for endpoint={endpoint} params={params}."
            ) from exc

        payload = self._parse_json_response(response=response)
        self._validate_api_payload(payload=payload)
        return payload

    def _load_mock_payload(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if self._mock_dir is None:
            return None

        api_mock_dir = self._mock_dir / "api_football"
        if endpoint == "fixtures/statistics":
            file_name = f"fixtures_statistics_{params.get('fixture')}.json"
        elif endpoint == "fixtures" and "date" in params:
            file_name = (
                f"fixtures_date_{params.get('date')}_league_{params.get('league', 'all')}"
                f"_season_{params.get('season', 'all')}.json"
            )
        elif endpoint == "fixtures":
            file_name = (
                f"fixtures_league_{params.get('league', 'all')}"
                f"_season_{params.get('season', 'all')}.json"
            )
        else:
            return None

        file_path = api_mock_dir / file_name
        if not file_path.exists():
            raise ApiFootballETLError(f"Mock file not found: {file_path}")

        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiFootballETLError(f"Invalid JSON in mock file: {file_path}") from exc

    @staticmethod
    def _parse_json_response(response: Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiFootballETLError(
                "API-Football returned a non-JSON response."
            ) from exc

        if not isinstance(payload, dict):
            raise ApiFootballETLError(
                "API-Football response JSON is not an object."
            )

        return payload

    @staticmethod
    def _validate_api_payload(payload: dict[str, Any]) -> None:
        errors = payload.get("errors")
        response_data = payload.get("response")

        if errors:
            raise ApiFootballETLError(
                f"API-Football reported errors: {errors}"
            )

        if response_data is None:
            raise ApiFootballETLError(
                "API-Football response does not contain a 'response' field."
            )

        if not isinstance(response_data, list):
            raise ApiFootballETLError(
                "API-Football 'response' field is not a list."
            )

    @staticmethod
    def _extract_response_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        response_items = payload.get("response", [])

        return [
            item for item in response_items
            if isinstance(item, dict)
        ]

    def _clean_fixtures(
        self,
        fixtures: list[dict[str, Any]],
        query: FixtureQuery | None,
        finished_only: bool,
    ) -> pd.DataFrame:
        if not fixtures:
            return pd.DataFrame()

        frame = pd.json_normalize(fixtures, sep="_")

        rename_map = {
            "fixture_id": "fixture_id",
            "fixture_referee": "referee",
            "fixture_timezone": "fixture_timezone",
            "fixture_date": "match_date",
            "fixture_timestamp": "match_timestamp",
            "fixture_periods_first": "first_half_timestamp",
            "fixture_periods_second": "second_half_timestamp",
            "fixture_venue_id": "venue_id",
            "fixture_venue_name": "venue_name",
            "fixture_venue_city": "venue_city",
            "fixture_status_long": "status_long",
            "fixture_status_short": "status_short",
            "fixture_status_elapsed": "status_elapsed_minutes",
            "league_id": "league_id",
            "league_name": "league_name",
            "league_country": "league_country",
            "league_logo": "league_logo",
            "league_flag": "league_flag",
            "league_season": "season",
            "league_round": "round",
            "teams_home_id": "home_team_id",
            "teams_home_name": "home_team_name",
            "teams_home_logo": "home_team_logo",
            "teams_home_winner": "home_team_winner",
            "teams_away_id": "away_team_id",
            "teams_away_name": "away_team_name",
            "teams_away_logo": "away_team_logo",
            "teams_away_winner": "away_team_winner",
            "goals_home": "home_goals",
            "goals_away": "away_goals",
            "score_halftime_home": "halftime_home_goals",
            "score_halftime_away": "halftime_away_goals",
            "score_fulltime_home": "fulltime_home_goals",
            "score_fulltime_away": "fulltime_away_goals",
            "score_extratime_home": "extratime_home_goals",
            "score_extratime_away": "extratime_away_goals",
            "score_penalty_home": "penalty_home_goals",
            "score_penalty_away": "penalty_away_goals",
        }

        available_columns = {
            original: renamed
            for original, renamed in rename_map.items()
            if original in frame.columns
        }
        cleaned = frame.rename(columns=available_columns)
        cleaned = cleaned.loc[:, list(available_columns.values())].copy()

        if finished_only and "status_short" in cleaned.columns:
            cleaned = cleaned.loc[cleaned["status_short"].isin(FINISHED_STATUSES)].copy()

        if "match_date" in cleaned.columns:
            cleaned["match_date"] = pd.to_datetime(
                cleaned["match_date"],
                utc=True,
                errors="coerce",
            )

        integer_like_columns = [
            "fixture_id",
            "league_id",
            "season",
            "venue_id",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "halftime_home_goals",
            "halftime_away_goals",
            "fulltime_home_goals",
            "fulltime_away_goals",
            "extratime_home_goals",
            "extratime_away_goals",
            "penalty_home_goals",
            "penalty_away_goals",
            "status_elapsed_minutes",
            "first_half_timestamp",
            "second_half_timestamp",
            "match_timestamp",
        ]
        present_integer_like = [
            column for column in integer_like_columns if column in cleaned.columns
        ]
        for column in present_integer_like:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

        for column in ("home_team_winner", "away_team_winner"):
            if column in cleaned.columns:
                cleaned[column] = cleaned[column].astype("boolean")

        cleaned = cleaned.sort_values(
            by=["match_date", "fixture_id"],
            ascending=[False, False],
            na_position="last",
        ).head(query.last_n)
        cleaned = cleaned.reset_index(drop=True)

        if query is not None:
            LOGGER.info(
                "Fetched %s cleaned fixtures for league=%s season=%s.",
                len(cleaned),
                query.league_id,
                query.season,
            )
        else:
            LOGGER.info("Fetched %s cleaned fixtures for date query.", len(cleaned))

        return cleaned

    def _clean_fixture_statistics(
        self,
        fixture_id: int,
        stats_items: list[dict[str, Any]],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for item in stats_items:
            team = item.get("team", {})
            statistics = item.get("statistics", [])

            row: dict[str, Any] = {
                "fixture_id": fixture_id,
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "team_logo": team.get("logo"),
            }

            for stat in statistics:
                stat_type = stat.get("type")
                stat_value = stat.get("value")

                if not stat_type:
                    continue

                normalized_name = STAT_KEY_MAP.get(stat_type)
                if not normalized_name:
                    continue

                row[normalized_name] = self._coerce_stat_value(stat_value)

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows)
        numeric_columns = [column for column in frame.columns if column not in {"team_name", "team_logo"}]
        for column in numeric_columns:
            if column in {"fixture_id", "team_id"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
                continue

            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        return frame

    @staticmethod
    def _coerce_stat_value(value: Any) -> float | int | None:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            cleaned = value.strip().replace("%", "")
            if cleaned == "":
                return None

            try:
                numeric_value = float(cleaned)
            except ValueError:
                return None

            if numeric_value.is_integer():
                return int(numeric_value)

            return numeric_value

        return None
