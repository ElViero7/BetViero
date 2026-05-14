from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

import requests
from requests import Response, Session
from requests.exceptions import RequestException

from src.config import get_settings


LOGGER = logging.getLogger(__name__)


class TheOddsAPIError(RuntimeError):
    """Raised when The Odds API request or parsing fails."""


class TheOddsAPIClient:
    def __init__(self, session: Session | None = None, mock_dir: Path | None = None) -> None:
        self._mock_dir = mock_dir
        self._session = session or requests.Session()

        if self._mock_dir is not None:
            self._api_key = "mock"
            self._base_url = ""
            self._timeout = 30.0
            return

        settings = get_settings()
        self._api_key = settings.the_odds_api_key
        self._base_url = settings.the_odds_api_base_url.rstrip("/")
        self._timeout = settings.the_odds_api_timeout_seconds

    def get_events(
        self,
        sport_key: str,
        commence_time_from: str | None = None,
        event_ids: list[str] | None = None,
        date_format: str = "iso",
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        params: dict[str, Any] = {
            "apiKey": self._api_key,
            "dateFormat": date_format,
        }
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if event_ids:
            params["eventIds"] = ",".join(event_ids)

        return self._get(
            endpoint=f"/sports/{sport_key}/events",
            params=params,
        )

    def get_event_odds(
        self,
        sport_key: str,
        event_id: str,
        regions: list[str],
        markets: list[str],
        bookmakers: list[str] | None = None,
        odds_format: str = "decimal",
        date_format: str = "iso",
    ) -> tuple[dict[str, Any], dict[str, str]]:
        params: dict[str, Any] = {
            "apiKey": self._api_key,
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        data, headers = self._get(
            endpoint=f"/sports/{sport_key}/events/{event_id}/odds",
            params=params,
        )
        if not isinstance(data, dict):
            raise TheOddsAPIError(
                f"Expected dict response for event odds, got {type(data).__name__}."
            )
        return data, headers

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> tuple[Any, dict[str, str]]:
        mock_payload = self._load_mock_payload(endpoint=endpoint)
        if mock_payload is not None:
            return mock_payload, {
                "x-requests-remaining": "mock",
                "x-requests-used": "mock",
                "x-requests-last": "0",
            }

        url = f"{self._base_url}{endpoint}"

        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except RequestException as exc:
            raise TheOddsAPIError(
                f"The Odds API request failed for endpoint={endpoint} params={params}."
            ) from exc

        payload = self._parse_json_response(response)
        quota_headers = self._extract_quota_headers(response)
        LOGGER.info(
            "The Odds API request ok: endpoint=%s remaining=%s used=%s last_cost=%s",
            endpoint,
            quota_headers.get("x-requests-remaining"),
            quota_headers.get("x-requests-used"),
            quota_headers.get("x-requests-last"),
        )
        return payload, quota_headers

    def _load_mock_payload(self, endpoint: str) -> Any | None:
        if self._mock_dir is None:
            return None

        odds_mock_dir = self._mock_dir / "the_odds_api"
        if endpoint.endswith("/events"):
            file_path = odds_mock_dir / "events.json"
        elif "/events/" in endpoint and endpoint.endswith("/odds"):
            event_id = endpoint.split("/events/", 1)[1].split("/odds", 1)[0]
            file_path = odds_mock_dir / f"event_odds_{event_id}.json"
        else:
            return None

        if not file_path.exists():
            raise TheOddsAPIError(f"Mock file not found: {file_path}")

        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TheOddsAPIError(f"Invalid JSON in mock file: {file_path}") from exc

    @staticmethod
    def _parse_json_response(response: Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise TheOddsAPIError("The Odds API returned a non-JSON response.") from exc

    @staticmethod
    def _extract_quota_headers(response: Response) -> dict[str, str]:
        headers = {
            "x-requests-remaining": response.headers.get("x-requests-remaining", ""),
            "x-requests-used": response.headers.get("x-requests-used", ""),
            "x-requests-last": response.headers.get("x-requests-last", ""),
        }
        return headers
