from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import SettingsError
from src.db import DatabaseConnectionError, get_supabase_client
from src.the_odds_api import TheOddsAPIClient, TheOddsAPIError


LOGGER = logging.getLogger(__name__)

SUPPORTED_MARKET_MAP = {
    "match_totals": "totals",
    "team_totals": "team_totals",
    "team_shots_on_target": "team_shots_on_target",
    "match_corners": "alternate_totals_corners",
    "match_cards": "alternate_totals_cards",
    "player_shots_on_target": "player_shots_on_target",
}


@dataclass(frozen=True)
class OddsListenerConfig:
    sport_key: str
    regions: list[str]
    bookmakers: list[str]
    horizon_hours: int
    min_edge: float
    min_ev: float
    mock_dir: Path | None = None


class OddsListenerError(RuntimeError):
    """Raised when the odds listener pipeline fails."""


class OddsListener:
    def __init__(self, odds_client: TheOddsAPIClient | None = None) -> None:
        self._odds_client = odds_client or TheOddsAPIClient()
        self._supabase = get_supabase_client()

    def run_once(self, config: OddsListenerConfig) -> dict[str, Any]:
        context = self._load_prediction_context(config=config)
        if context["predictions_df"].empty:
            return {
                "matches_considered": 0,
                "predictions_considered": 0,
                "odds_inserted": 0,
                "value_bets_inserted": 0,
                "unsupported_predictions": 0,
            }

        event_map = self._map_events_to_matches(
            config=config,
            matches_df=context["matches_df"],
            teams_df=context["teams_df"],
        )
        latest_odds_lookup = self._load_latest_odds_lookup(
            match_ids=context["matches_df"]["id"].tolist()
        )
        predictions_by_match = self._group_predictions_by_match(
            predictions_df=context["predictions_df"],
        )

        odds_rows: list[dict[str, Any]] = []
        value_bet_rows: list[dict[str, Any]] = []
        unsupported_predictions = 0

        for _, match_row in context["matches_df"].iterrows():
            match_id = match_row["id"]
            event = event_map.get(match_id)
            if event is None:
                LOGGER.warning(
                    "No The Odds API event mapped for match_id=%s fixture_id=%s.",
                    match_id,
                    match_row["api_football_fixture_id"],
                )
                continue

            prediction_rows = predictions_by_match.get(match_id, [])
            supported_market_keys = sorted(
                {
                    SUPPORTED_MARKET_MAP[prediction["market_type"]]
                    for prediction in prediction_rows
                    if prediction["market_type"] in SUPPORTED_MARKET_MAP
                }
            )

            unsupported_predictions += sum(
                1
                for prediction in prediction_rows
                if prediction["market_type"] not in SUPPORTED_MARKET_MAP
            )

            if not supported_market_keys:
                continue

            event_odds, _headers = self._odds_client.get_event_odds(
                sport_key=config.sport_key,
                event_id=event["id"],
                regions=config.regions,
                markets=supported_market_keys,
                bookmakers=config.bookmakers or None,
            )
            new_odds_rows, new_value_rows = self._process_event_odds(
                event_odds=event_odds,
                match_row=match_row,
                prediction_rows=prediction_rows,
                latest_odds_lookup=latest_odds_lookup,
                min_edge=config.min_edge,
                min_ev=config.min_ev,
            )
            odds_rows.extend(new_odds_rows)
            value_bet_rows.extend(new_value_rows)

            for odds_row in new_odds_rows:
                lookup_key = self._odds_lookup_key(odds_row)
                latest_odds_lookup[lookup_key] = odds_row

        inserted_odds_df = self._insert_odds_history(odds_rows=odds_rows)
        value_bets_inserted = self._insert_value_bets(
            value_bet_rows=value_bet_rows,
            inserted_odds_df=inserted_odds_df,
        )

        return {
            "matches_considered": int(len(context["matches_df"])),
            "predictions_considered": int(len(context["predictions_df"])),
            "odds_inserted": int(len(inserted_odds_df)),
            "value_bets_inserted": int(value_bets_inserted),
            "unsupported_predictions": int(unsupported_predictions),
        }

    def _load_prediction_context(self, config: OddsListenerConfig) -> dict[str, pd.DataFrame]:
        now_iso = datetime.now(timezone.utc).isoformat()
        horizon_iso = (datetime.now(timezone.utc) + timedelta(hours=config.horizon_hours)).isoformat()

        try:
            matches_response = (
                self._supabase.table("matches")
                .select("id,api_football_fixture_id,match_date,home_team_id,away_team_id,competition_name,status")
                .gte("match_date", now_iso)
                .lte("match_date", horizon_iso)
                .execute()
            )
        except Exception as exc:
            raise DatabaseConnectionError(
                "Failed to fetch upcoming matches from Supabase."
            ) from exc

        matches_df = pd.DataFrame(matches_response.data or [])
        if matches_df.empty:
            return {
                "matches_df": pd.DataFrame(),
                "predictions_df": pd.DataFrame(),
                "teams_df": pd.DataFrame(),
            }

        match_ids = matches_df["id"].tolist()
        team_ids = sorted(set(matches_df["home_team_id"].tolist() + matches_df["away_team_id"].tolist()))

        try:
            predictions_response = (
                self._supabase.table("predictions")
                .select("id,match_id,market_type,selection,line_value,model_version,predicted_probability,fair_odds")
                .in_("match_id", match_ids)
                .execute()
            )
            teams_response = (
                self._supabase.table("teams_mapping")
                .select("id,canonical_team_name,api_football_team_name,odds_api_team_name")
                .in_("id", team_ids)
                .execute()
            )
        except Exception as exc:
            raise DatabaseConnectionError(
                "Failed to fetch predictions or teams_mapping rows from Supabase."
            ) from exc

        predictions_df = pd.DataFrame(predictions_response.data or [])
        teams_df = pd.DataFrame(teams_response.data or [])

        if not matches_df.empty:
            matches_df["match_date"] = pd.to_datetime(matches_df["match_date"], utc=True, errors="coerce")
        if not teams_df.empty and not matches_df.empty:
            teams_df = teams_df.copy()
            teams_df["display_name"] = teams_df.apply(
                lambda row: next(
                    (
                        str(value)
                        for value in [
                            row.get("canonical_team_name"),
                            row.get("api_football_team_name"),
                            row.get("odds_api_team_name"),
                        ]
                        if isinstance(value, str) and value.strip()
                    ),
                    "",
                ),
                axis=1,
            )
            team_name_map = teams_df.set_index("id")["display_name"].to_dict()
            matches_df["home_team_name"] = matches_df["home_team_id"].map(team_name_map)
            matches_df["away_team_name"] = matches_df["away_team_id"].map(team_name_map)

        return {
            "matches_df": matches_df,
            "predictions_df": predictions_df,
            "teams_df": teams_df,
        }

    def _map_events_to_matches(
        self,
        config: OddsListenerConfig,
        matches_df: pd.DataFrame,
        teams_df: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        if matches_df.empty:
            return {}

        min_time = matches_df["match_date"].min()
        commence_time_from = (min_time - pd.Timedelta(hours=6)).isoformat().replace("+00:00", "Z")

        events, _headers = self._odds_client.get_events(
            sport_key=config.sport_key,
            commence_time_from=commence_time_from,
        )

        team_lookup = {
            row["id"]: {
                value
                for value in [
                    row.get("canonical_team_name"),
                    row.get("api_football_team_name"),
                    row.get("odds_api_team_name"),
                ]
                if isinstance(value, str) and value.strip()
            }
            for _, row in teams_df.iterrows()
        }

        normalized_match_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for _, match_row in matches_df.iterrows():
            home_names = team_lookup.get(match_row["home_team_id"], set())
            away_names = team_lookup.get(match_row["away_team_id"], set())
            match_candidates = [
                {
                    "match_id": match_row["id"],
                    "match_date": match_row["match_date"],
                    "fixture_id": match_row["api_football_fixture_id"],
                }
            ]
            for home_name in home_names:
                for away_name in away_names:
                    key = (
                        self._normalize_name(home_name),
                        self._normalize_name(away_name),
                    )
                    normalized_match_index.setdefault(key, []).extend(match_candidates)

        event_map: dict[str, dict[str, Any]] = {}
        for event in events:
            if not isinstance(event, dict):
                continue

            home_key = self._normalize_name(str(event.get("home_team", "")))
            away_key = self._normalize_name(str(event.get("away_team", "")))
            candidates = normalized_match_index.get((home_key, away_key), [])
            if not candidates:
                continue

            event_time = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
            if pd.isna(event_time):
                continue

            best_candidate = min(
                candidates,
                key=lambda candidate: abs((candidate["match_date"] - event_time).total_seconds()),
            )
            time_delta_hours = abs((best_candidate["match_date"] - event_time).total_seconds()) / 3600
            if time_delta_hours > 12:
                continue

            event_map[best_candidate["match_id"]] = event

        return event_map

    def _load_latest_odds_lookup(self, match_ids: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
        if not match_ids:
            return {}

        try:
            response = (
                self._supabase.table("odds_history")
                .select("id,match_id,sportsbook_key,market_type,selection,line_value,price_decimal,collected_at")
                .in_("match_id", match_ids)
                .order("collected_at", desc=True)
                .execute()
            )
        except Exception as exc:
            raise DatabaseConnectionError(
                "Failed to fetch odds_history rows from Supabase."
            ) from exc

        lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in response.data or []:
            key = (
                row["match_id"],
                row["sportsbook_key"],
                row["market_type"],
                row["selection"],
                self._normalize_line_value(row.get("line_value")),
            )
            if key not in lookup:
                lookup[key] = row

        return lookup

    @staticmethod
    def _group_predictions_by_match(predictions_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in predictions_df.to_dict(orient="records"):
            grouped.setdefault(row["match_id"], []).append(row)
        return grouped

    def _process_event_odds(
        self,
        event_odds: dict[str, Any],
        match_row: pd.Series,
        prediction_rows: list[dict[str, Any]],
        latest_odds_lookup: dict[tuple[Any, ...], dict[str, Any]],
        min_edge: float,
        min_ev: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        odds_rows: list[dict[str, Any]] = []
        value_bet_rows: list[dict[str, Any]] = []

        bookmakers = event_odds.get("bookmakers", [])
        prediction_groups = self._group_predictions_by_market_and_line(prediction_rows)

        for bookmaker in bookmakers:
            sportsbook_key = bookmaker.get("key")
            bookmaker_last_update = bookmaker.get("last_update")

            for market in bookmaker.get("markets", []):
                odds_api_market_key = market.get("key")
                for prediction_key, grouped_predictions in prediction_groups.items():
                    internal_market_type, line_value = prediction_key
                    if SUPPORTED_MARKET_MAP.get(internal_market_type) != odds_api_market_key:
                        continue

                    fair_prob_lookup = self._build_no_vig_lookup(
                        internal_market_type=internal_market_type,
                        market=market,
                        match_row=match_row,
                        line_value=line_value,
                    )
                    if not fair_prob_lookup:
                        continue

                    for prediction in grouped_predictions:
                        outcome = self._find_matching_outcome(
                            internal_market_type=internal_market_type,
                            selection=prediction["selection"],
                            market=market,
                            match_row=match_row,
                            line_value=line_value,
                        )
                        if outcome is None:
                            continue

                        price_decimal = self._coerce_price_to_decimal(outcome.get("price"))
                        if price_decimal is None:
                            continue

                        selection_key = self._selection_key_for_value_lookup(
                            internal_market_type=internal_market_type,
                            selection=prediction["selection"],
                            match_row=match_row,
                        )
                        implied_probability_after_margin = fair_prob_lookup.get(selection_key)
                        if implied_probability_after_margin is None:
                            continue

                        odds_row = {
                            "match_id": match_row["id"],
                            "sportsbook_key": sportsbook_key,
                            "market_type": prediction["market_type"],
                            "selection": prediction["selection"],
                            "line_value": self._normalize_line_value(line_value),
                            "price_decimal": price_decimal,
                            "bookmaker_last_update": bookmaker_last_update,
                            "source_event_id": event_odds.get("id"),
                            "raw_payload": {
                                "bookmaker": bookmaker,
                                "market": market,
                                "outcome": outcome,
                            },
                        }

                        if self._should_insert_odds_row(
                            odds_row=odds_row,
                            latest_odds_lookup=latest_odds_lookup,
                        ):
                            odds_rows.append(odds_row)

                        edge = float(prediction["predicted_probability"]) - float(implied_probability_after_margin)
                        expected_value = (float(prediction["predicted_probability"]) * price_decimal) - 1.0

                        if edge > min_edge and expected_value > min_ev:
                            value_bet_rows.append(
                                {
                                    "prediction_id": prediction["id"],
                                    "match_id": match_row["id"],
                                    "sportsbook_key": sportsbook_key,
                                    "market_type": prediction["market_type"],
                                    "selection": prediction["selection"],
                                    "line_value": self._normalize_line_value(line_value),
                                    "model_version": prediction["model_version"],
                                    "predicted_probability": float(prediction["predicted_probability"]),
                                    "fair_odds": float(prediction["fair_odds"]) if prediction.get("fair_odds") is not None else round(1 / float(prediction["predicted_probability"]), 4),
                                    "offered_odds": price_decimal,
                                    "implied_probability_after_margin": float(implied_probability_after_margin),
                                    "edge": edge,
                                    "expected_value": expected_value,
                                }
                            )

        return odds_rows, value_bet_rows

    @staticmethod
    def _group_predictions_by_market_and_line(
        prediction_rows: list[dict[str, Any]],
    ) -> dict[tuple[str, float | None], list[dict[str, Any]]]:
        grouped: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
        for prediction in prediction_rows:
            key = (
                prediction["market_type"],
                OddsListener._normalize_line_value(prediction.get("line_value")),
            )
            grouped.setdefault(key, []).append(prediction)
        return grouped

    def _build_no_vig_lookup(
        self,
        internal_market_type: str,
        market: dict[str, Any],
        match_row: pd.Series,
        line_value: float | None,
    ) -> dict[str, float]:
        relevant_outcomes = [
            outcome
            for outcome in market.get("outcomes", [])
            if self._line_matches(line_value, outcome.get("point"))
        ]
        if len(relevant_outcomes) < 2:
            return {}

        priced_outcomes: list[tuple[str, float]] = []
        for outcome in relevant_outcomes:
            selection_key = self._derive_selection_key(
                internal_market_type=internal_market_type,
                outcome=outcome,
                match_row=match_row,
            )
            price_decimal = self._coerce_price_to_decimal(outcome.get("price"))
            if selection_key is None or price_decimal is None or price_decimal <= 1:
                continue

            priced_outcomes.append((selection_key, 1.0 / price_decimal))

        if len(priced_outcomes) < 2:
            return {}

        if internal_market_type in {"team_totals", "team_shots_on_target"}:
            grouped: dict[str, list[tuple[str, float]]] = {}
            for key, prob in priced_outcomes:
                pair_key = key.rsplit("_", 1)[0]
                grouped.setdefault(pair_key, []).append((key, prob))

            normalized: dict[str, float] = {}
            for items in grouped.values():
                total = sum(prob for _, prob in items)
                if total <= 0:
                    continue
                for key, prob in items:
                    normalized[key] = prob / total
            return normalized

        if internal_market_type == "player_shots_on_target":
            grouped = {}
            for key, prob in priced_outcomes:
                pair_key = key.rsplit("_", 1)[0]
                grouped.setdefault(pair_key, []).append((key, prob))

            normalized = {}
            for items in grouped.values():
                total = sum(prob for _, prob in items)
                if total <= 0:
                    continue
                for key, prob in items:
                    normalized[key] = prob / total
            return normalized

        total = sum(prob for _, prob in priced_outcomes)
        if total <= 0:
            return {}

        return {key: prob / total for key, prob in priced_outcomes}

    def _find_matching_outcome(
        self,
        internal_market_type: str,
        selection: str,
        market: dict[str, Any],
        match_row: pd.Series,
        line_value: float | None,
    ) -> dict[str, Any] | None:
        selection_key = self._selection_key_for_value_lookup(
            internal_market_type=internal_market_type,
            selection=selection,
            match_row=match_row,
        )

        for outcome in market.get("outcomes", []):
            if not self._line_matches(line_value, outcome.get("point")):
                continue

            derived_key = self._derive_selection_key(
                internal_market_type=internal_market_type,
                outcome=outcome,
                match_row=match_row,
            )
            if derived_key == selection_key:
                return outcome

        return None

    def _derive_selection_key(
        self,
        internal_market_type: str,
        outcome: dict[str, Any],
        match_row: pd.Series,
    ) -> str | None:
        name = str(outcome.get("name", "")).strip().lower()
        description = str(outcome.get("description", "")).strip().lower()
        home_team = self._normalize_name(str(match_row.get("home_team_name", "")))
        away_team = self._normalize_name(str(match_row.get("away_team_name", "")))

        if internal_market_type in {"match_totals", "match_corners", "match_cards"}:
            if name in {"over", "under"}:
                return name
            return None

        if internal_market_type in {"team_totals", "team_shots_on_target"}:
            target_side = None
            if self._normalize_name(description) == home_team:
                target_side = "home"
            elif self._normalize_name(description) == away_team:
                target_side = "away"
            elif self._normalize_name(name) == home_team:
                target_side = "home"
            elif self._normalize_name(name) == away_team:
                target_side = "away"

            if target_side and name in {"over", "under"}:
                return f"{target_side}_{name}"
            return None

        if internal_market_type == "player_shots_on_target":
            if name in {"over", "under"}:
                player_key = self._normalize_name(description)
                if player_key:
                    return f"{player_key}_{name}"
            return None

        return None

    def _selection_key_for_value_lookup(
        self,
        internal_market_type: str,
        selection: str,
        match_row: pd.Series,
    ) -> str:
        if internal_market_type != "player_shots_on_target":
            return selection.lower()

        return self._normalize_name(selection)

    def _should_insert_odds_row(
        self,
        odds_row: dict[str, Any],
        latest_odds_lookup: dict[tuple[Any, ...], dict[str, Any]],
    ) -> bool:
        key = self._odds_lookup_key(odds_row)
        previous_row = latest_odds_lookup.get(key)
        if previous_row is None:
            return True

        previous_price = self._normalize_line_value(previous_row.get("price_decimal"))
        current_price = self._normalize_line_value(odds_row.get("price_decimal"))
        return previous_price != current_price

    @staticmethod
    def _odds_lookup_key(odds_row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            odds_row["match_id"],
            odds_row["sportsbook_key"],
            odds_row["market_type"],
            odds_row["selection"],
            OddsListener._normalize_line_value(odds_row.get("line_value")),
        )

    def _insert_odds_history(self, odds_rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not odds_rows:
            return pd.DataFrame()

        try:
            response = self._supabase.table("odds_history").insert(odds_rows).execute()
        except Exception as exc:
            raise DatabaseConnectionError(
                "Failed to insert odds_history rows into Supabase."
            ) from exc

        return pd.DataFrame(response.data or [])

    def _insert_value_bets(
        self,
        value_bet_rows: list[dict[str, Any]],
        inserted_odds_df: pd.DataFrame,
    ) -> int:
        if not value_bet_rows or inserted_odds_df.empty:
            return 0

        odds_history_id_lookup = {
            self._odds_lookup_key(row): row["id"]
            for row in inserted_odds_df.to_dict(orient="records")
        }

        rows_to_insert: list[dict[str, Any]] = []
        for row in value_bet_rows:
            odds_lookup_key = (
                row["match_id"],
                row["sportsbook_key"],
                row["market_type"],
                row["selection"],
                self._normalize_line_value(row.get("line_value")),
            )
            odds_history_id = odds_history_id_lookup.get(odds_lookup_key)
            if odds_history_id is None:
                continue

            rows_to_insert.append(
                {
                    **row,
                    "odds_history_id": odds_history_id,
                    "stake_plan": {},
                    "alert_status": "pending",
                    "metadata": {},
                }
            )

        if not rows_to_insert:
            return 0

        try:
            self._supabase.table("value_bets_tracked").upsert(
                rows_to_insert,
                on_conflict="prediction_id,odds_history_id",
            ).execute()
        except Exception as exc:
            raise DatabaseConnectionError(
                "Failed to upsert value_bets_tracked rows into Supabase."
            ) from exc

        return len(rows_to_insert)

    @staticmethod
    def _coerce_price_to_decimal(value: Any) -> float | None:
        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if numeric_value <= 0:
            return None

        return numeric_value

    @staticmethod
    def _normalize_line_value(value: Any) -> float | None:
        if value is None or value == "":
            return None

        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _line_matches(cls, expected: float | None, candidate: Any) -> bool:
        normalized_expected = cls._normalize_line_value(expected)
        normalized_candidate = cls._normalize_line_value(candidate)
        return normalized_expected == normalized_candidate

    @staticmethod
    def _normalize_name(value: str) -> str:
        lowered = value.lower().strip()
        lowered = re.sub(r"[^a-z0-9]+", "", lowered)
        return lowered


def parse_args() -> OddsListenerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport-key", type=str, required=True)
    parser.add_argument("--regions", type=str, default="eu")
    parser.add_argument("--bookmakers", type=str, default="")
    parser.add_argument("--horizon-hours", type=int, default=36)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--min-ev", type=float, default=0.01)
    parser.add_argument("--mock-dir", type=str, default="")
    args = parser.parse_args()

    return OddsListenerConfig(
        sport_key=args.sport_key,
        regions=[item.strip() for item in args.regions.split(",") if item.strip()],
        bookmakers=[item.strip() for item in args.bookmakers.split(",") if item.strip()],
        horizon_hours=args.horizon_hours,
        min_edge=args.min_edge,
        min_ev=args.min_ev,
        mock_dir=Path(args.mock_dir) if args.mock_dir else None,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = parse_args()

    try:
        listener = OddsListener(
            odds_client=TheOddsAPIClient(mock_dir=config.mock_dir) if config.mock_dir else None
        )
        summary = listener.run_once(config=config)
    except (
        OddsListenerError,
        TheOddsAPIError,
        DatabaseConnectionError,
        SettingsError,
    ) as exc:
        LOGGER.exception("Odds listener failed: %s", exc)
        raise SystemExit(1) from exc

    print("Odds listener completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
