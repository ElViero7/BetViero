from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

from .base import LoLPipelineError
from .multimarket import (
    MultiMarketTrainingConfig,
    calculate_ev,
    calculate_fair_odd,
    calculate_implied_probability,
    calculate_kelly_fraction,
    estimate_over_under_probability,
    add_multimarket_rolling_features,
    prepare_training_team_frame,
)

MAJOR_LEAGUE_TOKENS = ("lck", "lpl", "lec", "lcs")
EXCLUDED_LEAGUE_TOKENS = (
    "challenger",
    "challengers",
    "academy",
    "prime league",
    "regional",
    "2nd division",
)
SUPPORTED_MARKETS = {"winner", "firstblood", "firsttower", "kills"}


@dataclass(frozen=True)
class BotConfig:
    pandascore_api_key: str
    pandascore_base_url: str
    pandascore_lookahead_hours: int
    telegram_bot_token: str
    telegram_chat_id: str
    supabase_url: str
    supabase_service_role_key: str
    supabase_schema: str
    model_artifact_path: Path
    historical_team_rows_path: Path
    predictions_output_path: Path
    team_name_overrides_path: Path
    bot_poll_interval_seconds: int
    bankroll_amount: float
    kelly_fraction: float
    max_stake_amount: float


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise LoLPipelineError(f"Missing required environment variable: {name}")
    return value


def load_config() -> BotConfig:
    load_dotenv()
    return BotConfig(
        pandascore_api_key=require_env("PANDASCORE_API_KEY"),
        pandascore_base_url=os.getenv("PANDASCORE_BASE_URL", "https://api.pandascore.co").rstrip("/"),
        pandascore_lookahead_hours=int(os.getenv("PANDASCORE_LOOKAHEAD_HOURS", "96")),
        telegram_bot_token=require_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=require_env("TELEGRAM_CHAT_ID"),
        supabase_url=require_env("SUPABASE_URL").rstrip("/"),
        supabase_service_role_key=require_env("SUPABASE_SERVICE_ROLE_KEY"),
        supabase_schema=os.getenv("SUPABASE_SCHEMA", "public").strip() or "public",
        model_artifact_path=Path(os.getenv("MODEL_ARTIFACT_PATH", "models/modelo_multimarket.pkl")),
        historical_team_rows_path=Path(
            os.getenv(
                "ORACLES_ELIXIR_CANONICAL_PARQUET_PATH",
                "data/processed/oracles_elixir_team_rows_training.parquet",
            )
        ),
        predictions_output_path=Path(
            os.getenv(
                "PREDICTIONS_INPUT_PATH",
                "data/predictions/upcoming_multimarket_predictions.csv",
            )
        ),
        team_name_overrides_path=Path(
            os.getenv("TEAM_NAME_OVERRIDES_PATH", "data/config/team_name_overrides.json")
        ),
        bot_poll_interval_seconds=int(os.getenv("BOT_POLL_INTERVAL_SECONDS", "5")),
        bankroll_amount=float(os.getenv("BANKROLL_AMOUNT", "26")),
        kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
        max_stake_amount=float(os.getenv("MAX_STAKE_AMOUNT", "1.0")),
    )


def normalize_team_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower()).strip()


def normalize_patch(value: str) -> str:
    text = str(value).strip()
    match = re.search(r"(\d{2}\.\d{2})", text)
    if match:
        return match.group(1)
    return text


def load_name_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {"pandascore": {}, "predictions": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("pandascore", {})
    payload.setdefault("predictions", {})
    return payload


def canonicalize_team_name(name: str, provider: str, overrides: dict[str, dict[str, str]]) -> str:
    normalized = normalize_team_name(name)
    mapped = overrides.get(provider, {}).get(normalized, name)
    return normalize_team_name(mapped)


def is_major_league(match_payload: dict[str, Any]) -> bool:
    texts = [
        str(match_payload.get("name", "")),
        str((match_payload.get("league") or {}).get("name", "")),
        str((match_payload.get("league") or {}).get("slug", "")),
        str((match_payload.get("serie") or {}).get("name", "")),
        str((match_payload.get("serie") or {}).get("slug", "")),
        str((match_payload.get("tournament") or {}).get("name", "")),
        str((match_payload.get("tournament") or {}).get("slug", "")),
    ]
    haystack = " ".join(texts).lower()
    if any(token in haystack for token in EXCLUDED_LEAGUE_TOKENS):
        return False
    return any(token in haystack for token in MAJOR_LEAGUE_TOKENS)


class HistoricalFeatureStore:
    def __init__(self, history_path: Path, overrides: dict[str, dict[str, str]]) -> None:
        if not history_path.exists():
            raise LoLPipelineError(f"Historical team rows file not found: {history_path}")

        if history_path.suffix.lower() in {".parquet", ".pq"}:
            raw_frame = duckdb.read_parquet(str(history_path)).df()
        else:
            raw_frame = pd.read_csv(history_path, low_memory=False)

        self.overrides = overrides
        self.team_frame = prepare_training_team_frame(raw_frame)
        self.engineered_frame = add_multimarket_rolling_features(
            self.team_frame,
            MultiMarketTrainingConfig(),
        )
        self.engineered_frame["normalized_team_name"] = self.engineered_frame["team_name"].apply(
            normalize_team_name
        )
        self.team_lookup = {
            row["normalized_team_name"]: row["team_name"]
            for _, row in self.engineered_frame[["normalized_team_name", "team_name"]]
            .drop_duplicates()
            .iterrows()
        }
        self.latest_by_team = (
            self.engineered_frame.sort_values(["match_date", "game_id"])
            .groupby("team_name", as_index=False)
            .tail(1)
            .set_index("team_name")
        )
        self.latest_by_team_side = (
            self.engineered_frame.sort_values(["match_date", "game_id"])
            .groupby(["team_name", "side"], as_index=False)
            .tail(1)
            .set_index(["team_name", "side"])
        )
        blue_rows = self.team_frame.loc[self.team_frame["side"] == "Blue"].copy()
        self.league_patch_priors = {
            "firstblood": blue_rows.groupby(["league", "patch"])["firstblood"].mean().to_dict(),
            "winner": blue_rows.groupby(["league", "patch"])["result"].mean().to_dict(),
            "firsttower": blue_rows.groupby(["league", "patch"])["firsttower"].mean().to_dict(),
            "kills": (
                self.team_frame.groupby(["league", "patch"])["kills"].mean().mul(2.0).to_dict()
            ),
        }
        self.league_priors = {
            "firstblood": blue_rows.groupby("league")["firstblood"].mean().to_dict(),
            "winner": blue_rows.groupby("league")["result"].mean().to_dict(),
            "firsttower": blue_rows.groupby("league")["firsttower"].mean().to_dict(),
            "kills": self.team_frame.groupby("league")["kills"].mean().mul(2.0).to_dict(),
        }
        self.global_priors = {
            "firstblood": float(blue_rows["firstblood"].mean()),
            "winner": float(blue_rows["result"].mean()),
            "firsttower": float(blue_rows["firsttower"].mean()),
            "kills": float(self.team_frame["kills"].mean() * 2.0),
        }
        latest_patch_rows = (
            self.team_frame.sort_values(["match_date", "game_id"])
            .groupby("league", as_index=False)
            .tail(1)
        )
        self.latest_patch_by_league = {
            row["league"]: str(row["patch"]) for _, row in latest_patch_rows.iterrows()
        }
        self.global_latest_patch = str(
            self.team_frame.sort_values(["match_date", "game_id"]).iloc[-1]["patch"]
        )
        self.default_stat_values = {
            "golddiffat15_roll5": float(self.latest_by_team["golddiffat15_roll5"].median()),
            "xpdiffat15_roll5": float(self.latest_by_team["xpdiffat15_roll5"].median()),
            "vspm_roll5": float(self.latest_by_team["vspm_roll5"].median()),
            "ckpm_roll5": float(self.latest_by_team["ckpm_roll5"].median()),
            "firstblood_roll5": float(self.latest_by_team["firstblood_roll5"].median()),
            "firsttower_roll5": float(self.latest_by_team["firsttower_roll5"].median()),
            "result_roll5": float(self.latest_by_team["result_roll5"].median()),
            "kills_roll5": float(self.latest_by_team["kills_roll5"].median()),
            "towers_roll5": float(self.latest_by_team["towers_roll5"].median()),
            "dragons_roll5": float(self.latest_by_team["dragons_roll5"].median()),
            "barons_roll5": float(self.latest_by_team["barons_roll5"].median()),
        }

    def resolve_team_name(self, external_name: str, provider: str) -> str | None:
        canonical = canonicalize_team_name(external_name, provider, self.overrides)
        return self.team_lookup.get(canonical)

    def resolve_patch(self, league: str, raw_patch: Any) -> str:
        patch = normalize_patch(str(raw_patch or ""))
        if patch:
            return patch
        return self.latest_patch_by_league.get(league, self.global_latest_patch)

    def league_patch_prior(self, league: str, patch: str, market: str) -> float:
        if (league, patch) in self.league_patch_priors[market]:
            return float(self.league_patch_priors[market][(league, patch)])
        if league in self.league_priors[market]:
            return float(self.league_priors[market][league])
        return self.global_priors[market]

    def team_context(self, team_name: str | None, side: str) -> dict[str, float]:
        if team_name is None or team_name not in self.latest_by_team.index:
            side_defaults = {
                "firstblood_side_roll5": self.global_priors["firstblood"],
                "firsttower_side_roll5": self.global_priors["firsttower"],
                "result_side_roll5": self.global_priors["winner"],
                "kills_side_roll5": self.global_priors["kills"] / 2.0,
                "towers_side_roll5": 6.0,
            }
            return {
                "games_played_pre": 0.0,
                "side_games_played_pre": 0.0,
                **self.default_stat_values,
                **side_defaults,
            }

        overall_row = self.latest_by_team.loc[team_name]
        side_key = (team_name, side)
        side_row = self.latest_by_team_side.loc[side_key] if side_key in self.latest_by_team_side.index else None

        def get_value(row: pd.Series, column: str, fallback: float) -> float:
            value = row.get(column, fallback)
            return fallback if pd.isna(value) else float(value)

        return {
            "games_played_pre": get_value(overall_row, "team_games_played_pre", 0.0),
            "side_games_played_pre": get_value(
                side_row if side_row is not None else overall_row,
                "team_side_games_played_pre",
                0.0,
            ),
            "golddiffat15_roll5": get_value(
                overall_row,
                "golddiffat15_roll5",
                self.default_stat_values["golddiffat15_roll5"],
            ),
            "xpdiffat15_roll5": get_value(
                overall_row,
                "xpdiffat15_roll5",
                self.default_stat_values["xpdiffat15_roll5"],
            ),
            "vspm_roll5": get_value(
                overall_row,
                "vspm_roll5",
                self.default_stat_values["vspm_roll5"],
            ),
            "ckpm_roll5": get_value(
                overall_row,
                "ckpm_roll5",
                self.default_stat_values["ckpm_roll5"],
            ),
            "firstblood_roll5": get_value(
                overall_row,
                "firstblood_roll5",
                self.default_stat_values["firstblood_roll5"],
            ),
            "firsttower_roll5": get_value(
                overall_row,
                "firsttower_roll5",
                self.default_stat_values["firsttower_roll5"],
            ),
            "result_roll5": get_value(
                overall_row,
                "result_roll5",
                self.default_stat_values["result_roll5"],
            ),
            "kills_roll5": get_value(
                overall_row,
                "kills_roll5",
                self.default_stat_values["kills_roll5"],
            ),
            "towers_roll5": get_value(
                overall_row,
                "towers_roll5",
                self.default_stat_values["towers_roll5"],
            ),
            "dragons_roll5": get_value(
                overall_row,
                "dragons_roll5",
                self.default_stat_values["dragons_roll5"],
            ),
            "barons_roll5": get_value(
                overall_row,
                "barons_roll5",
                self.default_stat_values["barons_roll5"],
            ),
            "firstblood_side_roll5": get_value(
                side_row if side_row is not None else overall_row,
                "firstblood_side_roll5",
                self.global_priors["firstblood"],
            ),
            "firsttower_side_roll5": get_value(
                side_row if side_row is not None else overall_row,
                "firsttower_side_roll5",
                self.global_priors["firsttower"],
            ),
            "result_side_roll5": get_value(
                side_row if side_row is not None else overall_row,
                "result_side_roll5",
                self.global_priors["winner"],
            ),
            "kills_side_roll5": get_value(
                side_row if side_row is not None else overall_row,
                "kills_side_roll5",
                self.global_priors["kills"] / 2.0,
            ),
            "towers_side_roll5": get_value(
                side_row if side_row is not None else overall_row,
                "towers_side_roll5",
                6.0,
            ),
        }


class FairOddsEngine:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.overrides = load_name_overrides(config.team_name_overrides_path)
        self.feature_store = HistoricalFeatureStore(config.historical_team_rows_path, self.overrides)
        with config.model_artifact_path.open("rb") as file_obj:
            self.artifact = pickle.load(file_obj)
        self.market_models = self.artifact["models"]

    def fetch_upcoming_matches(self) -> list[dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.config.pandascore_api_key}"}
        matches: list[dict[str, Any]] = []
        page = 1
        while page <= 5:
            response = requests.get(
                f"{self.config.pandascore_base_url}/lol/matches/upcoming",
                headers=headers,
                params={"page": page, "per_page": 100},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload:
                break
            matches.extend(payload)
            if len(payload) < 100:
                break
            page += 1
        return [match for match in matches if is_major_league(match)]

    def _extract_match_datetime(self, match_payload: dict[str, Any]) -> datetime | None:
        for key in ("begin_at", "scheduled_at", "original_scheduled_at"):
            raw_value = match_payload.get(key)
            if raw_value:
                return datetime.fromisoformat(str(raw_value).replace("Z", "+00:00"))
        return None

    def _extract_patch(self, match_payload: dict[str, Any], league_name: str) -> str:
        for key in ("videogame_version", "version", "patch"):
            raw_value = match_payload.get(key)
            if isinstance(raw_value, dict):
                for nested_key in ("name", "slug"):
                    if raw_value.get(nested_key):
                        return self.feature_store.resolve_patch(league_name, raw_value[nested_key])
            if raw_value:
                return self.feature_store.resolve_patch(league_name, raw_value)
        return self.feature_store.resolve_patch(league_name, "")

    def _build_matchup_row(
        self,
        blue_external_name: str,
        red_external_name: str,
        blue_has_firstpick: float,
        league_name: str,
        patch: str,
        match_name: str,
        scheduled_at: datetime | None,
        source_match_id: Any,
    ) -> dict[str, Any]:
        blue_history_name = self.feature_store.resolve_team_name(blue_external_name, "pandascore")
        red_history_name = self.feature_store.resolve_team_name(red_external_name, "pandascore")
        blue = self.feature_store.team_context(blue_history_name, "Blue")
        red = self.feature_store.team_context(red_history_name, "Red")
        return {
            "match_name": match_name,
            "source_match_id": str(source_match_id),
            "scheduled_at": scheduled_at.isoformat().replace("+00:00", "Z")
            if scheduled_at is not None
            else None,
            "league": league_name,
            "patch": patch,
            "blue_team": blue_external_name,
            "red_team": red_external_name,
            "blue_has_firstpick": float(blue_has_firstpick),
            "blue_games_played_pre": blue["games_played_pre"],
            "red_games_played_pre": red["games_played_pre"],
            "blue_side_games_played_pre": blue["side_games_played_pre"],
            "red_side_games_played_pre": red["side_games_played_pre"],
            "blue_golddiffat15_roll5": blue["golddiffat15_roll5"],
            "red_golddiffat15_roll5": red["golddiffat15_roll5"],
            "blue_xpdiffat15_roll5": blue["xpdiffat15_roll5"],
            "red_xpdiffat15_roll5": red["xpdiffat15_roll5"],
            "blue_vspm_roll5": blue["vspm_roll5"],
            "red_vspm_roll5": red["vspm_roll5"],
            "blue_ckpm_roll5": blue["ckpm_roll5"],
            "red_ckpm_roll5": red["ckpm_roll5"],
            "blue_firstblood_rate_roll5": blue["firstblood_roll5"],
            "red_firstblood_rate_roll5": red["firstblood_roll5"],
            "blue_firsttower_rate_roll5": blue["firsttower_roll5"],
            "red_firsttower_rate_roll5": red["firsttower_roll5"],
            "blue_win_rate_roll5": blue["result_roll5"],
            "red_win_rate_roll5": red["result_roll5"],
            "blue_kills_roll5": blue["kills_roll5"],
            "red_kills_roll5": red["kills_roll5"],
            "blue_towers_roll5": blue["towers_roll5"],
            "red_towers_roll5": red["towers_roll5"],
            "blue_dragons_roll5": blue["dragons_roll5"],
            "red_dragons_roll5": red["dragons_roll5"],
            "blue_barons_roll5": blue["barons_roll5"],
            "red_barons_roll5": red["barons_roll5"],
            "blue_firstblood_side_rate_roll5": blue["firstblood_side_roll5"],
            "red_firstblood_side_rate_roll5": red["firstblood_side_roll5"],
            "blue_firsttower_side_rate_roll5": blue["firsttower_side_roll5"],
            "red_firsttower_side_rate_roll5": red["firsttower_side_roll5"],
            "blue_win_side_rate_roll5": blue["result_side_roll5"],
            "red_win_side_rate_roll5": red["result_side_roll5"],
            "blue_kills_side_roll5": blue["kills_side_roll5"],
            "red_kills_side_roll5": red["kills_side_roll5"],
            "blue_towers_side_roll5": blue["towers_side_roll5"],
            "red_towers_side_roll5": red["towers_side_roll5"],
            "league_patch_blue_firstblood_rate_pre": self.feature_store.league_patch_prior(
                league_name,
                patch,
                "firstblood",
            ),
            "league_patch_blue_win_rate_pre": self.feature_store.league_patch_prior(
                league_name,
                patch,
                "winner",
            ),
            "league_patch_blue_firsttower_rate_pre": self.feature_store.league_patch_prior(
                league_name,
                patch,
                "firsttower",
            ),
            "league_patch_total_kills_mean_pre": self.feature_store.league_patch_prior(
                league_name,
                patch,
                "kills",
            ),
            "diff_golddiffat15_roll5": blue["golddiffat15_roll5"] - red["golddiffat15_roll5"],
            "diff_xpdiffat15_roll5": blue["xpdiffat15_roll5"] - red["xpdiffat15_roll5"],
            "diff_vspm_roll5": blue["vspm_roll5"] - red["vspm_roll5"],
            "diff_ckpm_roll5": blue["ckpm_roll5"] - red["ckpm_roll5"],
            "diff_firstblood_rate_roll5": blue["firstblood_roll5"] - red["firstblood_roll5"],
            "diff_firsttower_rate_roll5": blue["firsttower_roll5"] - red["firsttower_roll5"],
            "diff_win_rate_roll5": blue["result_roll5"] - red["result_roll5"],
            "diff_kills_roll5": blue["kills_roll5"] - red["kills_roll5"],
            "diff_towers_roll5": blue["towers_roll5"] - red["towers_roll5"],
            "diff_dragons_roll5": blue["dragons_roll5"] - red["dragons_roll5"],
            "diff_barons_roll5": blue["barons_roll5"] - red["barons_roll5"],
            "diff_firstblood_side_rate_roll5": blue["firstblood_side_roll5"]
            - red["firstblood_side_roll5"],
            "diff_firsttower_side_rate_roll5": blue["firsttower_side_roll5"]
            - red["firsttower_side_roll5"],
            "diff_win_side_rate_roll5": blue["result_side_roll5"] - red["result_side_roll5"],
            "diff_kills_side_roll5": blue["kills_side_roll5"] - red["kills_side_roll5"],
            "diff_towers_side_roll5": blue["towers_side_roll5"] - red["towers_side_roll5"],
        }

    def _predict_blue_probability(self, market: str, matchup_row: dict[str, Any]) -> float:
        market_artifact = self.market_models[market]
        frame = pd.DataFrame([matchup_row])
        feature_columns = market_artifact["feature_columns"]
        missing_columns = [column for column in feature_columns if column not in frame.columns]
        if missing_columns:
            missing_list = ", ".join(missing_columns)
            raise LoLPipelineError(f"Missing inference features for {market}: {missing_list}")
        probabilities = market_artifact["model"].predict_proba(frame[feature_columns])[:, 1]
        return float(probabilities[0])

    def _predict_total_kills(self, matchup_row: dict[str, Any]) -> float:
        market_artifact = self.market_models["kills"]
        frame = pd.DataFrame([matchup_row])
        feature_columns = market_artifact["feature_columns"]
        missing_columns = [column for column in feature_columns if column not in frame.columns]
        if missing_columns:
            missing_list = ", ".join(missing_columns)
            raise LoLPipelineError(f"Missing inference features for kills: {missing_list}")
        prediction = market_artifact["model"].predict(frame[feature_columns])
        return float(prediction[0])

    def score_upcoming_matches(self) -> pd.DataFrame:
        raw_matches = self.fetch_upcoming_matches()
        scored_rows: list[dict[str, Any]] = []
        horizon = datetime.now(UTC) + timedelta(hours=self.config.pandascore_lookahead_hours)

        for match_payload in raw_matches:
            opponents = match_payload.get("opponents") or []
            team_names = [
                str(opponent.get("opponent", {}).get("name", "")).strip()
                for opponent in opponents
                if opponent.get("opponent", {}).get("name")
            ]
            if len(team_names) != 2:
                continue

            scheduled_at = self._extract_match_datetime(match_payload)
            if scheduled_at is not None and scheduled_at > horizon:
                continue

            team_a, team_b = team_names[0], team_names[1]
            league_name = str((match_payload.get("league") or {}).get("name", "Unknown")).strip()
            patch = self._extract_patch(match_payload, league_name)
            match_name = f"{team_a} vs {team_b}"

            scenario_a_blue = self._build_matchup_row(
                blue_external_name=team_a,
                red_external_name=team_b,
                blue_has_firstpick=1.0,
                league_name=league_name,
                patch=patch,
                match_name=match_name,
                scheduled_at=scheduled_at,
                source_match_id=match_payload.get("id"),
            )
            scenario_a_blue_red_fp = self._build_matchup_row(
                blue_external_name=team_a,
                red_external_name=team_b,
                blue_has_firstpick=0.0,
                league_name=league_name,
                patch=patch,
                match_name=match_name,
                scheduled_at=scheduled_at,
                source_match_id=match_payload.get("id"),
            )
            scenario_b_blue = self._build_matchup_row(
                blue_external_name=team_b,
                red_external_name=team_a,
                blue_has_firstpick=1.0,
                league_name=league_name,
                patch=patch,
                match_name=match_name,
                scheduled_at=scheduled_at,
                source_match_id=match_payload.get("id"),
            )
            scenario_b_blue_red_fp = self._build_matchup_row(
                blue_external_name=team_b,
                red_external_name=team_a,
                blue_has_firstpick=0.0,
                league_name=league_name,
                patch=patch,
                match_name=match_name,
                scheduled_at=scheduled_at,
                source_match_id=match_payload.get("id"),
            )

            row_payload: dict[str, Any] = {
                "source_match_id": str(match_payload.get("id")),
                "match_name": match_name,
                "league": league_name,
                "patch": patch,
                "scheduled_at": scheduled_at.isoformat().replace("+00:00", "Z")
                if scheduled_at is not None
                else None,
                "team_a": team_a,
                "team_b": team_b,
                "side_assignment_method": "average_blue_red_scenarios",
            }

            for market in ["winner", "firstblood", "firsttower"]:
                team_a_if_blue = self._predict_blue_probability(market, scenario_a_blue)
                team_a_if_blue_red_fp = self._predict_blue_probability(market, scenario_a_blue_red_fp)
                team_a_if_red = 1.0 - self._predict_blue_probability(market, scenario_b_blue)
                team_a_if_red_red_fp = 1.0 - self._predict_blue_probability(
                    market,
                    scenario_b_blue_red_fp,
                )
                team_a_probability = (
                    team_a_if_blue
                    + team_a_if_blue_red_fp
                    + team_a_if_red
                    + team_a_if_red_red_fp
                ) / 4.0
                team_b_probability = 1.0 - team_a_probability
                row_payload[f"{market}_team_a_prob"] = team_a_probability
                row_payload[f"{market}_team_b_prob"] = team_b_probability
                row_payload[f"{market}_team_a_fair_odd"] = calculate_fair_odd(team_a_probability)
                row_payload[f"{market}_team_b_fair_odd"] = calculate_fair_odd(team_b_probability)

            kills_if_a_blue = self._predict_total_kills(scenario_a_blue)
            kills_if_a_blue_red_fp = self._predict_total_kills(scenario_a_blue_red_fp)
            kills_if_b_blue = self._predict_total_kills(scenario_b_blue)
            kills_if_b_blue_red_fp = self._predict_total_kills(scenario_b_blue_red_fp)
            row_payload["kills_prediction"] = (
                kills_if_a_blue
                + kills_if_a_blue_red_fp
                + kills_if_b_blue
                + kills_if_b_blue_red_fp
            ) / 4.0
            row_payload["kills_residual_std"] = float(
                self.market_models["kills"]["metrics"].get("residual_std", 6.0)
            )
            scored_rows.append(row_payload)

        output = pd.DataFrame(scored_rows)
        if not output.empty:
            output = output.sort_values(["scheduled_at", "league", "match_name"]).reset_index(drop=True)
            self.config.predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(self.config.predictions_output_path, index=False)
        return output

    def score_manual_matches(self, manual_matches: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"league", "team_a", "team_b"}
        missing_columns = required_columns.difference(manual_matches.columns)
        if missing_columns:
            raise LoLPipelineError(
                f"Manual matches file is missing columns: {', '.join(sorted(missing_columns))}"
            )

        scored_rows: list[dict[str, Any]] = []
        for _, row in manual_matches.iterrows():
            team_a = str(row["team_a"]).strip()
            team_b = str(row["team_b"]).strip()
            league_name = str(row["league"]).strip()
            if not team_a or not team_b or not league_name:
                continue

            scheduled_at_raw = row.get("scheduled_at")
            scheduled_at: datetime | None = None
            if pd.notna(scheduled_at_raw) and str(scheduled_at_raw).strip():
                scheduled_at = pd.to_datetime(scheduled_at_raw, utc=True, errors="coerce")
                if pd.notna(scheduled_at):
                    scheduled_at = scheduled_at.to_pydatetime()
                else:
                    scheduled_at = None

            patch = self.feature_store.resolve_patch(league_name, row.get("patch", ""))
            match_name = f"{team_a} vs {team_b}"
            scenario_a_blue = self._build_matchup_row(
                blue_external_name=team_a,
                red_external_name=team_b,
                blue_has_firstpick=1.0,
                league_name=league_name,
                patch=patch,
                match_name=match_name,
                scheduled_at=scheduled_at,
                source_match_id=row.get("source_match_id", f"manual::{match_name}"),
            )
            scenario_a_blue_red_fp = self._build_matchup_row(
                blue_external_name=team_a,
                red_external_name=team_b,
                blue_has_firstpick=0.0,
                league_name=league_name,
                patch=patch,
                match_name=match_name,
                scheduled_at=scheduled_at,
                source_match_id=row.get("source_match_id", f"manual::{match_name}"),
            )
            scenario_b_blue = self._build_matchup_row(
                blue_external_name=team_b,
                red_external_name=team_a,
                blue_has_firstpick=1.0,
                league_name=league_name,
                patch=patch,
                match_name=match_name,
                scheduled_at=scheduled_at,
                source_match_id=row.get("source_match_id", f"manual::{match_name}"),
            )
            scenario_b_blue_red_fp = self._build_matchup_row(
                blue_external_name=team_b,
                red_external_name=team_a,
                blue_has_firstpick=0.0,
                league_name=league_name,
                patch=patch,
                match_name=match_name,
                scheduled_at=scheduled_at,
                source_match_id=row.get("source_match_id", f"manual::{match_name}"),
            )

            row_payload: dict[str, Any] = {
                "source_match_id": str(row.get("source_match_id", f"manual::{match_name}")),
                "match_name": match_name,
                "league": league_name,
                "patch": patch,
                "scheduled_at": scheduled_at.isoformat().replace("+00:00", "Z")
                if scheduled_at is not None
                else None,
                "team_a": team_a,
                "team_b": team_b,
                "side_assignment_method": "average_blue_red_scenarios",
                "book_team_a_odd": pd.to_numeric(row.get("book_team_a_odd"), errors="coerce"),
                "book_team_b_odd": pd.to_numeric(row.get("book_team_b_odd"), errors="coerce"),
            }

            for market in ["winner", "firstblood", "firsttower"]:
                team_a_if_blue = self._predict_blue_probability(market, scenario_a_blue)
                team_a_if_blue_red_fp = self._predict_blue_probability(market, scenario_a_blue_red_fp)
                team_a_if_red = 1.0 - self._predict_blue_probability(market, scenario_b_blue)
                team_a_if_red_red_fp = 1.0 - self._predict_blue_probability(
                    market,
                    scenario_b_blue_red_fp,
                )
                team_a_probability = (
                    team_a_if_blue
                    + team_a_if_blue_red_fp
                    + team_a_if_red
                    + team_a_if_red_red_fp
                ) / 4.0
                team_b_probability = 1.0 - team_a_probability
                row_payload[f"{market}_team_a_prob"] = team_a_probability
                row_payload[f"{market}_team_b_prob"] = team_b_probability
                row_payload[f"{market}_team_a_fair_odd"] = calculate_fair_odd(team_a_probability)
                row_payload[f"{market}_team_b_fair_odd"] = calculate_fair_odd(team_b_probability)

            kills_if_a_blue = self._predict_total_kills(scenario_a_blue)
            kills_if_a_blue_red_fp = self._predict_total_kills(scenario_a_blue_red_fp)
            kills_if_b_blue = self._predict_total_kills(scenario_b_blue)
            kills_if_b_blue_red_fp = self._predict_total_kills(scenario_b_blue_red_fp)
            row_payload["kills_prediction"] = (
                kills_if_a_blue
                + kills_if_a_blue_red_fp
                + kills_if_b_blue
                + kills_if_b_blue_red_fp
            ) / 4.0
            row_payload["kills_residual_std"] = float(
                self.market_models["kills"]["metrics"].get("residual_std", 6.0)
            )
            scored_rows.append(row_payload)

        output = pd.DataFrame(scored_rows)
        if not output.empty:
            output = output.sort_values(["league", "scheduled_at", "match_name"]).reset_index(drop=True)
            self.config.predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(self.config.predictions_output_path, index=False)
        return output


def build_alert_message(match_row: pd.Series) -> str:
    team_a = str(match_row["team_a"])
    team_b = str(match_row["team_b"])
    league = str(match_row["league"])
    scheduled_at_text = ""
    scheduled_at = match_row.get("scheduled_at")
    if pd.notna(scheduled_at) and scheduled_at:
        scheduled_at_text = f"Hora UTC: {scheduled_at}\n"
    firstblood_line = ""
    if league != "LPL":
        firstblood_line = (
            f"First Blood: Equipo 1 @{float(match_row['firstblood_team_a_fair_odd']):.2f} | "
            f"Equipo 2 @{float(match_row['firstblood_team_b_fair_odd']):.2f}\n"
        )
    reply_examples = [
        "1 2.10 winner",
        "1 2.00 firsttower",
        "Over 29.5 1.84 kills",
        "kills",
        "28.5 1.68 2.00",
        "29.5 1.84 1.86",
        "30.5 2.00 1.70",
    ]
    if league != "LPL":
        reply_examples.insert(1, "2 1.85 firstblood")
    return (
        "ALERTA MAPA 1\n"
        f"Partido: {team_a} vs {team_b}\n"
        f"Liga: {league}\n"
        f"{scheduled_at_text}"
        f"Equipo 1: {team_a}\n"
        f"Equipo 2: {team_b}\n"
        "Cuotas justas modelo\n"
        f"Winner: Equipo 1 @{float(match_row['winner_team_a_fair_odd']):.2f} | "
        f"Equipo 2 @{float(match_row['winner_team_b_fair_odd']):.2f}\n"
        f"{firstblood_line}"
        f"First Tower: Equipo 1 @{float(match_row['firsttower_team_a_fair_odd']):.2f} | "
        f"Equipo 2 @{float(match_row['firsttower_team_b_fair_odd']):.2f}\n"
        f"Total Kills: {float(match_row['kills_prediction']):.1f}\n"
        "Responde con:\n"
        + "\n".join(reply_examples)
    )


def parse_match_name_from_alert(text: str) -> str | None:
    match = re.search(r"Partido:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_league_from_alert(text: str) -> str | None:
    match = re.search(r"Liga:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_team_slots_from_alert(text: str) -> dict[str, str]:
    matches = re.findall(r"Equipo\s+([12]):\s*(.+)$", text, flags=re.MULTILINE)
    return {slot.strip(): team.strip() for slot, team in matches}


def parse_kills_prediction_from_alert(text: str) -> float | None:
    match = re.search(r"Total Kills:\s*(\d+(?:\.\d+)?)$", text, flags=re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def parse_kills_residual_std_from_alert(text: str) -> float | None:
    return None


def load_prediction_row(config: BotConfig, match_name: str) -> pd.Series:
    if not config.predictions_output_path.exists():
        raise LoLPipelineError(
            f"No existe el snapshot de predicciones: {config.predictions_output_path}"
        )
    prediction_frame = pd.read_csv(config.predictions_output_path)
    matches = prediction_frame.loc[prediction_frame["match_name"].astype(str) == match_name]
    if matches.empty:
        raise LoLPipelineError(
            f"No encontre el partido `{match_name}` en el ultimo snapshot de predicciones."
        )
    return matches.iloc[0]


def parse_pick_message(
    text: str,
) -> dict[str, Any] | None:
    normalized = text.strip().replace(",", ".").replace("@", "")
    kills_match = re.fullmatch(
        r"(Over|Under)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(kills)",
        normalized,
        flags=re.IGNORECASE,
    )
    if kills_match:
        odd = float(kills_match.group(3))
        if odd <= 1.0:
            return None
        return {
            "market": "kills",
            "selection": kills_match.group(1).title(),
            "line": float(kills_match.group(2)),
            "my_odd": odd,
        }

    binary_match = re.fullmatch(
        r"(.+?)\s+(\d+(?:\.\d+)?)\s+(winner|firstblood|firsttower)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not binary_match:
        return None
    odd = float(binary_match.group(2))
    if odd <= 1.0:
        return None
    return {
        "market": binary_match.group(3).lower(),
        "selection": binary_match.group(1).strip(),
        "line": None,
        "my_odd": odd,
    }


def parse_kills_batch_message(text: str) -> list[dict[str, float]] | None:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    if lines[0].lower() != "kills":
        return None

    options: list[dict[str, float]] = []
    for line in lines[1:]:
        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)",
            line.replace(",", ".").replace("@", "").strip(),
        )
        if not match:
            raise LoLPipelineError(
                "Formato kills invalido. Usa `kills` y luego lineas tipo `29.5 1.84 1.86`."
            )
        options.append(
            {
                "line": float(match.group(1)),
                "over_odd": float(match.group(2)),
                "under_odd": float(match.group(3)),
            }
        )
    if not options:
        return None
    return options


def build_pick_payload_from_reply(
    config: BotConfig,
    reply_text: str,
    alert_text: str,
) -> dict[str, Any]:
    parsed_pick = parse_pick_message(reply_text)
    if parsed_pick is None:
        raise LoLPipelineError(
            "Formato invalido. Usa `Equipo cuota winner`, `Equipo cuota firstblood`, `Equipo cuota firsttower` o `Over linea cuota kills`."
        )

    match_name = parse_match_name_from_alert(alert_text)
    if not match_name:
        raise LoLPipelineError("No pude extraer el partido del mensaje original.")
    league = parse_league_from_alert(alert_text)
    market = parsed_pick["market"]
    my_odd = float(parsed_pick["my_odd"])

    if market not in SUPPORTED_MARKETS:
        raise LoLPipelineError(f"Mercado no soportado: {market}")
    if league == "LPL" and market == "firstblood":
        raise LoLPipelineError("First Blood queda desactivado para LPL en esta version.")

    model_probability: float
    model_fair_odd: float | None
    predicted_value: float | None = None

    if market == "kills":
        predicted_total = parse_kills_prediction_from_alert(alert_text)
        if predicted_total is None:
            raise LoLPipelineError("No pude extraer la prediccion de kills del mensaje original.")
        prediction_row = load_prediction_row(config, match_name)
        residual_std = float(prediction_row.get("kills_residual_std", 6.0))
        line = float(parsed_pick["line"])
        selection = str(parsed_pick["selection"])
        model_probability = estimate_over_under_probability(
            predicted_total=predicted_total,
            line=line,
            residual_std=residual_std,
            direction=selection,
        )
        model_fair_odd = calculate_fair_odd(model_probability)
        predicted_value = predicted_total
    else:
        prediction_row = load_prediction_row(config, match_name)
        team_slots = parse_team_slots_from_alert(alert_text)
        team_map = {
            str(prediction_row["team_a"]): {
                "probability": float(prediction_row[f"{market}_team_a_prob"]),
                "fair_odd": float(prediction_row[f"{market}_team_a_fair_odd"]),
            },
            str(prediction_row["team_b"]): {
                "probability": float(prediction_row[f"{market}_team_b_prob"]),
                "fair_odd": float(prediction_row[f"{market}_team_b_fair_odd"]),
            },
        }
        raw_selection = str(parsed_pick["selection"]).strip()
        slot_aliases = {
            "1": team_slots.get("1", str(prediction_row["team_a"])),
            "2": team_slots.get("2", str(prediction_row["team_b"])),
            "equipo 1": team_slots.get("1", str(prediction_row["team_a"])),
            "equipo1": team_slots.get("1", str(prediction_row["team_a"])),
            "team 1": team_slots.get("1", str(prediction_row["team_a"])),
            "team1": team_slots.get("1", str(prediction_row["team_a"])),
            "equipo 2": team_slots.get("2", str(prediction_row["team_b"])),
            "equipo2": team_slots.get("2", str(prediction_row["team_b"])),
            "team 2": team_slots.get("2", str(prediction_row["team_b"])),
            "team2": team_slots.get("2", str(prediction_row["team_b"])),
        }
        canonical_selection = slot_aliases.get(raw_selection.lower(), raw_selection)
        selection = next(
            (team for team in team_map if team.lower() == canonical_selection.lower()),
            None,
        )
        if selection is None:
            available = ", ".join(team_map.keys())
            raise LoLPipelineError(
                f"No pude identificar la seleccion. Equipos validos: {available}"
            )
        model_probability = float(team_map[selection]["probability"])
        model_fair_odd = float(team_map[selection]["fair_odd"])
        predicted_value = None

    implied_probability = calculate_implied_probability(my_odd)
    edge = model_probability - implied_probability
    ev = calculate_ev(model_probability, my_odd)
    full_kelly = calculate_kelly_fraction(model_probability, my_odd)
    applied_kelly = full_kelly * config.kelly_fraction
    recommended_stake = min(
        config.bankroll_amount * applied_kelly,
        config.max_stake_amount,
    )

    return {
        "match_name": match_name,
        "league": league,
        "market": market,
        "selection": selection,
        "line": parsed_pick["line"],
        "my_odd": my_odd,
        "model_probability": model_probability,
        "model_fair_odd": model_fair_odd,
        "predicted_value": predicted_value,
        "implied_probability": implied_probability,
        "edge": edge,
        "ev": ev,
        "full_kelly": full_kelly,
        "kelly_fraction": config.kelly_fraction,
        "recommended_stake": recommended_stake,
        "bankroll_snapshot": config.bankroll_amount,
        "status": "pending",
    }


def evaluate_kills_batch_from_reply(
    config: BotConfig,
    reply_text: str,
    alert_text: str,
) -> list[dict[str, Any]] | None:
    options = parse_kills_batch_message(reply_text)
    if options is None:
        return None

    match_name = parse_match_name_from_alert(alert_text)
    if not match_name:
        raise LoLPipelineError("No pude extraer el partido del mensaje original.")
    league = parse_league_from_alert(alert_text)
    predicted_total = parse_kills_prediction_from_alert(alert_text)
    if predicted_total is None:
        raise LoLPipelineError("No pude extraer la prediccion de kills del mensaje original.")
    prediction_row = load_prediction_row(config, match_name)
    residual_std = float(prediction_row.get("kills_residual_std", 6.0))

    evaluations: list[dict[str, Any]] = []
    for option in options:
        for direction, odd in [("Over", option["over_odd"]), ("Under", option["under_odd"])]:
            probability = estimate_over_under_probability(
                predicted_total=predicted_total,
                line=float(option["line"]),
                residual_std=residual_std,
                direction=direction,
            )
            fair_odd = calculate_fair_odd(probability)
            implied_probability = calculate_implied_probability(float(odd))
            edge = probability - implied_probability
            ev = calculate_ev(probability, float(odd))
            full_kelly = calculate_kelly_fraction(probability, float(odd))
            applied_kelly = full_kelly * config.kelly_fraction
            recommended_stake = min(
                config.bankroll_amount * applied_kelly,
                config.max_stake_amount,
            )
            evaluations.append(
                {
                    "match_name": match_name,
                    "league": league,
                    "market": "kills",
                    "selection": direction,
                    "line": float(option["line"]),
                    "my_odd": float(odd),
                    "model_probability": probability,
                    "model_fair_odd": fair_odd,
                    "predicted_value": predicted_total,
                    "implied_probability": implied_probability,
                    "edge": edge,
                    "ev": ev,
                    "full_kelly": full_kelly,
                    "kelly_fraction": config.kelly_fraction,
                    "recommended_stake": recommended_stake,
                    "bankroll_snapshot": config.bankroll_amount,
                    "status": "pending",
                }
            )

    evaluations.sort(
        key=lambda item: (
            item["recommended_stake"] > 0,
            item["ev"],
            item["edge"],
        ),
        reverse=True,
    )
    return evaluations


def insert_pick_to_supabase(config: BotConfig, payload: dict[str, Any]) -> None:
    url = f"{config.supabase_url}/rest/v1/picks"
    headers = {
        "apikey": config.supabase_service_role_key,
        "Authorization": f"Bearer {config.supabase_service_role_key}",
        "Content-Type": "application/json",
        "Accept-Profile": config.supabase_schema,
        "Content-Profile": config.supabase_schema,
        "Prefer": "return=minimal",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code >= 400:
        raise LoLPipelineError(
            f"Supabase insert failed ({response.status_code}): {response.text}"
        )
