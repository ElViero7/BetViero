from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.api_football import ApiFootballETL, ApiFootballETLError
from src.config import SettingsError
from src.db import DatabaseConnectionError, get_supabase_client


LOGGER = logging.getLogger(__name__)
MODELS_DIR = BASE_DIR / "models"
DEFAULT_ARTIFACT_PATH = MODELS_DIR / "xgboost_home_shots_on_target.pkl"
SCHEDULED_STATUSES = {"TBD", "NS", "PST"}
TEAM_STAT_COLUMNS = [
    "shots_on_goal",
    "shots_off_goal",
    "total_shots",
    "blocked_shots",
    "shots_inside_box",
    "shots_outside_box",
    "corner_kicks",
    "fouls",
    "yellow_cards",
    "red_cards",
    "goalkeeper_saves",
    "ball_possession_pct",
]


@dataclass(frozen=True)
class InferenceConfig:
    league_id: int
    season: int | None
    prediction_date: str
    rolling_window: int
    historical_matches: int
    line_value: float
    timezone: str
    artifact_path: Path
    mock_dir: Path | None = None


class DailyInferenceError(RuntimeError):
    """Raised when the daily inference pipeline fails."""


class DailyInferenceRunner:
    def __init__(self, etl: ApiFootballETL | None = None) -> None:
        self._etl = etl or ApiFootballETL()
        self._supabase = get_supabase_client()

    def run(self, config: InferenceConfig) -> dict[str, Any]:
        artifact = self._load_artifact(config.artifact_path)
        upcoming_df = self._fetch_upcoming_fixtures(config=config)
        historical_df = self._etl.fetch_last_league_matches(
            league_id=config.league_id,
            season=self._resolve_historical_season(upcoming_df=upcoming_df, config=config),
            last_n=config.historical_matches,
            timezone=config.timezone,
        )
        historical_stats_df = self._fetch_fixture_stats_batch(fixtures_df=historical_df)
        latest_team_features_df = self._build_latest_team_feature_snapshot(
            fixtures_df=historical_df,
            stats_df=historical_stats_df,
            rolling_window=config.rolling_window,
        )
        inference_df = self._build_inference_frame(
            upcoming_df=upcoming_df,
            latest_team_features_df=latest_team_features_df,
        )
        predictions_df = self._predict_fixture_markets(
            inference_df=inference_df,
            artifact=artifact,
            line_value=config.line_value,
            artifact_path=config.artifact_path,
        )
        persistence_summary = self._persist_predictions(predictions_df=predictions_df)

        return {
            "prediction_date": config.prediction_date,
            "fixtures_found": int(len(upcoming_df)),
            "predictions_saved": int(persistence_summary["predictions_saved"]),
            "model_version": persistence_summary["model_version"],
            "line_value": config.line_value,
        }

    def _load_artifact(self, artifact_path: Path) -> dict[str, Any]:
        if not artifact_path.exists():
            raise DailyInferenceError(f"Model artifact not found: {artifact_path}")

        with artifact_path.open("rb") as file_obj:
            artifact = pickle.load(file_obj)

        required_keys = {"model", "feature_columns", "target_column", "config"}
        missing_keys = [key for key in required_keys if key not in artifact]
        if missing_keys:
            raise DailyInferenceError(
                f"Model artifact is missing required keys: {missing_keys}"
            )

        return artifact

    def _fetch_upcoming_fixtures(self, config: InferenceConfig) -> pd.DataFrame:
        upcoming_df = self._etl.fetch_fixtures_by_date(
            fixture_date=config.prediction_date,
            league_id=config.league_id,
            season=config.season,
            timezone=config.timezone,
        )
        if upcoming_df.empty:
            raise DailyInferenceError(
                f"No fixtures returned for date={config.prediction_date} league_id={config.league_id}."
            )

        if "status_short" in upcoming_df.columns:
            upcoming_df = upcoming_df.loc[
                upcoming_df["status_short"].isin(SCHEDULED_STATUSES)
            ].copy()

        if upcoming_df.empty:
            raise DailyInferenceError(
                f"No scheduled fixtures found for date={config.prediction_date}."
            )

        return upcoming_df.sort_values("match_date").reset_index(drop=True)

    @staticmethod
    def _resolve_historical_season(upcoming_df: pd.DataFrame, config: InferenceConfig) -> int:
        if config.season is not None:
            return config.season

        if "season" not in upcoming_df.columns or upcoming_df["season"].dropna().empty:
            raise DailyInferenceError("Upcoming fixtures do not include a usable season value.")

        return int(upcoming_df["season"].dropna().mode().iloc[0])

    def _fetch_fixture_stats_batch(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        stats_frames: list[pd.DataFrame] = []

        for fixture_id in fixtures_df["fixture_id"].dropna().astype(int).tolist():
            fixture_stats = self._etl.fetch_fixture_statistics(fixture_id=fixture_id)
            if fixture_stats.empty:
                LOGGER.warning("No statistics returned for historical fixture_id=%s.", fixture_id)
                continue

            stats_frames.append(fixture_stats)

        if not stats_frames:
            raise DailyInferenceError(
                "No historical fixture statistics were retrieved for feature generation."
            )

        return pd.concat(stats_frames, ignore_index=True)

    def _build_latest_team_feature_snapshot(
        self,
        fixtures_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        rolling_window: int,
    ) -> pd.DataFrame:
        fixture_stats_df = self._build_fixture_stats_wide_frame(
            fixtures_df=fixtures_df,
            stats_df=stats_df,
        )
        team_match_df = self._build_team_match_frame(fixture_stats_df=fixture_stats_df)
        rolling_features_df = self._build_team_rolling_features(
            team_match_df=team_match_df,
            rolling_window=rolling_window,
        )

        rolling_feature_columns = [
            column for column in rolling_features_df.columns
            if "_avg_last_" in column
        ]
        if not rolling_feature_columns:
            raise DailyInferenceError("No rolling features were generated for inference.")

        latest_snapshot_df = (
            rolling_features_df.sort_values(["team_id", "match_date", "fixture_id"])
            .groupby("team_id", as_index=False)
            .tail(1)
            .loc[:, ["team_id", *rolling_feature_columns]]
            .reset_index(drop=True)
        )

        return latest_snapshot_df

    def _build_fixture_stats_wide_frame(
        self,
        fixtures_df: pd.DataFrame,
        stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        merged = stats_df.merge(
            fixtures_df[
                [
                    "fixture_id",
                    "match_date",
                    "league_id",
                    "league_name",
                    "season",
                    "home_team_id",
                    "home_team_name",
                    "away_team_id",
                    "away_team_name",
                    "home_goals",
                    "away_goals",
                ]
            ],
            on="fixture_id",
            how="left",
        )

        merged["team_side"] = pd.Series(pd.NA, index=merged.index, dtype="string")
        merged.loc[merged["team_id"] == merged["home_team_id"], "team_side"] = "home"
        merged.loc[merged["team_id"] == merged["away_team_id"], "team_side"] = "away"
        merged = merged.loc[merged["team_side"].isin(["home", "away"])].copy()

        value_columns = [
            column for column in TEAM_STAT_COLUMNS if column in merged.columns
        ]
        if not value_columns:
            raise DailyInferenceError("No supported statistic columns were returned by API-Football.")

        fixture_stats_wide = (
            merged.set_index(["fixture_id", "team_side"])[value_columns]
            .unstack("team_side")
        )
        fixture_stats_wide.columns = [
            f"{side}_{stat}" for stat, side in fixture_stats_wide.columns
        ]
        fixture_stats_wide = fixture_stats_wide.reset_index()

        output = fixtures_df.merge(fixture_stats_wide, on="fixture_id", how="inner")
        if output.empty:
            raise DailyInferenceError("Historical fixture statistics could not be aligned with fixtures.")

        return output

    def _build_team_match_frame(self, fixture_stats_df: pd.DataFrame) -> pd.DataFrame:
        home_frame = self._make_side_frame(fixture_stats_df=fixture_stats_df, side="home")
        away_frame = self._make_side_frame(fixture_stats_df=fixture_stats_df, side="away")
        team_match_df = pd.concat([home_frame, away_frame], ignore_index=True)
        return team_match_df.sort_values(["team_id", "match_date", "fixture_id"]).reset_index(drop=True)

    def _make_side_frame(self, fixture_stats_df: pd.DataFrame, side: str) -> pd.DataFrame:
        opponent_side = "away" if side == "home" else "home"

        rename_map = {
            f"{side}_team_id": "team_id",
            f"{opponent_side}_team_id": "opponent_team_id",
            f"{side}_goals": "goals_for",
            f"{opponent_side}_goals": "goals_against",
        }

        for stat in TEAM_STAT_COLUMNS:
            if f"{side}_{stat}" in fixture_stats_df.columns:
                rename_map[f"{side}_{stat}"] = stat
            if f"{opponent_side}_{stat}" in fixture_stats_df.columns:
                rename_map[f"{opponent_side}_{stat}"] = f"opp_{stat}"

        selected_columns = ["fixture_id", "match_date", *rename_map.keys()]
        selected_columns = [column for column in selected_columns if column in fixture_stats_df.columns]

        team_frame = fixture_stats_df[selected_columns].rename(columns=rename_map).copy()
        team_frame["team_side"] = side
        return team_frame

    def _build_team_rolling_features(
        self,
        team_match_df: pd.DataFrame,
        rolling_window: int,
    ) -> pd.DataFrame:
        feature_columns = [
            "shots_on_goal",
            "total_shots",
            "corner_kicks",
            "fouls",
            "yellow_cards",
            "ball_possession_pct",
            "goals_for",
            "goals_against",
            "opp_shots_on_goal",
            "opp_total_shots",
        ]
        present_feature_columns = [
            column for column in feature_columns if column in team_match_df.columns
        ]
        if not present_feature_columns:
            raise DailyInferenceError("No feature columns available for rolling inference features.")

        team_match_df = team_match_df.copy()
        grouped = team_match_df.groupby("team_id", group_keys=False)

        for column in present_feature_columns:
            feature_name = f"{column}_avg_last_{rolling_window}"
            team_match_df[feature_name] = grouped[column].transform(
                lambda series: series.shift(1).rolling(window=rolling_window, min_periods=1).mean()
            )

        return team_match_df

    def _build_inference_frame(
        self,
        upcoming_df: pd.DataFrame,
        latest_team_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rolling_feature_columns = [
            column for column in latest_team_features_df.columns
            if column != "team_id"
        ]

        home_snapshot_df = latest_team_features_df.rename(
            columns={
                "team_id": "home_team_id",
                **{column: f"home_{column}" for column in rolling_feature_columns},
            }
        )
        away_snapshot_df = latest_team_features_df.rename(
            columns={
                "team_id": "away_team_id",
                **{column: f"away_{column}" for column in rolling_feature_columns},
            }
        )

        inference_df = upcoming_df.merge(home_snapshot_df, on="home_team_id", how="left")
        inference_df = inference_df.merge(away_snapshot_df, on="away_team_id", how="left")

        return inference_df

    def _predict_fixture_markets(
        self,
        inference_df: pd.DataFrame,
        artifact: dict[str, Any],
        line_value: float,
        artifact_path: Path,
    ) -> pd.DataFrame:
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        rolling_window = int(artifact["config"]["rolling_window"])
        model_version = self._build_model_version(
            artifact=artifact,
            artifact_path=artifact_path,
        )

        required_columns = [
            column for column in feature_columns if column in inference_df.columns
        ]
        if len(required_columns) != len(feature_columns):
            missing = [column for column in feature_columns if column not in inference_df.columns]
            raise DailyInferenceError(
                f"Inference frame is missing feature columns required by the model: {missing}"
            )

        rows: list[dict[str, Any]] = []

        for side in ("home", "away"):
            feature_matrix = self._build_side_feature_matrix(
                inference_df=inference_df,
                feature_columns=feature_columns,
                side=side,
            )
            valid_mask = feature_matrix.notna().all(axis=1)
            if not valid_mask.any():
                LOGGER.warning("No complete feature rows available for side=%s.", side)
                continue

            predicted_lambda = model.predict(feature_matrix.loc[valid_mask, feature_columns])

            for row_index, lambda_value in zip(feature_matrix.index[valid_mask], predicted_lambda):
                fixture_row = inference_df.loc[row_index]
                expected_lambda = max(float(lambda_value), 0.01)
                probability = self._probability_over_line(
                    expected_lambda=expected_lambda,
                    line_value=line_value,
                )
                feature_payload = {
                    "expected_shots_on_target": expected_lambda,
                    "rolling_window": rolling_window,
                    "features": {
                        feature_name: self._json_safe_value(feature_matrix.at[row_index, feature_name])
                        for feature_name in feature_columns
                    },
                }

                rows.append(
                    {
                        "api_football_fixture_id": int(fixture_row["fixture_id"]),
                        "league_name": fixture_row.get("league_name"),
                        "season": int(fixture_row["season"]),
                        "round": fixture_row.get("round"),
                        "match_date": fixture_row["match_date"],
                        "home_team_api_id": int(fixture_row["home_team_id"]),
                        "home_team_name": fixture_row["home_team_name"],
                        "away_team_api_id": int(fixture_row["away_team_id"]),
                        "away_team_name": fixture_row["away_team_name"],
                        "venue_name": fixture_row.get("venue_name"),
                        "status": str(fixture_row.get("status_short", "NS")).lower(),
                        "market_type": "team_shots_on_target",
                        "selection": f"{side}_over",
                        "line_value": float(line_value),
                        "predicted_probability": probability,
                        "prediction_source": "daily_inference",
                        "model_version": model_version,
                        "feature_snapshot": feature_payload,
                        "model_inputs_hash": self._hash_payload(feature_payload),
                    }
                )

        if not rows:
            raise DailyInferenceError("No predictions could be generated from the upcoming fixtures.")

        return pd.DataFrame(rows)

    @staticmethod
    def _build_side_feature_matrix(
        inference_df: pd.DataFrame,
        feature_columns: list[str],
        side: str,
    ) -> pd.DataFrame:
        if side == "home":
            return inference_df.loc[:, feature_columns].copy()

        swapped_columns: dict[str, pd.Series] = {}
        for feature_name in feature_columns:
            if feature_name.startswith("home_"):
                source_column = f"away_{feature_name[5:]}"
            elif feature_name.startswith("away_"):
                source_column = f"home_{feature_name[5:]}"
            else:
                source_column = feature_name

            swapped_columns[feature_name] = inference_df[source_column]

        return pd.DataFrame(swapped_columns, index=inference_df.index)

    @staticmethod
    def _probability_over_line(expected_lambda: float, line_value: float) -> float:
        threshold = math.floor(line_value) + 1
        cumulative_probability = 0.0

        for goals in range(threshold):
            cumulative_probability += math.exp(-expected_lambda) * (expected_lambda ** goals) / math.factorial(goals)

        probability = 1.0 - cumulative_probability
        return min(max(probability, 1e-6), 0.999999)

    def _persist_predictions(self, predictions_df: pd.DataFrame) -> dict[str, Any]:
        predictions_records = predictions_df.to_dict(orient="records")
        team_lookup = self._upsert_teams(predictions_records=predictions_records)
        match_lookup = self._upsert_matches(
            predictions_records=predictions_records,
            team_lookup=team_lookup,
        )

        prediction_rows: list[dict[str, Any]] = []
        for record in predictions_records:
            fixture_id = record["api_football_fixture_id"]
            match_id = match_lookup[fixture_id]

            prediction_rows.append(
                {
                    "match_id": match_id,
                    "market_type": record["market_type"],
                    "selection": record["selection"],
                    "line_value": record["line_value"],
                    "model_version": record["model_version"],
                    "predicted_probability": record["predicted_probability"],
                    "prediction_source": record["prediction_source"],
                    "feature_snapshot": record["feature_snapshot"],
                    "model_inputs_hash": record["model_inputs_hash"],
                }
            )

        try:
            self._supabase.table("predictions").upsert(
                prediction_rows,
                on_conflict="match_id,market_type,selection,line_value,model_version",
            ).execute()
        except Exception as exc:
            raise DatabaseConnectionError(
                "Failed to upsert predictions into Supabase."
            ) from exc

        return {
            "predictions_saved": len(prediction_rows),
            "model_version": prediction_rows[0]["model_version"],
        }

    def _upsert_teams(self, predictions_records: list[dict[str, Any]]) -> dict[int, str]:
        team_payload_by_id: dict[int, dict[str, Any]] = {}

        for record in predictions_records:
            for team_id_key, team_name_key in (
                ("home_team_api_id", "home_team_name"),
                ("away_team_api_id", "away_team_name"),
            ):
                api_team_id = int(record[team_id_key])
                team_name = str(record[team_name_key])
                team_payload_by_id[api_team_id] = {
                    "canonical_team_name": team_name,
                    "canonical_team_slug": self._slugify(team_name, api_team_id),
                    "api_football_team_id": api_team_id,
                    "api_football_team_name": team_name,
                    "metadata": {},
                }

        payload = list(team_payload_by_id.values())

        try:
            self._supabase.table("teams_mapping").upsert(
                payload,
                on_conflict="api_football_team_id",
            ).execute()
            response = self._supabase.table("teams_mapping").select(
                "id,api_football_team_id"
            ).in_("api_football_team_id", list(team_payload_by_id.keys())).execute()
        except Exception as exc:
            raise DatabaseConnectionError(
                "Failed to upsert or fetch teams_mapping rows in Supabase."
            ) from exc

        return {
            int(row["api_football_team_id"]): row["id"]
            for row in response.data or []
        }

    def _upsert_matches(
        self,
        predictions_records: list[dict[str, Any]],
        team_lookup: dict[int, str],
    ) -> dict[int, str]:
        payload_by_fixture_id: dict[int, dict[str, Any]] = {}

        for record in predictions_records:
            fixture_id = int(record["api_football_fixture_id"])
            payload_by_fixture_id[fixture_id] = {
                "api_football_fixture_id": fixture_id,
                "competition_name": record["league_name"],
                "competition_stage": record["round"],
                "season": int(record["season"]),
                "match_date": self._to_iso_timestamp(record["match_date"]),
                "home_team_id": team_lookup[int(record["home_team_api_id"])],
                "away_team_id": team_lookup[int(record["away_team_api_id"])],
                "venue_name": record["venue_name"],
                "status": record["status"],
                "stats_snapshot": {},
                "metadata": {
                    "source": "api_football",
                    "api_football_fixture_id": fixture_id,
                },
            }

        payload = list(payload_by_fixture_id.values())

        try:
            self._supabase.table("matches").upsert(
                payload,
                on_conflict="api_football_fixture_id",
            ).execute()
            response = self._supabase.table("matches").select(
                "id,api_football_fixture_id"
            ).in_("api_football_fixture_id", list(payload_by_fixture_id.keys())).execute()
        except Exception as exc:
            raise DatabaseConnectionError(
                "Failed to upsert or fetch matches rows in Supabase."
            ) from exc

        return {
            int(row["api_football_fixture_id"]): row["id"]
            for row in response.data or []
        }

    @staticmethod
    def _build_model_version(artifact: dict[str, Any], artifact_path: Path) -> str:
        config_payload = {
            **artifact.get("config", {}),
            "artifact_name": artifact_path.name,
            "artifact_mtime_ns": artifact_path.stat().st_mtime_ns,
        }
        version_source = json.dumps(config_payload, sort_keys=True)
        version_hash = hashlib.md5(version_source.encode("utf-8")).hexdigest()[:10]
        return f"home_sot_xgb_{version_hash}"

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _slugify(team_name: str, team_id: int) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", team_name.lower()).strip("-")
        if not slug:
            slug = f"team-{team_id}"
        return slug

    @staticmethod
    def _to_iso_timestamp(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, datetime):
            return value.isoformat()

        return str(value)

    @staticmethod
    def _json_safe_value(value: Any) -> float | int | str | bool | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, float, str, bool)):
            return value
        return str(value)


def _default_prediction_date(timezone_name: str) -> str:
    current_dt = datetime.now(tz=ZoneInfo(timezone_name))
    return (current_dt + timedelta(days=1)).date().isoformat()


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-id", type=int, default=39)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--prediction-date", type=str, default=None)
    parser.add_argument("--rolling-window", type=int, default=3)
    parser.add_argument("--historical-matches", type=int, default=100)
    parser.add_argument("--line-value", type=float, default=3.5)
    parser.add_argument("--timezone", type=str, default="Europe/Madrid")
    parser.add_argument("--artifact-path", type=str, default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--mock-dir", type=str, default="")
    args = parser.parse_args()

    prediction_date = args.prediction_date or _default_prediction_date(args.timezone)

    return InferenceConfig(
        league_id=args.league_id,
        season=args.season,
        prediction_date=prediction_date,
        rolling_window=args.rolling_window,
        historical_matches=args.historical_matches,
        line_value=args.line_value,
        timezone=args.timezone,
        artifact_path=Path(args.artifact_path),
        mock_dir=Path(args.mock_dir) if args.mock_dir else None,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = parse_args()

    try:
        runner = DailyInferenceRunner(
            etl=ApiFootballETL(mock_dir=config.mock_dir) if config.mock_dir else None
        )
        summary = runner.run(config=config)
    except (
        ApiFootballETLError,
        DailyInferenceError,
        DatabaseConnectionError,
        SettingsError,
    ) as exc:
        LOGGER.exception("Daily inference pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    print("Daily inference completed.")
    print(f"Prediction date: {summary['prediction_date']}")
    print(f"Fixtures found: {summary['fixtures_found']}")
    print(f"Predictions saved: {summary['predictions_saved']}")
    print(f"Model version: {summary['model_version']}")
    print(f"Line value: {summary['line_value']}")


if __name__ == "__main__":
    main()
