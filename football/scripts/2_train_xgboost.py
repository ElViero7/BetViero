from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.api_football import ApiFootballETL, ApiFootballETLError
from src.config import SettingsError


LOGGER = logging.getLogger(__name__)
MODELS_DIR = BASE_DIR / "models"
ARTIFACT_PATH = MODELS_DIR / "xgboost_home_shots_on_target.pkl"

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
class TrainingConfig:
    league_id: int
    season: int
    last_n_matches: int
    rolling_window: int
    target_column: str
    min_train_rows: int
    mock_dir: Path | None = None


class ModelTrainingError(RuntimeError):
    """Raised when the training pipeline fails."""


class XGBoostTrainer:
    def __init__(self, etl: ApiFootballETL | None = None) -> None:
        self._etl = etl or ApiFootballETL()
        self._min_train_rows = 20

    def train_and_save(self, config: TrainingConfig, artifact_path: Path = ARTIFACT_PATH) -> dict[str, Any]:
        self._min_train_rows = config.min_train_rows
        fixtures_df = self._etl.fetch_last_league_matches(
            league_id=config.league_id,
            season=config.season,
            last_n=config.last_n_matches,
        )
        stats_df = self._fetch_fixture_stats_batch(fixtures_df=fixtures_df)
        model_df = self._build_training_frame(
            fixtures_df=fixtures_df,
            stats_df=stats_df,
            rolling_window=config.rolling_window,
        )
        trained = self._train_model(
            model_df=model_df,
            target_column=config.target_column,
        )
        self._save_artifact(
            artifact=trained,
            artifact_path=artifact_path,
            config=config,
        )
        return trained

    def _fetch_fixture_stats_batch(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        stats_frames: list[pd.DataFrame] = []

        for fixture_id in fixtures_df["fixture_id"].dropna().astype(int).tolist():
            fixture_stats = self._etl.fetch_fixture_statistics(fixture_id=fixture_id)
            if fixture_stats.empty:
                LOGGER.warning("No statistics returned for fixture_id=%s.", fixture_id)
                continue

            stats_frames.append(fixture_stats)

        if not stats_frames:
            raise ModelTrainingError(
                "No fixture statistics were retrieved. Training cannot continue."
            )

        stats_df = pd.concat(stats_frames, ignore_index=True)
        LOGGER.info("Fetched statistics for %s fixture-team rows.", len(stats_df))
        return stats_df

    def _build_training_frame(
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
        model_df = self._merge_rolling_features_into_fixtures(
            fixture_stats_df=fixture_stats_df,
            rolling_features_df=rolling_features_df,
        )

        suffix = f"_avg_last_{rolling_window}"
        required_columns = [
            "home_shots_on_goal",
            f"home_total_shots{suffix}",
            f"home_shots_on_goal{suffix}",
            f"away_shots_on_goal{suffix}",
        ]
        missing_columns = [
            column for column in required_columns if column not in model_df.columns
        ]
        if missing_columns:
            raise ModelTrainingError(
                f"Missing required training columns: {missing_columns}"
            )

        model_df = model_df.sort_values("match_date").reset_index(drop=True)
        return model_df

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
            raise ModelTrainingError("No supported statistic columns were returned by API-Football.")

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
            raise ModelTrainingError(
                "Fixture statistics could not be aligned with the fixture frame."
            )

        return output

    def _build_team_match_frame(self, fixture_stats_df: pd.DataFrame) -> pd.DataFrame:
        home_frame = self._make_side_frame(fixture_stats_df=fixture_stats_df, side="home")
        away_frame = self._make_side_frame(fixture_stats_df=fixture_stats_df, side="away")
        team_match_df = pd.concat([home_frame, away_frame], ignore_index=True)
        team_match_df = team_match_df.sort_values(["team_id", "match_date", "fixture_id"]).reset_index(drop=True)
        return team_match_df

    def _make_side_frame(self, fixture_stats_df: pd.DataFrame, side: str) -> pd.DataFrame:
        opponent_side = "away" if side == "home" else "home"

        rename_map = {
            f"{side}_team_id": "team_id",
            f"{side}_team_name": "team_name",
            f"{opponent_side}_team_id": "opponent_team_id",
            f"{opponent_side}_team_name": "opponent_team_name",
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
            raise ModelTrainingError("No feature columns available for rolling feature engineering.")

        team_match_df = team_match_df.copy()
        grouped = team_match_df.groupby("team_id", group_keys=False)

        for column in present_feature_columns:
            feature_name = f"{column}_avg_last_{rolling_window}"
            team_match_df[feature_name] = grouped[column].transform(
                lambda series: series.shift(1).rolling(window=rolling_window, min_periods=1).mean()
            )

        return team_match_df

    def _merge_rolling_features_into_fixtures(
        self,
        fixture_stats_df: pd.DataFrame,
        rolling_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rolling_feature_columns = [
            column for column in rolling_features_df.columns
            if "_avg_last_" in column
        ]

        home_features = (
            rolling_features_df.loc[rolling_features_df["team_side"] == "home", ["fixture_id", *rolling_feature_columns]]
            .rename(columns={column: f"home_{column}" for column in rolling_feature_columns})
        )
        away_features = (
            rolling_features_df.loc[rolling_features_df["team_side"] == "away", ["fixture_id", *rolling_feature_columns]]
            .rename(columns={column: f"away_{column}" for column in rolling_feature_columns})
        )

        output = fixture_stats_df.merge(home_features, on="fixture_id", how="left")
        output = output.merge(away_features, on="fixture_id", how="left")
        return output

    def _train_model(
        self,
        model_df: pd.DataFrame,
        target_column: str,
    ) -> dict[str, Any]:
        if target_column not in model_df.columns:
            raise ModelTrainingError(f"Target column '{target_column}' does not exist.")

        feature_columns = [
            column for column in model_df.columns
            if (
                column.startswith("home_") or column.startswith("away_")
            ) and "_avg_last_" in column
        ]

        training_frame = model_df[["fixture_id", "match_date", target_column, *feature_columns]].copy()
        training_frame = training_frame.dropna(subset=[target_column, *feature_columns]).reset_index(drop=True)

        if len(training_frame) < self._min_train_rows:
            raise ModelTrainingError(
                "Not enough complete rows to train a model. Increase sample size or lower feature requirements."
            )

        split_index = max(int(len(training_frame) * 0.8), 1)
        train_df = training_frame.iloc[:split_index].copy()
        test_df = training_frame.iloc[split_index:].copy()
        if test_df.empty:
            raise ModelTrainingError("Test split is empty; increase the sample size.")

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(len(feature_columns)),
        }

        LOGGER.info("Training finished with metrics: %s", metrics)

        return {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "metrics": metrics,
            "training_frame": training_frame,
        }

    def _save_artifact(
        self,
        artifact: dict[str, Any],
        artifact_path: Path,
        config: TrainingConfig,
    ) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        serializable_artifact = {
            "model": artifact["model"],
            "feature_columns": artifact["feature_columns"],
            "target_column": artifact["target_column"],
            "metrics": artifact["metrics"],
            "config": {
                "league_id": config.league_id,
                "season": config.season,
                "last_n_matches": config.last_n_matches,
                "rolling_window": config.rolling_window,
            },
        }

        with artifact_path.open("wb") as file_obj:
            pickle.dump(serializable_artifact, file_obj)

        LOGGER.info("Model artifact saved to %s", artifact_path)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-id", type=int, default=39)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--last-n-matches", type=int, default=100)
    parser.add_argument("--rolling-window", type=int, default=3)
    parser.add_argument("--target-column", type=str, default="home_shots_on_goal")
    parser.add_argument("--min-train-rows", type=int, default=20)
    parser.add_argument("--mock-dir", type=str, default="")
    args = parser.parse_args()

    return TrainingConfig(
        league_id=args.league_id,
        season=args.season,
        last_n_matches=args.last_n_matches,
        rolling_window=args.rolling_window,
        target_column=args.target_column,
        min_train_rows=args.min_train_rows,
        mock_dir=Path(args.mock_dir) if args.mock_dir else None,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = parse_args()

    try:
        trainer = XGBoostTrainer(
            etl=ApiFootballETL(mock_dir=config.mock_dir) if config.mock_dir else None
        )
        trained = trainer.train_and_save(config=config)
    except (ApiFootballETLError, ModelTrainingError, SettingsError) as exc:
        LOGGER.exception("Training pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    print("Training completed.")
    print(f"Target: {trained['target_column']}")
    print(f"Metrics: {trained['metrics']}")
    print(f"Artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
