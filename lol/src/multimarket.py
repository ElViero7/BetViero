from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .base import LoLPipelineError
from .firstblood import (
    LEAGUES_TO_INCLUDE,
    PARQUET_EXTENSIONS,
    SIDE_MAP,
    load_input_frame,
    normalize_binary,
    save_parquet_dataset,
)

TRAINING_RULE_LABEL = "2025 all games + 2026 map1 only"
TEAM_REQUIRED_COLUMNS = {
    "datacompleteness",
    "position",
    "game",
    "year",
    "side",
    "league",
    "patch",
    "firstpick",
    "golddiffat15",
    "xpdiffat15",
    "vspm",
    "ckpm",
    "firstblood",
    "firsttower",
    "result",
    "kills",
    "towers",
    "dragons",
    "barons",
}
GAME_ID_CANDIDATES = ["gameid", "game_id", "esportsgameid"]
TEAM_CANDIDATES = ["teamname", "team_name", "team"]
DATE_CANDIDATES = ["date", "gamedate", "game_date"]
ROLLING_COLUMNS = [
    "golddiffat15",
    "xpdiffat15",
    "vspm",
    "ckpm",
    "firstblood",
    "firsttower",
    "result",
    "kills",
    "towers",
    "dragons",
    "barons",
]
SIDE_AWARE_COLUMNS = ["firstblood", "firsttower", "result", "kills", "towers"]
BASE_CATEGORICAL_FEATURES = ["league", "patch"]
FEATURES_WINNER = [
    "league",
    "patch",
    "blue_has_firstpick",
    "blue_games_played_pre",
    "red_games_played_pre",
    "blue_side_games_played_pre",
    "red_side_games_played_pre",
    "blue_golddiffat15_roll5",
    "red_golddiffat15_roll5",
    "blue_xpdiffat15_roll5",
    "red_xpdiffat15_roll5",
    "blue_vspm_roll5",
    "red_vspm_roll5",
    "blue_ckpm_roll5",
    "red_ckpm_roll5",
    "blue_win_rate_roll5",
    "red_win_rate_roll5",
    "blue_firsttower_rate_roll5",
    "red_firsttower_rate_roll5",
    "blue_dragons_roll5",
    "red_dragons_roll5",
    "blue_barons_roll5",
    "red_barons_roll5",
    "blue_win_side_rate_roll5",
    "red_win_side_rate_roll5",
    "league_patch_blue_win_rate_pre",
    "diff_golddiffat15_roll5",
    "diff_xpdiffat15_roll5",
    "diff_vspm_roll5",
    "diff_ckpm_roll5",
    "diff_win_rate_roll5",
    "diff_firsttower_rate_roll5",
    "diff_dragons_roll5",
    "diff_barons_roll5",
    "diff_win_side_rate_roll5",
]
FEATURES_FIRSTBLOOD = [
    "league",
    "patch",
    "blue_has_firstpick",
    "blue_games_played_pre",
    "red_games_played_pre",
    "blue_side_games_played_pre",
    "red_side_games_played_pre",
    "blue_golddiffat15_roll5",
    "red_golddiffat15_roll5",
    "blue_xpdiffat15_roll5",
    "red_xpdiffat15_roll5",
    "blue_vspm_roll5",
    "red_vspm_roll5",
    "blue_ckpm_roll5",
    "red_ckpm_roll5",
    "blue_firstblood_rate_roll5",
    "red_firstblood_rate_roll5",
    "blue_firstblood_side_rate_roll5",
    "red_firstblood_side_rate_roll5",
    "league_patch_blue_firstblood_rate_pre",
    "diff_golddiffat15_roll5",
    "diff_xpdiffat15_roll5",
    "diff_vspm_roll5",
    "diff_ckpm_roll5",
    "diff_firstblood_rate_roll5",
    "diff_firstblood_side_rate_roll5",
]
FEATURES_FIRSTTOWER = [
    "league",
    "patch",
    "blue_has_firstpick",
    "blue_games_played_pre",
    "red_games_played_pre",
    "blue_side_games_played_pre",
    "red_side_games_played_pre",
    "blue_golddiffat15_roll5",
    "red_golddiffat15_roll5",
    "blue_xpdiffat15_roll5",
    "red_xpdiffat15_roll5",
    "blue_vspm_roll5",
    "red_vspm_roll5",
    "blue_ckpm_roll5",
    "red_ckpm_roll5",
    "blue_firsttower_rate_roll5",
    "red_firsttower_rate_roll5",
    "blue_towers_roll5",
    "red_towers_roll5",
    "blue_firsttower_side_rate_roll5",
    "red_firsttower_side_rate_roll5",
    "blue_towers_side_roll5",
    "red_towers_side_roll5",
    "blue_win_rate_roll5",
    "red_win_rate_roll5",
    "league_patch_blue_firsttower_rate_pre",
    "diff_golddiffat15_roll5",
    "diff_xpdiffat15_roll5",
    "diff_vspm_roll5",
    "diff_ckpm_roll5",
    "diff_firsttower_rate_roll5",
    "diff_towers_roll5",
    "diff_firsttower_side_rate_roll5",
    "diff_towers_side_roll5",
    "diff_win_rate_roll5",
]
FEATURES_KILLS = [
    "league",
    "patch",
    "blue_has_firstpick",
    "blue_games_played_pre",
    "red_games_played_pre",
    "blue_side_games_played_pre",
    "red_side_games_played_pre",
    "blue_vspm_roll5",
    "red_vspm_roll5",
    "blue_ckpm_roll5",
    "red_ckpm_roll5",
    "blue_kills_roll5",
    "red_kills_roll5",
    "blue_kills_side_roll5",
    "red_kills_side_roll5",
    "blue_firstblood_rate_roll5",
    "red_firstblood_rate_roll5",
    "blue_firsttower_rate_roll5",
    "red_firsttower_rate_roll5",
    "blue_dragons_roll5",
    "red_dragons_roll5",
    "blue_barons_roll5",
    "red_barons_roll5",
    "league_patch_total_kills_mean_pre",
    "diff_vspm_roll5",
    "diff_ckpm_roll5",
    "diff_kills_roll5",
    "diff_kills_side_roll5",
    "diff_firstblood_rate_roll5",
    "diff_firsttower_rate_roll5",
    "diff_dragons_roll5",
    "diff_barons_roll5",
]
MARKET_FEATURE_COLUMNS = {
    "winner": FEATURES_WINNER,
    "firstblood": FEATURES_FIRSTBLOOD,
    "firsttower": FEATURES_FIRSTTOWER,
    "kills": FEATURES_KILLS,
}


def is_allowed_completeness(league: pd.Series, datacompleteness: pd.Series) -> pd.Series:
    return (datacompleteness == "complete") | (
        (league == "LPL") & (datacompleteness == "partial")
    )


@dataclass(frozen=True)
class MultiMarketTrainingConfig:
    rolling_window: int = 5
    min_train_rows: int = 200
    test_size: float = 0.2
    random_state: int = 42


def resolve_column(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise LoLPipelineError(
        f"Could not find a {label} column. Checked: {', '.join(candidates)}"
    )


def validate_team_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    missing_columns = TEAM_REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise LoLPipelineError(f"Missing required Oracle's Elixir columns: {missing_list}")

    game_id_column = resolve_column(frame, GAME_ID_CANDIDATES, "game id")
    team_column = resolve_column(frame, TEAM_CANDIDATES, "team")
    date_column = resolve_column(frame, DATE_CANDIDATES, "date")
    return game_id_column, team_column, date_column


def normalize_result(series: pd.Series) -> pd.Series:
    return normalize_binary(series)


def filter_training_team_rows(raw_frame: pd.DataFrame) -> pd.DataFrame:
    game_id_column, team_column, date_column = validate_team_columns(raw_frame)

    frame = raw_frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame["side"] = frame["side"].astype(str).str.strip().str.lower().map(SIDE_MAP)
    frame["league"] = frame["league"].astype(str).str.strip()
    frame["patch"] = frame["patch"].astype(str).str.strip()
    frame["datacompleteness"] = frame["datacompleteness"].astype(str).str.strip().str.lower()
    frame["position"] = frame["position"].astype(str).str.strip().str.lower()
    frame[team_column] = frame[team_column].astype(str).str.strip()
    frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
    frame["game"] = pd.to_numeric(frame["game"], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")

    for binary_column in ["firstblood", "firsttower", "result"]:
        frame[binary_column] = normalize_binary(frame[binary_column])

    for numeric_column in [
        "firstpick",
        "golddiffat15",
        "xpdiffat15",
        "vspm",
        "ckpm",
        "kills",
        "towers",
        "dragons",
        "barons",
    ]:
        frame[numeric_column] = pd.to_numeric(frame[numeric_column], errors="coerce")

    filtered = frame.loc[
        is_allowed_completeness(frame["league"], frame["datacompleteness"])
        & (frame["position"] == "team")
        & (
            ((frame["year"] == 2025) & (frame["game"] >= 1))
            | ((frame["year"] >= 2026) & (frame["game"] == 1))
        )
        & (frame["league"].isin(LEAGUES_TO_INCLUDE))
        & (frame["side"].isin({"Blue", "Red"}))
    ].copy()

    filtered = filtered.rename(
        columns={
            game_id_column: "game_id",
            team_column: "team_name",
            date_column: "match_date",
        }
    )
    filtered = filtered.dropna(subset=["game_id", "match_date", "team_name"])
    filtered = filtered.sort_values(["team_name", "match_date", "game_id"]).reset_index(drop=True)
    return filtered


def prepare_training_team_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if {"game_id", "match_date", "team_name", "result", "kills", "towers", "firsttower"}.issubset(
        frame.columns
    ):
        clean_frame = frame.copy()
        if "firstpick" not in clean_frame.columns:
            clean_frame["firstpick"] = pd.NA
        clean_frame["side"] = clean_frame["side"].astype(str).str.strip()
        clean_frame["league"] = clean_frame["league"].astype(str).str.strip()
        clean_frame["patch"] = clean_frame["patch"].astype(str).str.strip()
        clean_frame["team_name"] = clean_frame["team_name"].astype(str).str.strip()
        clean_frame["match_date"] = pd.to_datetime(clean_frame["match_date"], utc=True, errors="coerce")
        clean_frame["year"] = pd.to_numeric(clean_frame["year"], errors="coerce")
        clean_frame["game"] = pd.to_numeric(clean_frame["game"], errors="coerce")
        for binary_column in ["firstblood", "firsttower", "result"]:
            clean_frame[binary_column] = normalize_binary(clean_frame[binary_column])
        for numeric_column in [
            "firstpick",
            "golddiffat15",
            "xpdiffat15",
            "vspm",
            "ckpm",
            "kills",
            "towers",
            "dragons",
            "barons",
        ]:
            clean_frame[numeric_column] = pd.to_numeric(clean_frame[numeric_column], errors="coerce")
        clean_frame = clean_frame.dropna(subset=["game_id", "match_date", "team_name"])
        return clean_frame.sort_values(["team_name", "match_date", "game_id"]).reset_index(drop=True)
    return filter_training_team_rows(frame)


def add_multimarket_rolling_features(
    team_frame: pd.DataFrame,
    config: MultiMarketTrainingConfig,
) -> pd.DataFrame:
    frame = team_frame.copy()
    frame["team_games_played_pre"] = frame.groupby("team_name").cumcount()
    frame["team_side_games_played_pre"] = frame.groupby(["team_name", "side"]).cumcount()

    for column in ROLLING_COLUMNS:
        rolling_column = f"{column}_roll{config.rolling_window}"
        frame[rolling_column] = (
            frame.groupby("team_name")[column]
            .transform(
                lambda series: series.shift(1).rolling(
                    window=config.rolling_window,
                    min_periods=1,
                ).mean()
            )
        )
        if column in SIDE_AWARE_COLUMNS:
            side_rolling_column = f"{column}_side_roll{config.rolling_window}"
            frame[side_rolling_column] = (
                frame.groupby(["team_name", "side"])[column]
                .transform(
                    lambda series: series.shift(1).rolling(
                        window=config.rolling_window,
                        min_periods=1,
                    ).mean()
                )
            )

    ordered = frame.sort_values(["match_date", "game_id", "side"]).reset_index(drop=True)
    ordered["league_patch_blue_firstblood_rate_pre"] = (
        ordered.groupby(["league", "patch"])["firstblood"]
        .transform(lambda series: series.shift(1).expanding(min_periods=1).mean())
    )
    ordered["league_patch_blue_win_rate_pre"] = (
        ordered.groupby(["league", "patch"])["result"]
        .transform(lambda series: series.shift(1).expanding(min_periods=1).mean())
    )
    ordered["league_patch_blue_firsttower_rate_pre"] = (
        ordered.groupby(["league", "patch"])["firsttower"]
        .transform(lambda series: series.shift(1).expanding(min_periods=1).mean())
    )
    ordered["league_patch_total_kills_mean_pre"] = (
        ordered.groupby(["league", "patch"])["kills"]
        .transform(lambda series: series.shift(1).expanding(min_periods=1).mean() * 2.0)
    )
    return ordered


def build_multimarket_matchup_frame(
    team_frame: pd.DataFrame,
    config: MultiMarketTrainingConfig,
) -> pd.DataFrame:
    suffix = f"roll{config.rolling_window}"
    blue = team_frame.loc[team_frame["side"] == "Blue"].copy()
    red = team_frame.loc[team_frame["side"] == "Red"].copy()

    keep_columns = [
        "game_id",
        "match_date",
        "year",
        "game",
        "league",
        "patch",
        "team_name",
        "firstpick",
        "firstblood",
        "firsttower",
        "result",
        "kills",
        "towers",
        "dragons",
        "barons",
        "team_games_played_pre",
        "team_side_games_played_pre",
        f"golddiffat15_{suffix}",
        f"xpdiffat15_{suffix}",
        f"vspm_{suffix}",
        f"ckpm_{suffix}",
        f"firstblood_{suffix}",
        f"firsttower_{suffix}",
        f"result_{suffix}",
        f"kills_{suffix}",
        f"towers_{suffix}",
        f"dragons_{suffix}",
        f"barons_{suffix}",
        f"firstblood_side_{suffix}",
        f"firsttower_side_{suffix}",
        f"result_side_{suffix}",
        f"kills_side_{suffix}",
        f"towers_side_{suffix}",
        "league_patch_blue_firstblood_rate_pre",
        "league_patch_blue_win_rate_pre",
        "league_patch_blue_firsttower_rate_pre",
        "league_patch_total_kills_mean_pre",
    ]

    blue = blue[keep_columns].rename(
        columns={
            "team_name": "blue_team",
            "firstpick": "blue_has_firstpick",
            "firstblood": "target_blue_firstblood",
            "firsttower": "target_blue_firsttower",
            "result": "target_blue_winner",
            "kills": "blue_kills",
            "towers": "blue_towers",
            "dragons": "blue_dragons",
            "barons": "blue_barons",
            "team_games_played_pre": "blue_games_played_pre",
            "team_side_games_played_pre": "blue_side_games_played_pre",
            f"golddiffat15_{suffix}": "blue_golddiffat15_roll5",
            f"xpdiffat15_{suffix}": "blue_xpdiffat15_roll5",
            f"vspm_{suffix}": "blue_vspm_roll5",
            f"ckpm_{suffix}": "blue_ckpm_roll5",
            f"firstblood_{suffix}": "blue_firstblood_rate_roll5",
            f"firsttower_{suffix}": "blue_firsttower_rate_roll5",
            f"result_{suffix}": "blue_win_rate_roll5",
            f"kills_{suffix}": "blue_kills_roll5",
            f"towers_{suffix}": "blue_towers_roll5",
            f"dragons_{suffix}": "blue_dragons_roll5",
            f"barons_{suffix}": "blue_barons_roll5",
            f"firstblood_side_{suffix}": "blue_firstblood_side_rate_roll5",
            f"firsttower_side_{suffix}": "blue_firsttower_side_rate_roll5",
            f"result_side_{suffix}": "blue_win_side_rate_roll5",
            f"kills_side_{suffix}": "blue_kills_side_roll5",
            f"towers_side_{suffix}": "blue_towers_side_roll5",
        }
    )
    red = red[keep_columns].rename(
        columns={
            "team_name": "red_team",
            "firstpick": "red_has_firstpick",
            "firstblood": "red_firstblood_current",
            "firsttower": "red_firsttower_current",
            "result": "red_win_current",
            "kills": "red_kills",
            "towers": "red_towers",
            "dragons": "red_dragons",
            "barons": "red_barons",
            "team_games_played_pre": "red_games_played_pre",
            "team_side_games_played_pre": "red_side_games_played_pre",
            f"golddiffat15_{suffix}": "red_golddiffat15_roll5",
            f"xpdiffat15_{suffix}": "red_xpdiffat15_roll5",
            f"vspm_{suffix}": "red_vspm_roll5",
            f"ckpm_{suffix}": "red_ckpm_roll5",
            f"firstblood_{suffix}": "red_firstblood_rate_roll5",
            f"firsttower_{suffix}": "red_firsttower_rate_roll5",
            f"result_{suffix}": "red_win_rate_roll5",
            f"kills_{suffix}": "red_kills_roll5",
            f"towers_{suffix}": "red_towers_roll5",
            f"dragons_{suffix}": "red_dragons_roll5",
            f"barons_{suffix}": "red_barons_roll5",
            f"firstblood_side_{suffix}": "red_firstblood_side_rate_roll5",
            f"firsttower_side_{suffix}": "red_firsttower_side_rate_roll5",
            f"result_side_{suffix}": "red_win_side_rate_roll5",
            f"kills_side_{suffix}": "red_kills_side_roll5",
            f"towers_side_{suffix}": "red_towers_side_roll5",
            "league_patch_blue_firstblood_rate_pre": "league_patch_blue_firstblood_rate_pre_red",
            "league_patch_blue_win_rate_pre": "league_patch_blue_win_rate_pre_red",
            "league_patch_blue_firsttower_rate_pre": "league_patch_blue_firsttower_rate_pre_red",
            "league_patch_total_kills_mean_pre": "league_patch_total_kills_mean_pre_red",
        }
    )

    matchup = blue.merge(
        red,
        on=["game_id", "match_date", "year", "game", "league", "patch"],
        how="inner",
        validate="one_to_one",
    )
    if matchup.empty:
        raise LoLPipelineError("No Blue vs Red matchups could be created from the filtered data.")

    matchup["target_blue_firstblood"] = pd.to_numeric(
        matchup["target_blue_firstblood"],
        errors="coerce",
    )
    matchup["target_blue_firsttower"] = pd.to_numeric(
        matchup["target_blue_firsttower"],
        errors="coerce",
    )
    matchup["target_blue_winner"] = pd.to_numeric(
        matchup["target_blue_winner"],
        errors="coerce",
    )
    matchup["target_total_kills"] = matchup["blue_kills"] + matchup["red_kills"]
    matchup["target_kill_diff"] = matchup["blue_kills"] - matchup["red_kills"]
    matchup["blue_has_firstpick"] = pd.to_numeric(
        matchup["blue_has_firstpick"],
        errors="coerce",
    )
    matchup["red_has_firstpick"] = pd.to_numeric(
        matchup["red_has_firstpick"],
        errors="coerce",
    )

    diff_pairs = [
        ("golddiffat15_roll5", "diff_golddiffat15_roll5"),
        ("xpdiffat15_roll5", "diff_xpdiffat15_roll5"),
        ("vspm_roll5", "diff_vspm_roll5"),
        ("ckpm_roll5", "diff_ckpm_roll5"),
        ("firstblood_rate_roll5", "diff_firstblood_rate_roll5"),
        ("firsttower_rate_roll5", "diff_firsttower_rate_roll5"),
        ("win_rate_roll5", "diff_win_rate_roll5"),
        ("kills_roll5", "diff_kills_roll5"),
        ("towers_roll5", "diff_towers_roll5"),
        ("dragons_roll5", "diff_dragons_roll5"),
        ("barons_roll5", "diff_barons_roll5"),
        ("firstblood_side_rate_roll5", "diff_firstblood_side_rate_roll5"),
        ("firsttower_side_rate_roll5", "diff_firsttower_side_rate_roll5"),
        ("win_side_rate_roll5", "diff_win_side_rate_roll5"),
        ("kills_side_roll5", "diff_kills_side_roll5"),
        ("towers_side_roll5", "diff_towers_side_roll5"),
    ]
    for stem, diff_name in diff_pairs:
        matchup[diff_name] = matchup[f"blue_{stem}"] - matchup[f"red_{stem}"]

    matchup = matchup.sort_values(["match_date", "game_id"]).reset_index(drop=True)
    return matchup


def build_market_feature_columns(market: str) -> tuple[list[str], list[str], list[str]]:
    if market not in MARKET_FEATURE_COLUMNS:
        raise LoLPipelineError(f"Unsupported market for feature selection: {market}")
    feature_columns = MARKET_FEATURE_COLUMNS[market]
    categorical_features = [column for column in feature_columns if column in BASE_CATEGORICAL_FEATURES]
    numeric_features = [column for column in feature_columns if column not in BASE_CATEGORICAL_FEATURES]
    return categorical_features, numeric_features, feature_columns


def build_classifier_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
    random_state: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=5000, random_state=random_state)),
        ]
    )


def build_xgb_classifier_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
    random_state: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=5,
                    gamma=1.0,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_regressor_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))])


def build_xgb_regressor_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
    random_state: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=160,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=5,
                    reg_alpha=0.25,
                    reg_lambda=2.0,
                    objective="reg:squarederror",
                    random_state=random_state,
                ),
            ),
        ]
    )


def split_train_test(
    frame: pd.DataFrame,
    config: MultiMarketTrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < config.min_train_rows:
        raise LoLPipelineError(
            f"Not enough matchup rows to train. Need at least {config.min_train_rows}, got {len(frame)}."
        )
    test_rows = max(1, int(len(frame) * config.test_size))
    train_frame = frame.iloc[:-test_rows].copy()
    test_frame = frame.iloc[-test_rows:].copy()
    if train_frame.empty or test_frame.empty:
        raise LoLPipelineError("Train/test split produced an empty partition.")
    return train_frame, test_frame


def _classification_metrics(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float | None]:
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "roc_auc": None,
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def _regression_metrics(y_true: pd.Series, predictions: pd.Series) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, predictions)))
    mae = float(mean_absolute_error(y_true, predictions))
    residual_std = float((y_true - predictions).std(ddof=1)) if len(y_true) > 1 else 1.0
    return {"rmse": rmse, "mae": mae, "residual_std": residual_std}


def train_binary_market_model(
    matchup_frame: pd.DataFrame,
    target_column: str,
    market: str,
    config: MultiMarketTrainingConfig,
) -> dict[str, Any]:
    matchup_frame = matchup_frame.dropna(subset=[target_column]).copy()
    matchup_frame[target_column] = matchup_frame[target_column].astype(int)
    train_frame, test_frame = split_train_test(matchup_frame, config)
    categorical_features, numeric_features, feature_columns = build_market_feature_columns(market)
    candidates: list[dict[str, Any]] = []

    for name, pipeline in [
        (
            "logreg",
            build_classifier_pipeline(categorical_features, numeric_features, config.random_state),
        ),
        (
            "xgboost",
            build_xgb_classifier_pipeline(categorical_features, numeric_features, config.random_state),
        ),
    ]:
        pipeline.fit(train_frame[feature_columns], train_frame[target_column])
        probabilities = pd.Series(
            pipeline.predict_proba(test_frame[feature_columns])[:, 1],
            index=test_frame.index,
        )
        candidates.append(
            {
                "name": name,
                "model": pipeline,
                "metrics": _classification_metrics(test_frame[target_column], probabilities),
            }
        )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["log_loss"],
            -(
                candidate["metrics"]["roc_auc"]
                if candidate["metrics"]["roc_auc"] is not None
                else float("-inf")
            ),
        ),
    )
    champion = ranked_candidates[0]
    return {
        "model": champion["model"],
        "feature_columns": feature_columns,
        "metrics": {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "selected_model_name": champion["name"],
            "candidate_metrics": {
                candidate["name"]: candidate["metrics"] for candidate in ranked_candidates
            },
            **champion["metrics"],
        },
    }


def train_regression_market_model(
    matchup_frame: pd.DataFrame,
    target_column: str,
    market: str,
    config: MultiMarketTrainingConfig,
) -> dict[str, Any]:
    train_frame, test_frame = split_train_test(matchup_frame, config)
    categorical_features, numeric_features, feature_columns = build_market_feature_columns(market)
    candidates: list[dict[str, Any]] = []

    for name, pipeline in [
        ("ridge", build_regressor_pipeline(categorical_features, numeric_features)),
        (
            "xgboost",
            build_xgb_regressor_pipeline(categorical_features, numeric_features, config.random_state),
        ),
    ]:
        pipeline.fit(train_frame[feature_columns], train_frame[target_column])
        predictions = pd.Series(
            pipeline.predict(test_frame[feature_columns]),
            index=test_frame.index,
        )
        candidates.append(
            {
                "name": name,
                "model": pipeline,
                "metrics": _regression_metrics(test_frame[target_column], predictions),
            }
        )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (candidate["metrics"]["mae"], candidate["metrics"]["rmse"]),
    )
    champion = ranked_candidates[0]
    return {
        "model": champion["model"],
        "feature_columns": feature_columns,
        "metrics": {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "selected_model_name": champion["name"],
            "candidate_metrics": {
                candidate["name"]: candidate["metrics"] for candidate in ranked_candidates
            },
            **champion["metrics"],
        },
    }


def build_multimarket_training_dataset(
    input_path: Path,
    config: MultiMarketTrainingConfig | None = None,
    team_rows_output_path: Path | None = None,
    matchup_output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or MultiMarketTrainingConfig()
    raw_frame = load_input_frame(input_path)
    team_frame = prepare_training_team_frame(raw_frame)
    team_frame = add_multimarket_rolling_features(team_frame, config)
    matchup_frame = build_multimarket_matchup_frame(team_frame, config)
    if team_rows_output_path is not None:
        save_parquet_dataset(team_frame, team_rows_output_path)
    if matchup_output_path is not None:
        save_parquet_dataset(matchup_frame, matchup_output_path)
    return team_frame, matchup_frame


def save_multimarket_artifact(
    market_models: dict[str, dict[str, Any]],
    output_path: Path,
    config: MultiMarketTrainingConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "models": market_models,
        "config": asdict(config),
        "training_rule": TRAINING_RULE_LABEL,
        "prediction_scope": "map1_only",
    }
    with output_path.open("wb") as file_obj:
        pickle.dump(artifact, file_obj)


def train_multimarket_pipeline(
    input_path: Path,
    model_output_path: Path,
    team_rows_output_path: Path | None = None,
    matchup_output_path: Path | None = None,
    config: MultiMarketTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or MultiMarketTrainingConfig()
    team_frame, matchup_frame = build_multimarket_training_dataset(
        input_path=input_path,
        config=config,
        team_rows_output_path=team_rows_output_path,
        matchup_output_path=matchup_output_path,
    )
    market_models = {
        "winner": {
            "market_type": "classification",
            "target_column": "target_blue_winner",
            **train_binary_market_model(matchup_frame, "target_blue_winner", "winner", config),
        },
        "firstblood": {
            "market_type": "classification",
            "target_column": "target_blue_firstblood",
            **train_binary_market_model(
                matchup_frame,
                "target_blue_firstblood",
                "firstblood",
                config,
            ),
        },
        "firsttower": {
            "market_type": "classification",
            "target_column": "target_blue_firsttower",
            **train_binary_market_model(
                matchup_frame,
                "target_blue_firsttower",
                "firsttower",
                config,
            ),
        },
        "kills": {
            "market_type": "regression",
            "target_column": "target_total_kills",
            **train_regression_market_model(matchup_frame, "target_total_kills", "kills", config),
        },
    }
    save_multimarket_artifact(
        market_models=market_models,
        output_path=model_output_path,
        config=config,
    )
    return {
        "team_rows": int(len(team_frame)),
        "matchup_rows": int(len(matchup_frame)),
        "model_output_path": str(model_output_path),
        "team_rows_output_path": str(team_rows_output_path) if team_rows_output_path else None,
        "matchup_output_path": str(matchup_output_path) if matchup_output_path else None,
        "training_rule": TRAINING_RULE_LABEL,
        "market_metrics": {market: payload["metrics"] for market, payload in market_models.items()},
    }


def calculate_fair_odd(probability: float) -> float | None:
    if probability <= 0:
        return None
    return 1.0 / probability


def calculate_implied_probability(odd: float) -> float:
    if odd <= 1.0:
        raise LoLPipelineError("Odds must be greater than 1.0.")
    return 1.0 / odd


def calculate_ev(probability: float, odd: float) -> float:
    return probability * odd - 1.0


def calculate_kelly_fraction(probability: float, odd: float) -> float:
    edge = calculate_ev(probability, odd)
    if edge <= 0:
        return 0.0
    b = odd - 1.0
    q = 1.0 - probability
    kelly = ((b * probability) - q) / b
    return max(0.0, float(kelly))


def estimate_over_under_probability(
    predicted_total: float,
    line: float,
    residual_std: float,
    direction: str,
) -> float:
    std = max(float(residual_std), 1.0)
    dist = NormalDist(mu=predicted_total, sigma=std)
    if direction.lower() == "over":
        return float(1.0 - dist.cdf(line))
    if direction.lower() == "under":
        return float(dist.cdf(line))
    raise LoLPipelineError(f"Unsupported direction: {direction}")
