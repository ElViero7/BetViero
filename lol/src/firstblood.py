from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .base import LoLPipelineError

LEAGUES_TO_INCLUDE = {"LCK", "LPL", "LEC", "LCS", "MSI", "WLDs"}
ROLLING_FEATURE_COLUMNS = [
    "golddiffat15",
    "xpdiffat15",
    "vspm",
    "ckpm",
    "firstblood",
]
REQUIRED_COLUMNS = {
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
}
CLEAN_TEAM_FRAME_COLUMNS = {
    "game_id",
    "match_date",
    "year",
    "game",
    "team_name",
    "side",
    "league",
    "patch",
    "firstpick",
    "golddiffat15",
    "xpdiffat15",
    "vspm",
    "ckpm",
    "firstblood",
}
GAME_ID_CANDIDATES = ["gameid", "game_id", "esportsgameid"]
TEAM_CANDIDATES = ["teamname", "team_name", "team"]
DATE_CANDIDATES = ["date", "gamedate", "game_date"]
SIDE_MAP = {
    "blue": "Blue",
    "red": "Red",
}
PARQUET_EXTENSIONS = {".parquet", ".pq"}


@dataclass(frozen=True)
class TrainingConfig:
    rolling_window: int = 5
    min_train_rows: int = 50
    test_size: float = 0.2
    random_state: int = 42


def load_oracles_elixir_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    if not csv_paths:
        raise LoLPipelineError("No CSV files were provided.")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise LoLPipelineError(f"CSV file not found: {csv_path}")
        frames.append(pd.read_csv(csv_path, low_memory=False))

    raw_frame = pd.concat(frames, ignore_index=True)
    raw_frame.columns = [str(column).strip().lower() for column in raw_frame.columns]
    return raw_frame


def load_parquet_frames(parquet_paths: list[Path]) -> pd.DataFrame:
    if not parquet_paths:
        raise LoLPipelineError("No Parquet files were provided.")

    frames: list[pd.DataFrame] = []
    for parquet_path in parquet_paths:
        if not parquet_path.exists():
            raise LoLPipelineError(f"Parquet file not found: {parquet_path}")
        frames.append(duckdb.read_parquet(str(parquet_path)).df())

    raw_frame = pd.concat(frames, ignore_index=True)
    raw_frame.columns = [str(column).strip().lower() for column in raw_frame.columns]
    return raw_frame


def resolve_column(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise LoLPipelineError(
        f"Could not find a {label} column. Checked: {', '.join(candidates)}"
    )


def validate_required_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise LoLPipelineError(f"Missing required Oracle's Elixir columns: {missing_list}")

    game_id_column = resolve_column(frame, GAME_ID_CANDIDATES, "game id")
    team_column = resolve_column(frame, TEAM_CANDIDATES, "team")
    date_column = resolve_column(frame, DATE_CANDIDATES, "date")
    return game_id_column, team_column, date_column


def normalize_binary(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"1": 1.0, "true": 1.0, "yes": 1.0, "0": 0.0, "false": 0.0, "no": 0.0})
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return normalized.fillna(numeric).astype(float)


def filter_team_rows(raw_frame: pd.DataFrame) -> pd.DataFrame:
    game_id_column, team_column, date_column = validate_required_columns(raw_frame)

    frame = raw_frame.copy()
    frame["side"] = (
        frame["side"].astype(str).str.strip().str.lower().map(SIDE_MAP)
    )
    frame["league"] = frame["league"].astype(str).str.strip()
    frame["patch"] = frame["patch"].astype(str).str.strip()
    frame["datacompleteness"] = frame["datacompleteness"].astype(str).str.strip().str.lower()
    frame["position"] = frame["position"].astype(str).str.strip().str.lower()
    frame[team_column] = frame[team_column].astype(str).str.strip()
    frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
    frame["game"] = pd.to_numeric(frame["game"], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["firstblood"] = normalize_binary(frame["firstblood"])

    numeric_columns = ["firstpick", "golddiffat15", "xpdiffat15", "vspm", "ckpm"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    filtered = frame.loc[
        (frame["datacompleteness"] == "complete")
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


def is_clean_team_frame(frame: pd.DataFrame) -> bool:
    return CLEAN_TEAM_FRAME_COLUMNS.issubset(frame.columns)


def prepare_team_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if is_clean_team_frame(frame):
        clean_frame = frame.copy()
        if "firstpick" not in clean_frame.columns:
            clean_frame["firstpick"] = pd.NA
        clean_frame["side"] = clean_frame["side"].astype(str).str.strip()
        clean_frame["league"] = clean_frame["league"].astype(str).str.strip()
        clean_frame["patch"] = clean_frame["patch"].astype(str).str.strip()
        clean_frame["team_name"] = clean_frame["team_name"].astype(str).str.strip()
        clean_frame["match_date"] = pd.to_datetime(
            clean_frame["match_date"],
            utc=True,
            errors="coerce",
        )
        clean_frame["year"] = pd.to_numeric(clean_frame["year"], errors="coerce")
        clean_frame["game"] = pd.to_numeric(clean_frame["game"], errors="coerce")
        clean_frame["firstblood"] = normalize_binary(clean_frame["firstblood"])
        for column in ["firstpick", "golddiffat15", "xpdiffat15", "vspm", "ckpm"]:
            clean_frame[column] = pd.to_numeric(clean_frame[column], errors="coerce")
        clean_frame = clean_frame.dropna(subset=["game_id", "match_date", "team_name"])
        return clean_frame.sort_values(["team_name", "match_date", "game_id"]).reset_index(drop=True)
    return filter_team_rows(frame)


def add_rolling_team_features(
    team_frame: pd.DataFrame, config: TrainingConfig
) -> pd.DataFrame:
    frame = team_frame.copy()
    frame["team_games_played_pre"] = frame.groupby("team_name").cumcount()
    frame["team_side_games_played_pre"] = frame.groupby(["team_name", "side"]).cumcount()

    for column in ROLLING_FEATURE_COLUMNS:
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
        if column == "firstblood":
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

    frame["firstblood_rate_roll5"] = frame[f"firstblood_roll{config.rolling_window}"]
    frame["firstblood_side_rate_roll5"] = frame[
        f"firstblood_side_roll{config.rolling_window}"
    ]
    frame = frame.drop(columns=[f"firstblood_roll{config.rolling_window}"])
    frame = frame.drop(columns=[f"firstblood_side_roll{config.rolling_window}"])

    ordered_frame = frame.sort_values(["match_date", "game_id", "side"]).reset_index(drop=True)
    ordered_frame["league_patch_blue_firstblood_rate_pre"] = (
        ordered_frame.groupby(["league", "patch"])["firstblood"]
        .transform(lambda series: series.shift(1).expanding(min_periods=1).mean())
    )
    return ordered_frame


def build_matchup_training_frame(team_frame: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
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
        "team_games_played_pre",
        "team_side_games_played_pre",
        f"golddiffat15_{suffix}",
        f"xpdiffat15_{suffix}",
        f"vspm_{suffix}",
        f"ckpm_{suffix}",
        "firstblood_rate_roll5",
        "firstblood_side_rate_roll5",
        "league_patch_blue_firstblood_rate_pre",
    ]
    blue = blue[keep_columns].rename(
        columns={
            "team_name": "blue_team",
            "firstpick": "blue_has_firstpick",
            "firstblood": "target_blue_firstblood",
            "team_games_played_pre": "blue_games_played_pre",
            "team_side_games_played_pre": "blue_side_games_played_pre",
            f"golddiffat15_{suffix}": "blue_golddiffat15_roll5",
            f"xpdiffat15_{suffix}": "blue_xpdiffat15_roll5",
            f"vspm_{suffix}": "blue_vspm_roll5",
            f"ckpm_{suffix}": "blue_ckpm_roll5",
            "firstblood_rate_roll5": "blue_firstblood_rate_roll5",
            "firstblood_side_rate_roll5": "blue_firstblood_side_rate_roll5",
            "league_patch_blue_firstblood_rate_pre": "league_patch_blue_firstblood_rate_pre",
        }
    )
    red = red[keep_columns].rename(
        columns={
            "team_name": "red_team",
            "firstpick": "red_has_firstpick",
            "firstblood": "red_firstblood_current",
            "team_games_played_pre": "red_games_played_pre",
            "team_side_games_played_pre": "red_side_games_played_pre",
            f"golddiffat15_{suffix}": "red_golddiffat15_roll5",
            f"xpdiffat15_{suffix}": "red_xpdiffat15_roll5",
            f"vspm_{suffix}": "red_vspm_roll5",
            f"ckpm_{suffix}": "red_ckpm_roll5",
            "firstblood_rate_roll5": "red_firstblood_rate_roll5",
            "firstblood_side_rate_roll5": "red_firstblood_side_rate_roll5",
            "league_patch_blue_firstblood_rate_pre": "league_patch_blue_firstblood_rate_pre_red",
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
    matchup["blue_has_firstpick"] = pd.to_numeric(
        matchup["blue_has_firstpick"],
        errors="coerce",
    )
    matchup["red_has_firstpick"] = pd.to_numeric(
        matchup["red_has_firstpick"],
        errors="coerce",
    )
    matchup["diff_golddiffat15_roll5"] = (
        matchup["blue_golddiffat15_roll5"] - matchup["red_golddiffat15_roll5"]
    )
    matchup["diff_xpdiffat15_roll5"] = (
        matchup["blue_xpdiffat15_roll5"] - matchup["red_xpdiffat15_roll5"]
    )
    matchup["diff_vspm_roll5"] = matchup["blue_vspm_roll5"] - matchup["red_vspm_roll5"]
    matchup["diff_ckpm_roll5"] = matchup["blue_ckpm_roll5"] - matchup["red_ckpm_roll5"]
    matchup["diff_firstblood_rate_roll5"] = (
        matchup["blue_firstblood_rate_roll5"] - matchup["red_firstblood_rate_roll5"]
    )
    matchup["diff_firstblood_side_rate_roll5"] = (
        matchup["blue_firstblood_side_rate_roll5"] - matchup["red_firstblood_side_rate_roll5"]
    )

    matchup = matchup.sort_values(["match_date", "game_id"]).reset_index(drop=True)
    return matchup


def build_feature_columns() -> tuple[list[str], list[str]]:
    categorical_features = ["league", "patch"]
    numeric_features = [
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
    return categorical_features, numeric_features


def build_compact_feature_columns() -> tuple[list[str], list[str]]:
    categorical_features = ["patch"]
    numeric_features = [
        "blue_has_firstpick",
        "diff_golddiffat15_roll5",
        "diff_xpdiffat15_roll5",
        "diff_vspm_roll5",
        "diff_ckpm_roll5",
        "diff_firstblood_rate_roll5",
        "league_patch_blue_firstblood_rate_pre",
    ]
    return categorical_features, numeric_features


def build_xgb_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
    config: TrainingConfig,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )
    model = XGBClassifier(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        gamma=1.0,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_compact_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    compact_preprocessor = ColumnTransformer(
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
            ("preprocessor", compact_preprocessor),
            ("model", LogisticRegression(max_iter=5000)),
        ]
    )


def evaluate_candidate_metrics(
    pipeline: Pipeline,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float | int | None]:
    probabilities = pipeline.predict_proba(test_frame[feature_columns])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, float | int | None] = {
        "accuracy": float(accuracy_score(test_frame["target_blue_firstblood"], predictions)),
        "log_loss": float(log_loss(test_frame["target_blue_firstblood"], probabilities)),
        "roc_auc": None,
    }
    if test_frame["target_blue_firstblood"].nunique() > 1:
        metrics["roc_auc"] = float(
            roc_auc_score(test_frame["target_blue_firstblood"], probabilities)
        )
    return metrics


def train_candidate_model(
    name: str,
    pipeline: Pipeline,
    feature_columns: list[str],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> dict[str, Any]:
    pipeline.fit(train_frame[feature_columns], train_frame["target_blue_firstblood"])
    metrics = evaluate_candidate_metrics(
        pipeline=pipeline,
        train_frame=train_frame,
        test_frame=test_frame,
        feature_columns=feature_columns,
    )
    return {
        "name": name,
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "metrics": metrics,
    }


def split_train_test(frame: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def train_firstblood_model(
    matchup_frame: pd.DataFrame, config: TrainingConfig
) -> tuple[Pipeline, dict[str, Any], list[str]]:
    matchup_frame = matchup_frame.dropna(subset=["target_blue_firstblood"]).copy()
    matchup_frame["target_blue_firstblood"] = matchup_frame["target_blue_firstblood"].astype(int)
    train_frame, test_frame = split_train_test(matchup_frame, config)
    xgb_categorical_features, xgb_numeric_features = build_feature_columns()
    xgb_feature_columns = xgb_categorical_features + xgb_numeric_features
    compact_categorical_features, compact_numeric_features = build_compact_feature_columns()
    compact_feature_columns = compact_categorical_features + compact_numeric_features

    candidates = [
        train_candidate_model(
            name="xgboost_full",
            pipeline=build_xgb_pipeline(xgb_categorical_features, xgb_numeric_features, config),
            feature_columns=xgb_feature_columns,
            train_frame=train_frame,
            test_frame=test_frame,
        ),
        train_candidate_model(
            name="logreg_compact",
            pipeline=build_compact_pipeline(
                compact_categorical_features,
                compact_numeric_features,
            ),
            feature_columns=compact_feature_columns,
            train_frame=train_frame,
            test_frame=test_frame,
        ),
    ]

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
    metrics: dict[str, Any] = {
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "selected_model_name": str(champion["name"]),
        "candidate_metrics": {
            candidate["name"]: candidate["metrics"] for candidate in ranked_candidates
        },
        "accuracy": champion["metrics"]["accuracy"],
        "log_loss": champion["metrics"]["log_loss"],
        "roc_auc": champion["metrics"]["roc_auc"],
    }
    return champion["pipeline"], metrics, champion["feature_columns"]


def save_training_dataset(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def save_parquet_dataset(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relation = duckdb.from_df(frame)
    relation.write_parquet(str(output_path))


def save_model_artifact(
    model: Pipeline,
    metrics: dict[str, Any],
    feature_columns: list[str],
    output_path: Path,
    config: TrainingConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "target_column": "target_blue_firstblood",
        "metrics": metrics,
        "config": asdict(config),
        "market": "first_blood_map1",
    }
    with output_path.open("wb") as file_obj:
        pickle.dump(artifact, file_obj)


def resolve_csv_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        csv_paths = sorted(input_path.rglob("*.csv"))
        if csv_paths:
            return csv_paths
    raise LoLPipelineError(f"No CSV files found under: {input_path}")


def resolve_parquet_paths(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in PARQUET_EXTENSIONS:
        return [input_path]
    if input_path.is_dir():
        parquet_paths = sorted(
            path for path in input_path.rglob("*") if path.suffix.lower() in PARQUET_EXTENSIONS
        )
        if parquet_paths:
            return parquet_paths
    raise LoLPipelineError(f"No Parquet files found under: {input_path}")


def load_input_frame(input_path: Path) -> pd.DataFrame:
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix == ".csv":
            return load_oracles_elixir_csvs([input_path])
        if suffix in PARQUET_EXTENSIONS:
            return load_parquet_frames([input_path])
    if input_path.is_dir():
        parquet_paths = sorted(
            path for path in input_path.rglob("*") if path.suffix.lower() in PARQUET_EXTENSIONS
        )
        if parquet_paths:
            return load_parquet_frames(parquet_paths)
        csv_paths = sorted(input_path.rglob("*.csv"))
        if csv_paths:
            return load_oracles_elixir_csvs(csv_paths)
    raise LoLPipelineError(
        f"Unsupported input path or no readable CSV/Parquet files found under: {input_path}"
    )


def build_training_dataset_from_csvs(
    input_path: Path,
    dataset_output_path: Path | None = None,
    dataset_parquet_output_path: Path | None = None,
    config: TrainingConfig | None = None,
) -> pd.DataFrame:
    config = config or TrainingConfig()
    raw_frame = load_input_frame(input_path)
    team_frame = prepare_team_frame(raw_frame)
    matchup_frame = build_matchup_training_frame(
        add_rolling_team_features(team_frame, config),
        config,
    )

    if dataset_output_path is not None:
        save_training_dataset(matchup_frame, dataset_output_path)
    if dataset_parquet_output_path is not None:
        save_parquet_dataset(matchup_frame, dataset_parquet_output_path)
    return matchup_frame


def run_training_pipeline(
    input_path: Path,
    dataset_output_path: Path,
    dataset_parquet_output_path: Path | None,
    model_output_path: Path,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or TrainingConfig()
    matchup_frame = build_training_dataset_from_csvs(
        input_path=input_path,
        dataset_output_path=dataset_output_path,
        dataset_parquet_output_path=dataset_parquet_output_path,
        config=config,
    )
    model, metrics, feature_columns = train_firstblood_model(matchup_frame, config)
    save_model_artifact(
        model=model,
        metrics=metrics,
        feature_columns=feature_columns,
        output_path=model_output_path,
        config=config,
    )
    return {
        "rows": int(len(matchup_frame)),
        "metrics": metrics,
        "dataset_output_path": str(dataset_output_path),
        "dataset_parquet_output_path": (
            str(dataset_parquet_output_path) if dataset_parquet_output_path is not None else None
        ),
        "model_output_path": str(model_output_path),
    }
