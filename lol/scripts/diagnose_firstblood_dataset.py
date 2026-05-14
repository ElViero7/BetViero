from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


NUMERIC_FEATURES = [
    "blue_games_played_pre",
    "red_games_played_pre",
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
    "diff_golddiffat15_roll5",
    "diff_xpdiffat15_roll5",
    "diff_vspm_roll5",
    "diff_ckpm_roll5",
    "diff_firstblood_rate_roll5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the First Blood Map 1 training dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/firstblood_training_dataset.csv"),
        help="Path to the matchup training dataset.",
    )
    return parser.parse_args()


def safe_auc(target: pd.Series, score: pd.Series) -> float | None:
    valid = pd.concat([target, score], axis=1).dropna()
    if valid.empty or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None
    return float(roc_auc_score(valid.iloc[:, 0], valid.iloc[:, 1]))


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.dataset_path)
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True, errors="coerce")
    target = frame["target_blue_firstblood"]

    print("=== Dataset Summary ===")
    print(f"rows: {len(frame)}")
    print(f"date_min: {frame['match_date'].min()}")
    print(f"date_max: {frame['match_date'].max()}")
    print(f"blue_firstblood_rate: {target.mean():.4f}")
    print()

    print("=== League Distribution ===")
    print(
        frame.groupby("league")["target_blue_firstblood"]
        .agg(["count", "mean"])
        .rename(columns={"count": "rows", "mean": "blue_firstblood_rate"})
        .to_string()
    )
    print()

    print("=== Patch Distribution ===")
    print(frame["patch"].value_counts().head(15).to_string())
    print()

    print("=== Missingness (%) ===")
    print((frame.isna().mean().sort_values(ascending=False) * 100).round(2).head(20).to_string())
    print()

    print("=== Teams With No Prior History On At Least One Side ===")
    cold_start = (
        (frame["blue_games_played_pre"] == 0) | (frame["red_games_played_pre"] == 0)
    ).mean()
    print(f"cold_start_match_share: {cold_start:.4f}")
    print()

    print("=== Single-Feature AUC Probe ===")
    auc_rows: list[tuple[str, float | None]] = []
    for feature in NUMERIC_FEATURES:
        auc_rows.append((feature, safe_auc(target, frame[feature])))
    auc_frame = pd.DataFrame(auc_rows, columns=["feature", "auc"]).sort_values(
        "auc",
        ascending=False,
        na_position="last",
    )
    print(auc_frame.to_string(index=False))
    print()

    print("=== Simple Baselines ===")
    global_rate = pd.Series(target.mean(), index=frame.index)
    league_rate = (
        frame.groupby("league")["target_blue_firstblood"].transform("mean")
    )
    patch_rate = (
        frame.groupby("patch")["target_blue_firstblood"].transform("mean")
    )
    print(f"global_rate_auc: {safe_auc(target, global_rate)}")
    print(f"league_rate_auc: {safe_auc(target, league_rate)}")
    print(f"patch_rate_auc: {safe_auc(target, patch_rate)}")


if __name__ == "__main__":
    main()
