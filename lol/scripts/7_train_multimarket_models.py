from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimarket import MultiMarketTrainingConfig, train_multimarket_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train winner, first blood, first tower and total kills models."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="CSV/Parquet file or directory containing Oracle's Elixir historical data.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/modelo_multimarket.pkl"),
        help="Output path for the multi-market artifact.",
    )
    parser.add_argument(
        "--team-rows-output",
        type=Path,
        default=Path("data/processed/oracles_elixir_team_rows_training.parquet"),
        help="Output Parquet path for canonical team rows.",
    )
    parser.add_argument(
        "--matchup-output",
        type=Path,
        default=Path("data/processed/multimarket_matchups.parquet"),
        help="Output Parquet path for matchup rows.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Chronological holdout ratio used for evaluation.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=200,
        help="Minimum number of matchup rows required to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_multimarket_pipeline(
        input_path=args.input_path,
        model_output_path=args.model_output,
        team_rows_output_path=args.team_rows_output,
        matchup_output_path=args.matchup_output,
        config=MultiMarketTrainingConfig(
            min_train_rows=args.min_train_rows,
            test_size=args.test_size,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
