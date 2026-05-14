from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.firstblood import TrainingConfig, run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Map 1 First Blood XGBoost model from Oracle's Elixir CSVs."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="CSV file or directory containing Oracle's Elixir CSVs for 2025/2026.",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("data/processed/firstblood_training_dataset.csv"),
        help="Where to save the merged Blue vs Red training dataset.",
    )
    parser.add_argument(
        "--dataset-parquet-output",
        type=Path,
        default=Path("data/processed/firstblood_training_dataset.parquet"),
        help="Where to save the merged Blue vs Red training dataset in Parquet.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/modelo_firstblood.pkl"),
        help="Where to save the trained model artifact.",
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
        default=50,
        help="Minimum number of matchup rows required to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        min_train_rows=args.min_train_rows,
        test_size=args.test_size,
    )
    summary = run_training_pipeline(
        input_path=args.input_path,
        dataset_output_path=args.dataset_output,
        dataset_parquet_output_path=args.dataset_parquet_output,
        model_output_path=args.model_output,
        config=config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
