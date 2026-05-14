from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.firstblood import TrainingConfig, build_training_dataset_from_csvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Map 1 First Blood training dataset from Oracle's Elixir CSVs."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    dataset = build_training_dataset_from_csvs(
        input_path=args.input_path,
        dataset_output_path=args.dataset_output,
        dataset_parquet_output_path=args.dataset_parquet_output,
        config=config,
    )
    print(
        f"Saved {len(dataset)} matchup rows to {args.dataset_output} "
        f"and {args.dataset_parquet_output}."
    )


if __name__ == "__main__":
    main()
