from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimarket import (
    build_multimarket_training_dataset,
    MultiMarketTrainingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical team rows and matchup dataset for the LoL multi-market stack."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="CSV/Parquet file or directory containing Oracle's Elixir historical data.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    team_frame, matchup_frame = build_multimarket_training_dataset(
        input_path=args.input_path,
        config=MultiMarketTrainingConfig(),
        team_rows_output_path=args.team_rows_output,
        matchup_output_path=args.matchup_output,
    )
    print(
        f"Saved {len(team_frame)} canonical team rows to {args.team_rows_output} "
        f"and {len(matchup_frame)} matchup rows to {args.matchup_output}."
    )


if __name__ == "__main__":
    main()
