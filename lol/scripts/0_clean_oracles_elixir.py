from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.firstblood import filter_team_rows, load_input_frame, save_parquet_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Oracle's Elixir raw files and export filtered Map 1 team rows to Parquet."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Raw CSV directory/file or existing Parquet directory/file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/oracles_elixir_team_rows_map1.parquet"),
        help="Destination Parquet path for cleaned team-level rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_frame = load_input_frame(args.input_path)
    clean_frame = filter_team_rows(raw_frame)
    save_parquet_dataset(clean_frame, args.output_path)
    print(f"Saved {len(clean_frame)} cleaned team rows to {args.output_path}.")


if __name__ == "__main__":
    main()
