from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.api_football import ApiFootballETL, ApiFootballETLError
from src.config import SettingsError


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-id", type=int, default=39)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--last-n", type=int, default=100)
    parser.add_argument("--mock-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    mock_dir = Path(args.mock_dir) if args.mock_dir else None

    try:
        etl = ApiFootballETL(mock_dir=mock_dir)
        df = etl.fetch_last_league_matches(
            league_id=args.league_id,
            season=args.season,
            last_n=args.last_n,
        )
    except (ApiFootballETLError, SettingsError) as exc:
        LOGGER.exception("ETL execution failed: %s", exc)
        raise SystemExit(1) from exc

    print(df.head(10).to_string(index=False))
    print(f"\nRows returned: {len(df)}")


if __name__ == "__main__":
    main()
