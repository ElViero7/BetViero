from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import gdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base import LoLPipelineError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Oracle's Elixir CSV files from a Google Drive folder into the raw data directory."
    )
    parser.add_argument(
        "--drive-url",
        type=str,
        default=None,
        help="Google Drive folder URL. Defaults to ORACLES_ELIXIR_DRIVE_URL from .env.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to ORACLES_ELIXIR_RAW_DIR from .env.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise LoLPipelineError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    load_dotenv()
    args = parse_args()
    drive_url = args.drive_url or require_env("ORACLES_ELIXIR_DRIVE_URL")
    output_dir = args.output_dir or Path(os.getenv("ORACLES_ELIXIR_RAW_DIR", "data/raw/oracles_elixir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = gdown.download_folder(
        url=drive_url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
        skip_download=args.skip_existing,
    )
    if not downloaded:
        raise LoLPipelineError(
            "No files were downloaded from the Oracle's Elixir Drive folder."
        )

    print("Downloaded files:")
    for path in downloaded:
        print(path)


if __name__ == "__main__":
    main()
