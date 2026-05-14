from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh Oracle raw data from Drive, retrain the multi-market model and sync legacy bets."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the Drive download step.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip retraining the multi-market artifact.",
    )
    parser.add_argument(
        "--skip-sync-bets",
        action="store_true",
        help="Skip syncing settled legacy bets to Supabase.",
    )
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    python_executable = sys.executable

    if not args.skip_download:
        run_step([python_executable, "scripts/10_download_oracles_elixir_drive.py"])
    if not args.skip_train:
        run_step(
            [
                python_executable,
                "scripts/7_train_multimarket_models.py",
                "--input-path",
                "data/raw/oracles_elixir",
            ]
        )
    if not args.skip_sync_bets:
        run_step([python_executable, "scripts/9_sync_legacy_bets_status.py"])


if __name__ == "__main__":
    main()
