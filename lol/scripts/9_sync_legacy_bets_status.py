from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard import (
    build_oracle_results_frame,
    load_dashboard_config,
    fetch_table,
    settle_legacy_bets,
    update_table_row,
)


def main() -> None:
    config = load_dashboard_config()
    bets_frame = fetch_table(config, "bets")
    if not bets_frame.empty:
        bets_frame = bets_frame.loc[
            bets_frame.get("status").fillna("pending").astype(str).str.lower().eq("pending")
        ].copy()
    results_frame = build_oracle_results_frame(config.raw_data_dir)
    settled = settle_legacy_bets(bets_frame, results_frame)

    if settled.empty:
        print("No legacy value bets found to sync.")
        return

    updated = 0
    for _, row in settled.iterrows():
        row_id = str(row["id"])
        status = str(row["settled_status"])
        payload = {"status": status}
        update_table_row(config, "bets", row_id, payload)
        updated += 1

    print(f"Synchronized {updated} legacy bets.")


if __name__ == "__main__":
    main()
