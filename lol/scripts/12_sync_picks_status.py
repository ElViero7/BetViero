from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard import (
    build_oracle_results_frame,
    fetch_table,
    load_dashboard_config,
    settle_picks,
    update_table_row,
)


def main() -> None:
    config = load_dashboard_config()
    picks_frame = fetch_table(config, "picks")
    if not picks_frame.empty:
        picks_frame = picks_frame.loc[
            picks_frame.get("status").fillna("pending").astype(str).str.lower().eq("pending")
        ].copy()
    results_frame = build_oracle_results_frame(config.raw_data_dir)
    settled = settle_picks(picks_frame, results_frame)

    if settled.empty:
        print("No picks found to sync.")
        return

    updated = 0
    for _, row in settled.iterrows():
        row_id = str(row["id"])
        payload = {"status": str(row["settled_status"])}
        update_table_row(config, "picks", row_id, payload)
        updated += 1

    print(f"Synchronized {updated} picks.")


if __name__ == "__main__":
    main()
