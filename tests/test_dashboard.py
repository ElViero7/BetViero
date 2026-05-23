import pandas as pd

from lol.src.dashboard import settle_picks


def test_settle_picks_kills_with_nan_line_is_void():
    picks = pd.DataFrame(
        [
            {
                "id": "pick-1",
                "match_name": "Team A vs Team B",
                "created_at": "2026-05-01T00:00:00Z",
                "league": "LCK",
                "market": "kills",
                "selection": "Over",
                "line": float("nan"),
                "predicted_value": 28.0,
                "status": "pending",
            }
        ]
    )
    results = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2026-05-02T00:00:00Z"),
                "league": "LCK",
                "pair_key": "teama||teamb",
                "winner": "Team A",
                "firstblood_winner": "Team A",
                "firsttower_winner": "Team B",
                "total_kills": 31.0,
            }
        ]
    )

    settled = settle_picks(picks, results)

    assert settled.loc[0, "settled_status"] == "void"
    assert pd.isna(settled.loc[0, "settled_actual_value"])
