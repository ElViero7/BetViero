import pandas as pd

from lol.src.dashboard import normalize_name_key
from lol.src.fair_odds_bot import sanitize_for_json
from lol.src.multimarket import calculate_kelly_fraction, estimate_over_under_probability


def test_kelly_fraction_negative_edge_returns_zero():
    assert calculate_kelly_fraction(probability=0.40, odd=1.80) == 0.0


def test_kelly_fraction_positive_edge():
    result = calculate_kelly_fraction(probability=0.60, odd=2.10)
    assert 0.0 < result < 1.0


def test_estimate_over_prob_clamps_zero_std():
    probability = estimate_over_under_probability(
        predicted_total=28.0,
        line=27.5,
        residual_std=0.0,
        direction="over",
    )
    assert 0.0 <= probability <= 1.0


def test_estimate_under_over_sum_to_one():
    over = estimate_over_under_probability(28.0, 27.5, 5.0, "over")
    under = estimate_over_under_probability(28.0, 27.5, 5.0, "under")
    assert abs(over + under - 1.0) < 1e-9


def test_sanitize_for_json_converts_non_json_nulls():
    payload = sanitize_for_json(
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "nat": pd.NaT,
            "na": pd.NA,
        }
    )
    assert payload == {"nan": None, "inf": None, "nat": None, "na": None}


def test_normalize_name_key_keeps_force():
    assert normalize_name_key("Nongshim RedForce") == "nongshimredforce"
