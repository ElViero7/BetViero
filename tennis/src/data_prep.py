from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "sackmann"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "atp_features_avanzadas.csv"
ROLLING_WINDOW = 20

INCOMPLETE_SCORE_TOKENS = {"RET", "W/O", "WO", "DEF", "ABD", "ABN", "Walkover"}


class TennisDataPrepError(RuntimeError):
    """Raised when advanced ATP features cannot be built."""


def parse_set_scores(score: object) -> list[tuple[int, int]]:
    if pd.isna(score):
        return []
    score_text = str(score).strip()
    if not score_text or any(token in score_text for token in INCOMPLETE_SCORE_TOKENS):
        return []

    sets: list[tuple[int, int]] = []
    for raw_set in score_text.split():
        if "[" in raw_set or "]" in raw_set:
            continue
        clean_set = raw_set.split("(", 1)[0]
        if "-" not in clean_set:
            continue
        left, right = clean_set.split("-", 1)
        if left.isdigit() and right.isdigit():
            sets.append((int(left), int(right)))
    return sets


def total_games(score: object) -> int | None:
    sets = parse_set_scores(score)
    if not sets:
        return None
    return sum(winner_games + loser_games for winner_games, loser_games in sets)


def game_difference(score: object) -> int | None:
    sets = parse_set_scores(score)
    if not sets:
        return None
    return sum(winner_games - loser_games for winner_games, loser_games in sets)


def load_raw_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("atp_matches_*.csv"))
    if not paths:
        raise TennisDataPrepError(f"No raw Sackmann CSVs found in {raw_dir}")

    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    frame["fecha"] = pd.to_datetime(frame["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame["source_row_id"] = range(len(frame))
    frame["total_juegos"] = frame["score"].map(total_games)
    frame["diferencia_juegos"] = frame["score"].map(game_difference)
    frame = frame.dropna(subset=["fecha", "winner_id", "loser_id", "score", "total_juegos", "diferencia_juegos"])
    frame["total_juegos"] = frame["total_juegos"].astype(int)
    frame["diferencia_juegos"] = frame["diferencia_juegos"].astype(int)
    return frame.sort_values(["fecha", "tourney_id", "match_num", "source_row_id"]).reset_index(drop=True)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def build_player_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    winner_rows = pd.DataFrame(
        {
            "source_row_id": matches["source_row_id"],
            "fecha": matches["fecha"],
            "tourney_id": matches["tourney_id"],
            "match_num": matches["match_num"],
            "surface": matches["surface"],
            "player_id": matches["winner_id"],
            "player_name": matches["winner_name"],
            "opponent_id": matches["loser_id"],
            "opponent_name": matches["loser_name"],
            "is_winner": 1,
            "minutes": matches["minutes"],
            "serve_points": matches["w_svpt"],
            "serve_points_won": matches["w_1stWon"] + matches["w_2ndWon"],
            "return_points": matches["l_svpt"],
            "return_points_won": matches["l_svpt"] - matches["l_1stWon"] - matches["l_2ndWon"],
        }
    )
    loser_rows = pd.DataFrame(
        {
            "source_row_id": matches["source_row_id"],
            "fecha": matches["fecha"],
            "tourney_id": matches["tourney_id"],
            "match_num": matches["match_num"],
            "surface": matches["surface"],
            "player_id": matches["loser_id"],
            "player_name": matches["loser_name"],
            "opponent_id": matches["winner_id"],
            "opponent_name": matches["winner_name"],
            "is_winner": 0,
            "minutes": matches["minutes"],
            "serve_points": matches["l_svpt"],
            "serve_points_won": matches["l_1stWon"] + matches["l_2ndWon"],
            "return_points": matches["w_svpt"],
            "return_points_won": matches["w_svpt"] - matches["w_1stWon"] - matches["w_2ndWon"],
        }
    )
    player_rows = pd.concat([winner_rows, loser_rows], ignore_index=True)
    numeric_columns = ["minutes", "serve_points", "serve_points_won", "return_points", "return_points_won"]
    for column in numeric_columns:
        player_rows[column] = pd.to_numeric(player_rows[column], errors="coerce")
    player_rows["serve_points_won_pct_match"] = safe_divide(
        player_rows["serve_points_won"],
        player_rows["serve_points"],
    )
    player_rows["return_points_won_pct_match"] = safe_divide(
        player_rows["return_points_won"],
        player_rows["return_points"],
    )
    return player_rows.sort_values(["player_id", "fecha", "tourney_id", "match_num", "source_row_id"]).reset_index(drop=True)


def add_no_leakage_player_features(player_rows: pd.DataFrame) -> pd.DataFrame:
    featured = player_rows.copy()
    grouped_by_player_surface = featured.groupby(["player_id", "surface"], group_keys=False)
    grouped_by_player = featured.groupby("player_id", group_keys=False)

    featured["serve_points_won_pct_last20"] = grouped_by_player_surface["serve_points_won_pct_match"].transform(
        lambda series: series.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
    )
    featured["return_points_won_pct_last20"] = grouped_by_player_surface["return_points_won_pct_match"].transform(
        lambda series: series.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
    )
    featured["prev_fecha"] = grouped_by_player["fecha"].shift(1)
    featured["dias_desde_ultimo_partido"] = (featured["fecha"] - featured["prev_fecha"]).dt.days
    is_offseason = (
        (featured["fecha"].dt.year > featured["prev_fecha"].dt.year) &
        (featured["dias_desde_ultimo_partido"] <= 90)
    )
    featured["returning_from_injury"] = (
        (featured["dias_desde_ultimo_partido"] > 45) & ~is_offseason
    ).astype(int)
    featured.drop(columns=["prev_fecha"], inplace=True)
    featured["minutos_acumulados_torneo"] = (
        featured.groupby(["player_id", "tourney_id"], group_keys=False)["minutes"]
        .transform(lambda series: series.shift(1).fillna(0).cumsum())
    )
    return featured


def merge_player_features(matches: pd.DataFrame, player_features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "source_row_id",
        "is_winner",
        "player_id",
        "serve_points_won_pct_last20",
        "return_points_won_pct_last20",
        "dias_desde_ultimo_partido",
        "returning_from_injury",
        "minutos_acumulados_torneo",
    ]
    winner_features = (
        player_features.loc[player_features["is_winner"].eq(1), feature_columns]
        .drop(columns=["is_winner"])
        .rename(
            columns={
                "player_id": "jugador_a_id",
                "serve_points_won_pct_last20": "jugador_a_serve_points_won_pct_last20",
                "return_points_won_pct_last20": "jugador_a_return_points_won_pct_last20",
                "dias_desde_ultimo_partido": "jugador_a_dias_desde_ultimo_partido",
                "returning_from_injury": "jugador_a_returning_from_injury",
                "minutos_acumulados_torneo": "jugador_a_minutos_acumulados_torneo",
            }
        )
    )
    loser_features = (
        player_features.loc[player_features["is_winner"].eq(0), feature_columns]
        .drop(columns=["is_winner"])
        .rename(
            columns={
                "player_id": "jugador_b_id",
                "serve_points_won_pct_last20": "jugador_b_serve_points_won_pct_last20",
                "return_points_won_pct_last20": "jugador_b_return_points_won_pct_last20",
                "dias_desde_ultimo_partido": "jugador_b_dias_desde_ultimo_partido",
                "returning_from_injury": "jugador_b_returning_from_injury",
                "minutos_acumulados_torneo": "jugador_b_minutos_acumulados_torneo",
            }
        )
    )

    output = matches.merge(winner_features, on="source_row_id", how="left")
    output = output.merge(loser_features, on="source_row_id", how="left")
    output = output.rename(
        columns={
            "winner_name": "jugador_a",
            "loser_name": "jugador_b",
        }
    )
    keep_columns = [
        "fecha",
        "tourney_id",
        "tourney_name",
        "surface",
        "best_of",
        "round",
        "jugador_a",
        "jugador_b",
        "jugador_a_id",
        "jugador_b_id",
        "total_juegos",
        "diferencia_juegos",
        "jugador_a_serve_points_won_pct_last20",
        "jugador_a_return_points_won_pct_last20",
        "jugador_a_dias_desde_ultimo_partido",
        "jugador_a_returning_from_injury",
        "jugador_a_minutos_acumulados_torneo",
        "jugador_b_serve_points_won_pct_last20",
        "jugador_b_return_points_won_pct_last20",
        "jugador_b_dias_desde_ultimo_partido",
        "jugador_b_returning_from_injury",
        "jugador_b_minutos_acumulados_torneo",
    ]
    output = output[keep_columns].copy()
    output["fecha"] = output["fecha"].dt.date
    return output.sort_values(["fecha", "tourney_id", "round", "jugador_a", "jugador_b"]).reset_index(drop=True)


def build_advanced_features(raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    matches = load_raw_matches(raw_dir)
    player_rows = build_player_match_rows(matches)
    player_features = add_no_leakage_player_features(player_rows)
    output = merge_player_features(matches, player_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-leakage advanced ATP features from Sackmann raw CSVs.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_advanced_features(raw_dir=args.raw_dir, output_path=args.output)
    print(f"Saved {len(frame)} rows to {args.output}")
    frame.info()
    null_rates = frame.isna().mean().sort_values(ascending=False)
    print("\nNull rates:")
    print(null_rates.to_string())


if __name__ == "__main__":
    main()
