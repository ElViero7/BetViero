from __future__ import annotations

import random
from collections import Counter
from typing import Any


DEFAULT_ATP_SERVE_POINT_WIN_PROB = 0.64
INJURY_SERVE_PENALTY = 0.02


def clamp_probability(value: float, minimum: float = 0.01, maximum: float = 0.99) -> float:
    return max(minimum, min(maximum, value))


def simulate_game(p_serve: float) -> str:
    """Simulate one tennis game point by point and return 'server' or 'returner'."""
    p_serve = clamp_probability(p_serve)
    server_points = 0
    returner_points = 0

    while True:
        if random.random() < p_serve:
            server_points += 1
        else:
            returner_points += 1

        if server_points >= 4 and server_points - returner_points >= 2:
            return "server"
        if returner_points >= 4 and returner_points - server_points >= 2:
            return "returner"


def _tiebreak_server_for_point(point_index: int, first_server: str = "A") -> str:
    if point_index == 0:
        return first_server

    block_index = (point_index - 1) // 2
    if block_index % 2 == 0:
        return "B" if first_server == "A" else "A"
    return first_server


def simulate_tiebreak(p_serve_A: float, p_serve_B: float) -> str:
    """Simulate a standard first-to-7 tiebreak with correct service rotation."""
    p_serve_A = clamp_probability(p_serve_A)
    p_serve_B = clamp_probability(p_serve_B)
    points_a = 0
    points_b = 0
    point_index = 0

    while True:
        server = _tiebreak_server_for_point(point_index, first_server="A")
        if server == "A":
            point_won_by_a = random.random() < p_serve_A
        else:
            point_won_by_a = random.random() >= p_serve_B

        if point_won_by_a:
            points_a += 1
        else:
            points_b += 1

        if points_a >= 7 and points_a - points_b >= 2:
            return "A"
        if points_b >= 7 and points_b - points_a >= 2:
            return "B"

        point_index += 1


def simulate_set(p_serve_A: float, p_serve_B: float) -> dict[str, Any]:
    """Simulate one set with alternating service games and a tiebreak at 6-6."""
    games_a = 0
    games_b = 0
    next_server = "A"

    while True:
        if games_a == 6 and games_b == 6:
            tiebreak_winner = simulate_tiebreak(p_serve_A, p_serve_B)
            if tiebreak_winner == "A":
                games_a += 1
            else:
                games_b += 1
            return {
                "winner": tiebreak_winner,
                "games_a": games_a,
                "games_b": games_b,
                "tiebreak": True,
            }

        if next_server == "A":
            game_winner = simulate_game(p_serve_A)
            if game_winner == "server":
                games_a += 1
            else:
                games_b += 1
            next_server = "B"
        else:
            game_winner = simulate_game(p_serve_B)
            if game_winner == "server":
                games_b += 1
            else:
                games_a += 1
            next_server = "A"

        if games_a >= 6 and games_a - games_b >= 2:
            return {
                "winner": "A",
                "games_a": games_a,
                "games_b": games_b,
                "tiebreak": False,
            }
        if games_b >= 6 and games_b - games_a >= 2:
            return {
                "winner": "B",
                "games_a": games_a,
                "games_b": games_b,
                "tiebreak": False,
            }


def simulate_match(p_serve_A: float, p_serve_B: float, best_of: int = 3) -> dict[str, Any]:
    """Simulate a best-of-3 or best-of-5 match."""
    if best_of not in {3, 5}:
        raise ValueError("best_of must be 3 or 5.")

    sets_to_win = best_of // 2 + 1
    sets_a = 0
    sets_b = 0
    total_games_a = 0
    total_games_b = 0
    set_scores: list[tuple[int, int]] = []

    while sets_a < sets_to_win and sets_b < sets_to_win:
        set_result = simulate_set(p_serve_A, p_serve_B)
        total_games_a += int(set_result["games_a"])
        total_games_b += int(set_result["games_b"])
        set_scores.append((int(set_result["games_a"]), int(set_result["games_b"])))

        if set_result["winner"] == "A":
            sets_a += 1
        else:
            sets_b += 1

    return {
        "winner": "A" if sets_a > sets_b else "B",
        "sets_a": sets_a,
        "sets_b": sets_b,
        "set_scores": set_scores,
        "total_games": total_games_a + total_games_b,
        "game_diff": total_games_a - total_games_b,
    }


def run_monte_carlo(
    p_serve_A: float,
    p_serve_B: float,
    best_of: int,
    n_simulations: int = 1000,
) -> dict[str, Any]:
    if n_simulations <= 0:
        raise ValueError("n_simulations must be greater than 0.")

    matches = [simulate_match(p_serve_A, p_serve_B, best_of=best_of) for _ in range(n_simulations)]
    total_games = [int(match["total_games"]) for match in matches]
    game_diffs = [int(match["game_diff"]) for match in matches]
    winners = Counter(str(match["winner"]) for match in matches)

    def over_probability(line: float) -> float:
        return sum(games > line for games in total_games) / n_simulations

    common_over_lines = [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5]
    return {
        "n_simulations": n_simulations,
        "mean_total_games": sum(total_games) / n_simulations,
        "mean_game_diff": sum(game_diffs) / n_simulations,
        "win_probability_A": winners["A"] / n_simulations,
        "win_probability_B": winners["B"] / n_simulations,
        "total_games_distribution": dict(sorted(Counter(total_games).items())),
        "game_diff_distribution": dict(sorted(Counter(game_diffs).items())),
        "over_probabilities": {line: over_probability(line) for line in common_over_lines},
        "over_probability": over_probability,
    }


def _safe_float(value: Any, fallback: float = DEFAULT_ATP_SERVE_POINT_WIN_PROB) -> float:
    try:
        if value is None:
            return fallback
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:
        return fallback
    return number


def calculate_match_serve_probs(row: Any) -> tuple[float, float]:
    a_serve = _safe_float(getattr(row, "jugador_a_serve_points_won_pct_last20", None))
    a_return = _safe_float(getattr(row, "jugador_a_return_points_won_pct_last20", None))
    b_serve = _safe_float(getattr(row, "jugador_b_serve_points_won_pct_last20", None))
    b_return = _safe_float(getattr(row, "jugador_b_return_points_won_pct_last20", None))

    if isinstance(row, dict):
        a_serve = _safe_float(row.get("jugador_a_serve_points_won_pct_last20"))
        a_return = _safe_float(row.get("jugador_a_return_points_won_pct_last20"))
        b_serve = _safe_float(row.get("jugador_b_serve_points_won_pct_last20"))
        b_return = _safe_float(row.get("jugador_b_return_points_won_pct_last20"))
        a_injury = int(_safe_float(row.get("jugador_a_returning_from_injury"), 0.0)) == 1
        b_injury = int(_safe_float(row.get("jugador_b_returning_from_injury"), 0.0)) == 1
    else:
        a_injury = int(_safe_float(getattr(row, "jugador_a_returning_from_injury", 0.0), 0.0)) == 1
        b_injury = int(_safe_float(getattr(row, "jugador_b_returning_from_injury", 0.0), 0.0)) == 1

    p_serve_A = (a_serve + (1.0 - b_return)) / 2.0
    p_serve_B = (b_serve + (1.0 - a_return)) / 2.0

    if a_injury:
        p_serve_A -= INJURY_SERVE_PENALTY
    if b_injury:
        p_serve_B -= INJURY_SERVE_PENALTY

    return clamp_probability(p_serve_A), clamp_probability(p_serve_B)


if __name__ == "__main__":
    random.seed(42)
    result = run_monte_carlo(
        p_serve_A=0.68,
        p_serve_B=0.62,
        best_of=3,
        n_simulations=1000,
    )
    print("Test Monte Carlo: A p_serve=0.68, B p_serve=0.62, best_of=3")
    print(f"Mean total games: {result['mean_total_games']:.2f}")
    print(f"Mean game diff (A-B): {result['mean_game_diff']:.2f}")
    print(f"Win probability A: {result['win_probability_A']:.2%}")
    print(f"Win probability B: {result['win_probability_B']:.2%}")
    print(f"Over 22.5 probability: {result['over_probabilities'][22.5]:.2%}")
