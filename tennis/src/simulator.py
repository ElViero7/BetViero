from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Any
import pickle
from pathlib import Path



DEFAULT_ATP_SERVE_POINT_WIN_PROB = 0.64
INJURY_SERVE_PENALTY = 0.02


def clamp_probability(value: float, minimum: float = 0.01, maximum: float = 0.99) -> float:
    return max(minimum, min(maximum, value))


def game_win_prob(p: np.ndarray) -> np.ndarray:
    """
    Ecuación de Markov cerrada para calcular la probabilidad de ganar un juego
    dada la probabilidad de ganar un punto con el saque.
    """
    p = np.clip(p, 0.01, 0.99)
    q = 1.0 - p
    # P_G = p^4 + 4*p^4*q + 10*p^4*q^2 + 20*p^3*q^3 * (p^2 / (p^2 + q^2))
    return p**4 + 4 * (p**4) * q + 10 * (p**4) * (q**2) + 20 * (p**3) * (q**3) * (p**2 / (p**2 + q**2))


def simulate_tiebreaks_vectorized(p_serve_A_arr: np.ndarray, p_serve_B_arr: np.ndarray) -> np.ndarray:
    """
    Simula de forma vectorizada los tiebreaks (hasta 7 puntos con diferencia de 2)
    para todas las simulaciones indicadas en los arrays de entrada.
    Retorna un array con el ganador de cada tiebreak (0 para A, 1 para B).
    """
    num_tbs = len(p_serve_A_arr)
    if num_tbs == 0:
        return np.array([], dtype=np.int32)

    tb_points_a = np.zeros(num_tbs, dtype=np.int32)
    tb_points_b = np.zeros(num_tbs, dtype=np.int32)
    finished = np.zeros(num_tbs, dtype=bool)
    winners = np.zeros(num_tbs, dtype=np.int32)

    point_idx = 0
    while not np.all(finished):
        active = ~finished
        num_active = np.sum(active)

        if point_idx == 0:
            server_is_A = True
        else:
            block_idx = (point_idx - 1) // 2
            server_is_A = (block_idx % 2 == 1)

        r = np.random.random(num_active)
        if server_is_A:
            a_wins = r < p_serve_A_arr[active]
        else:
            # Saca B, por ende A gana si B pierde el punto de saque
            a_wins = r >= p_serve_B_arr[active]

        tb_points_a[active] += a_wins.astype(np.int32)
        tb_points_b[active] += (~a_wins).astype(np.int32)

        a_finished = (tb_points_a >= 7) & (tb_points_a - tb_points_b >= 2)
        b_finished = (tb_points_b >= 7) & (tb_points_b - tb_points_a >= 2)

        finished_now_A = active & a_finished
        finished_now_B = active & b_finished

        winners[finished_now_A] = 0
        winners[finished_now_B] = 1

        finished[finished_now_A | finished_now_B] = True
        point_idx += 1

        if point_idx > 100:  # Límite de seguridad
            forced = active & (~finished)
            winners[forced] = np.where(tb_points_a[forced] >= tb_points_b[forced], 0, 1)
            finished[forced] = True
            break

    return winners


def simulate_sets_vectorized(p_serve_A: float, p_serve_B: float, num_sims: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula en paralelo sets de tenis.
    Retorna:
      winners: array con el ganador (0 para A, 1 para B)
      games_a: array con los juegos ganados por A
      games_b: array con los juegos ganados por B
    """
    p_serve_A_arr = np.full(num_sims, p_serve_A)
    p_serve_B_arr = np.full(num_sims, p_serve_B)

    g_win_A = game_win_prob(p_serve_A_arr)
    g_win_B = game_win_prob(p_serve_B_arr)

    games_a = np.zeros(num_sims, dtype=np.int32)
    games_b = np.zeros(num_sims, dtype=np.int32)
    finished = np.zeros(num_sims, dtype=bool)
    winners = np.zeros(num_sims, dtype=np.int32)

    # Simular juegos del 1 al 12 en orden alternado
    for k in range(12):
        active = ~finished
        num_active = np.sum(active)
        if num_active == 0:
            break

        r = np.random.random(num_active)
        if k % 2 == 0:
            # Saca A (A gana con prob g_win_A)
            a_wins = r < g_win_A[active]
        else:
            # Saca B (B gana con prob g_win_B, A gana con prob 1 - g_win_B)
            a_wins = r >= g_win_B[active]

        games_a[active] += a_wins.astype(np.int32)
        games_b[active] += (~a_wins).astype(np.int32)

        a_finished = (games_a >= 6) & (games_a - games_b >= 2)
        b_finished = (games_b >= 6) & (games_b - games_a >= 2)

        finished_now_A = active & a_finished
        finished_now_B = active & b_finished

        winners[finished_now_A] = 0
        winners[finished_now_B] = 1

        finished[finished_now_A | finished_now_B] = True

    # Para las simulaciones que quedaron 6-6, jugar tiebreak
    tiebreak_mask = (~finished) & (games_a == 6) & (games_b == 6)
    if np.any(tiebreak_mask):
        tb_winners = simulate_tiebreaks_vectorized(p_serve_A_arr[tiebreak_mask], p_serve_B_arr[tiebreak_mask])
        games_a[tiebreak_mask] += (tb_winners == 0).astype(np.int32)
        games_b[tiebreak_mask] += (tb_winners == 1).astype(np.int32)
        winners[tiebreak_mask] = tb_winners
        finished[tiebreak_mask] = True

    return winners, games_a, games_b


def run_monte_carlo(
    p_serve_A: float,
    p_serve_B: float,
    best_of: int,
    n_simulations: int = 10000,
) -> dict[str, Any]:
    """
    Simulación Monte Carlo ultra-rápida y vectorizada de un partido de tenis.
    """
    if n_simulations <= 0:
        raise ValueError("n_simulations must be greater than 0.")

    sets_to_win = best_of // 2 + 1

    sets_won_a = np.zeros(n_simulations, dtype=np.int32)
    sets_won_b = np.zeros(n_simulations, dtype=np.int32)
    total_games_a = np.zeros(n_simulations, dtype=np.int32)
    total_games_b = np.zeros(n_simulations, dtype=np.int32)
    finished = np.zeros(n_simulations, dtype=bool)
    match_winners = np.zeros(n_simulations, dtype=np.int32)

    max_sets = best_of
    for _ in range(max_sets):
        active = ~finished
        num_active = np.sum(active)
        if num_active == 0:
            break

        set_winners, g_a, g_b = simulate_sets_vectorized(p_serve_A, p_serve_B, num_active)
        active_indices = np.where(active)[0]

        sets_won_a[active_indices] += (set_winners == 0).astype(np.int32)
        sets_won_b[active_indices] += (set_winners == 1).astype(np.int32)
        total_games_a[active_indices] += g_a
        total_games_b[active_indices] += g_b

        a_finished = (sets_won_a == sets_to_win)
        b_finished = (sets_won_b == sets_to_win)

        finished_now_A = active & a_finished
        finished_now_B = active & b_finished

        match_winners[finished_now_A] = 0
        match_winners[finished_now_B] = 1

        finished[finished_now_A | finished_now_B] = True

    total_games = total_games_a + total_games_b
    game_diffs = total_games_a - total_games_b

    win_prob_A = np.sum(match_winners == 0) / n_simulations
    win_prob_B = np.sum(match_winners == 1) / n_simulations

    unique_games, counts_games = np.unique(total_games, return_counts=True)
    total_games_dist = dict(zip(unique_games.tolist(), counts_games.tolist()))

    unique_diffs, counts_diffs = np.unique(game_diffs, return_counts=True)
    game_diff_dist = dict(zip(unique_diffs.tolist(), counts_diffs.tolist()))

    def over_probability(line: float) -> float:
        return float(np.sum(total_games > line) / n_simulations)

    common_over_lines = [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5]
    over_probs = {line: over_probability(line) for line in common_over_lines}

    return {
        "n_simulations": n_simulations,
        "mean_total_games": float(np.mean(total_games)),
        "mean_game_diff": float(np.mean(game_diffs)),
        "win_probability_A": float(win_prob_A),
        "win_probability_B": float(win_prob_B),
        "total_games_distribution": total_games_dist,
        "game_diff_distribution": game_diff_dist,
        "over_probabilities": over_probs,
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


_SERVE_REGRESSOR = None

def get_serve_regressor():
    global _SERVE_REGRESSOR
    if _SERVE_REGRESSOR is None:
        model_path = Path(__file__).resolve().parent.parent / "models" / "serve_regressor.pkl"
        if not model_path.exists():
            print(f"[ADVERTENCIA] Archivo de modelo no encontrado en {model_path}. Se usará la fórmula estática promedio.")
            return None
        try:
            with open(model_path, "rb") as f:
                _SERVE_REGRESSOR = pickle.load(f)
            print(f"[INFO] Modelo ML cargado con xito desde {model_path} (Variables: {', '.join(_SERVE_REGRESSOR['features'])}).")
        except Exception as exc:
            print(f"[ADVERTENCIA] Error al cargar el modelo: {exc}. Se usará la fórmula estática promedio.")
            return None
    return _SERVE_REGRESSOR


def calculate_match_serve_probs(row: Any) -> tuple[float, float]:
    # Intentar obtener valores básicos
    if isinstance(row, dict):
        a_serve = _safe_float(row.get("jugador_a_serve_points_won_pct_last20"))
        a_return = _safe_float(row.get("jugador_a_return_points_won_pct_last20"))
        b_serve = _safe_float(row.get("jugador_b_serve_points_won_pct_last20"))
        b_return = _safe_float(row.get("jugador_b_return_points_won_pct_last20"))
        a_injury = int(_safe_float(row.get("jugador_a_returning_from_injury"), 0.0)) == 1
        b_injury = int(_safe_float(row.get("jugador_b_returning_from_injury"), 0.0)) == 1
        a_dias = _safe_float(row.get("jugador_a_dias_desde_ultimo_partido"), 30.0)
        b_dias = _safe_float(row.get("jugador_b_dias_desde_ultimo_partido"), 30.0)
        a_minutos = _safe_float(row.get("jugador_a_minutos_acumulados_torneo"), 0.0)
        b_minutos = _safe_float(row.get("jugador_b_minutos_acumulados_torneo"), 0.0)
        surface = str(row.get("surface", "Hard"))
        best_of = float(row.get("best_of", 3))
    else:
        a_serve = _safe_float(getattr(row, "jugador_a_serve_points_won_pct_last20", None))
        a_return = _safe_float(getattr(row, "jugador_a_return_points_won_pct_last20", None))
        b_serve = _safe_float(getattr(row, "jugador_b_serve_points_won_pct_last20", None))
        b_return = _safe_float(getattr(row, "jugador_b_return_points_won_pct_last20", None))
        a_injury = int(_safe_float(getattr(row, "jugador_a_returning_from_injury", 0.0), 0.0)) == 1
        b_injury = int(_safe_float(getattr(row, "jugador_b_returning_from_injury", 0.0), 0.0)) == 1
        a_dias = _safe_float(getattr(row, "jugador_a_dias_desde_ultimo_partido", 30.0), 30.0)
        b_dias = _safe_float(getattr(row, "jugador_b_dias_desde_ultimo_partido", 30.0), 30.0)
        a_minutos = _safe_float(getattr(row, "jugador_a_minutos_acumulados_torneo", 0.0), 0.0)
        b_minutos = _safe_float(getattr(row, "jugador_b_minutos_acumulados_torneo", 0.0), 0.0)
        surface = str(getattr(row, "surface", "Hard"))
        best_of = float(getattr(row, "best_of", 3))

    regressor_dict = get_serve_regressor()
    if regressor_dict is not None:
        try:
            model = regressor_dict["model"]
            
            # Predict for A serving against B
            # Feature columns:
            # "server_serve_pct_last20", "returner_return_pct_last20", "server_dias", "returner_dias",
            # "server_injury", "returner_injury", "server_minutos", "returner_minutos",
            # "is_hard", "is_clay", "is_grass", "is_carpet", "best_of"
            features_a = [
                a_serve, b_return, a_dias, b_dias,
                float(a_injury), float(b_injury), a_minutos, b_minutos,
                1 if surface == "Hard" else 0,
                1 if surface == "Clay" else 0,
                1 if surface == "Grass" else 0,
                1 if surface == "Carpet" else 0,
                best_of
            ]
            p_serve_A = float(model.predict([features_a])[0])
            
            # Predict for B serving against A
            features_b = [
                b_serve, a_return, b_dias, a_dias,
                float(b_injury), float(a_injury), b_minutos, a_minutos,
                1 if surface == "Hard" else 0,
                1 if surface == "Clay" else 0,
                1 if surface == "Grass" else 0,
                1 if surface == "Carpet" else 0,
                best_of
            ]
            p_serve_B = float(model.predict([features_b])[0])
            
            return clamp_probability(p_serve_A), clamp_probability(p_serve_B)
        except Exception as exc:
            print(f"[ADVERTENCIA] Error en la predicción del modelo ML: {exc}. Se usará la fórmula estática promedio.")

    # Fallback clásico a fórmula estática promedio
    p_serve_A = (a_serve + (1.0 - b_return)) / 2.0
    p_serve_B = (b_serve + (1.0 - a_return)) / 2.0

    if a_injury:
        p_serve_A -= INJURY_SERVE_PENALTY
    if b_injury:
        p_serve_B -= INJURY_SERVE_PENALTY

    return clamp_probability(p_serve_A), clamp_probability(p_serve_B)


if __name__ == "__main__":
    np.random.seed(42)
    result = run_monte_carlo(
        p_serve_A=0.68,
        p_serve_B=0.62,
        best_of=3,
        n_simulations=10000,
    )
    print("Test Monte Carlo: A p_serve=0.68, B p_serve=0.62, best_of=3 (10,000 sims)")
    print(f"Mean total games: {result['mean_total_games']:.2f}")
    print(f"Mean game diff (A-B): {result['mean_game_diff']:.2f}")
    print(f"Win probability A: {result['win_probability_A']:.2%}")
    print(f"Win probability B: {result['win_probability_B']:.2%}")
    print(f"Over 22.5 probability: {result['over_probabilities'][22.5]:.2%}")
