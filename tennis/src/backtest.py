from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from simulator import calculate_match_serve_probs, run_monte_carlo
except ModuleNotFoundError:
    from tennis.src.simulator import calculate_match_serve_probs, run_monte_carlo


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "atp_features_avanzadas.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "sackmann"


def load_historical_ranks(raw_dir: Path) -> dict[tuple[str, int], tuple[float, float]]:
    """
    Carga rankings históricos de los archivos raw de Sackmann por (tourney_id, match_num).
    Retorna un diccionario de (tourney_id, match_num) -> (winner_rank, loser_rank)
    """
    ranks = {}
    paths = sorted(raw_dir.glob("atp_matches_*.csv"))
    print(f"[Carga] Buscando rankings en {len(paths)} archivos Sackmann raw...")
    for path in paths:
        try:
            df = pd.read_csv(path, usecols=["tourney_id", "match_num", "winner_rank", "loser_rank"])
            for _, row in df.iterrows():
                tid = str(row["tourney_id"])
                mnum = int(row["match_num"])
                wr = float(row["winner_rank"]) if pd.notna(row["winner_rank"]) else 999.0
                lr = float(row["loser_rank"]) if pd.notna(row["loser_rank"]) else 999.0
                ranks[(tid, mnum)] = (wr, lr)
        except Exception as exc:
            print(f"[Advertencia] Error al procesar rankings en {path.name}: {exc}")
            continue
    print(f"[Carga] Rankings cargados para {len(ranks)} partidos históricos.")
    return ranks


def estimate_market_h2h_odds(rank_a: float, rank_b: float, overround: float = 0.05) -> tuple[float, float]:
    """
    Estima las cuotas H2H del mercado basándose en la diferencia logarítmica de rankings.
    """
    # Evitar log(0) o valores negativos
    r_a = max(1.0, rank_a)
    r_b = max(1.0, rank_b)

    # Si ambos son iguales, probabilidad de mercado es 50/50
    if abs(r_a - r_b) < 1e-9:
        p_mkt_a = 0.5
    else:
        # Sensibilidad c=1.5 es estándar para tenis
        p_mkt_a = 1.0 / (1.0 + 10 ** ((np.log10(r_a) - np.log10(r_b)) / 1.5))

    p_mkt_b = 1.0 - p_mkt_a

    # Cuotas de la casa con overround (comisión)
    cuota_a = (1.0 - overround) / p_mkt_a
    cuota_b = (1.0 - overround) / p_mkt_b

    return max(1.01, cuota_a), max(1.01, cuota_b)


def run_backtest(
    features_path: Path = FEATURES_PATH,
    raw_dir: Path = RAW_DIR,
    min_ev: float = 0.02,
    max_ev: float = 0.30,
    n_sims: int = 1000,
    target_year: int | None = None,
) -> dict[str, Any]:
    if not features_path.exists():
        print(f"Error: El archivo de features avanzadas no existe en {features_path}")
        return {}

    print(f"\n==================================================")
    print(f" INICIANDO BACKTEST HISTÓRICO DE APUESTAS EN TENIS")
    print(f"==================================================")

    # Cargar features y rankings
    df_features = pd.read_csv(features_path)
    df_features["fecha"] = pd.to_datetime(df_features["fecha"])

    # Filtrar por año si es requerido
    if target_year is not None:
        df_features = df_features[df_features["fecha"].dt.year == target_year].copy()
        print(f"[Filtro] Evaluando únicamente partidos del año {target_year} ({len(df_features)} filas).")
    else:
        print(f"[Carga] Evaluando todos los partidos de {df_features['fecha'].dt.year.min()} a {df_features['fecha'].dt.year.max()} ({len(df_features)} filas).")

    # Estadísticas globales del backtest
    total_matches_evaluated = 0
    total_bets_placed = 0
    total_bets_won = 0
    total_pnl = 0.0
    total_amount_wagered = 0.0

    bets_by_year = {}
    all_bets = []

    print(f"\n[Backtest] Procesando simulaciones Monte Carlo vectorizadas...")

    # Para cruzar con precisión y rapidez, busquemos rankings en df_features haciendo un cruce de nombres.
    print(f"[Carga] Indexando rankings históricos por nombres normalizados de partido...")
    ranks_by_names = {}
    for path in raw_dir.glob("atp_matches_*.csv"):
        try:
            df = pd.read_csv(path, usecols=["tourney_id", "winner_name", "loser_name", "winner_rank", "loser_rank"])
            for _, r in df.iterrows():
                tid = str(r["tourney_id"])
                w_norm = str(r["winner_name"]).lower().strip()
                l_norm = str(r["loser_name"]).lower().strip()
                wr = float(r["winner_rank"]) if pd.notna(r["winner_rank"]) else 999.0
                lr = float(r["loser_rank"]) if pd.notna(r["loser_rank"]) else 999.0
                ranks_by_names[(tid, w_norm, l_norm)] = (wr, lr)
        except Exception:
            continue

    # Ahora sí, corremos el bucle del backtest
    random.seed(42)
    np.random.seed(42)

    for idx, row in df_features.iterrows():
        if (
            pd.isna(row["jugador_a_serve_points_won_pct_last20"]) or
            pd.isna(row["jugador_b_serve_points_won_pct_last20"])
        ):
            continue

        total_matches_evaluated += 1
        year = row["fecha"].year

        # 1. Recuperar rankings y aleatorizar roles para evitar leakage del ganador
        tid = str(row["tourney_id"])
        
        # En df_features, jugador_a es el ganador y jugador_b es el perdedor.
        # Aleatorizamos para que el modelo no sepa quién ganó a través del orden de entrada.
        invertir = random.random() < 0.5
        
        if invertir:
            p_a_name = str(row["jugador_b"]).lower().strip()
            p_b_name = str(row["jugador_a"]).lower().strip()
            # rank_a = perdedor (B original), rank_b = ganador (A original)
            rank_b, rank_a = ranks_by_names.get((tid, p_b_name, p_a_name), (150.0, 150.0))
            
            p_serve_a_init, p_serve_b_init = calculate_match_serve_probs(row)
            p_serve_a = p_serve_b_init
            p_serve_b = p_serve_a_init
            ganador_real = "B"
            real_game_diff = -float(row["diferencia_juegos"]) if pd.notna(row["diferencia_juegos"]) else 0.0
        else:
            p_a_name = str(row["jugador_a"]).lower().strip()
            p_b_name = str(row["jugador_b"]).lower().strip()
            # rank_a = ganador (A original), rank_b = perdedor (B original)
            rank_a, rank_b = ranks_by_names.get((tid, p_a_name, p_b_name), (150.0, 150.0))
            
            p_serve_a, p_serve_b = calculate_match_serve_probs(row)
            ganador_real = "A"
            real_game_diff = float(row["diferencia_juegos"]) if pd.notna(row["diferencia_juegos"]) else 0.0

        # 2. Estimar las cuotas de mercado H2H de la casa
        cuota_casa_a, cuota_casa_b = estimate_market_h2h_odds(rank_a, rank_b)

        # 3. Simular las probabilidades del modelo usando Monte Carlo vectorizado
        best_of = int(row["best_of"]) if pd.notna(row["best_of"]) else 3

        sim = run_monte_carlo(p_serve_a, p_serve_b, best_of=best_of, n_simulations=n_sims)
        prob_a = sim["win_probability_A"]
        prob_b = sim["win_probability_B"]

        # 4. Calcular/Colocar apuestas
        # H2H A
        ev_h2h_a = prob_a * cuota_casa_a - 1.0
        ev_h2h_a_capped = min(ev_h2h_a, max_ev)
        if ev_h2h_a_capped >= min_ev:
            win = (ganador_real == "A")
            pnl = cuota_casa_a - 1.0 if win else -1.0
            bet_record = {
                "fecha": row["fecha"],
                "superficie": row["surface"],
                "mercado": "H2H",
                "seleccion": p_a_name,
                "linea": 0.0,
                "cuota": cuota_casa_a,
                "prob": prob_a,
                "ev": ev_h2h_a_capped,
                "ev_crudo": ev_h2h_a,
                "win": win,
                "pnl": pnl
            }
            all_bets.append(bet_record)
            total_bets_placed += 1
            total_amount_wagered += 1.0
            total_pnl += pnl
            if win:
                total_bets_won += 1
            if year not in bets_by_year:
                bets_by_year[year] = {"bets": 0, "wins": 0, "pnl": 0.0}
            bets_by_year[year]["bets"] += 1
            if win:
                bets_by_year[year]["wins"] += 1
            bets_by_year[year]["pnl"] += pnl

        # H2H B
        ev_h2h_b = prob_b * cuota_casa_b - 1.0
        ev_h2h_b_capped = min(ev_h2h_b, max_ev)
        if ev_h2h_b_capped >= min_ev:
            win = (ganador_real == "B")
            pnl = cuota_casa_b - 1.0 if win else -1.0
            bet_record = {
                "fecha": row["fecha"],
                "superficie": row["surface"],
                "mercado": "H2H",
                "seleccion": p_b_name,
                "linea": 0.0,
                "cuota": cuota_casa_b,
                "prob": prob_b,
                "ev": ev_h2h_b_capped,
                "ev_crudo": ev_h2h_b,
                "win": win,
                "pnl": pnl
            }
            all_bets.append(bet_record)
            total_bets_placed += 1
            total_amount_wagered += 1.0
            total_pnl += pnl
            if win:
                total_bets_won += 1
            if year not in bets_by_year:
                bets_by_year[year] = {"bets": 0, "wins": 0, "pnl": 0.0}
            bets_by_year[year]["bets"] += 1
            if win:
                bets_by_year[year]["wins"] += 1
            bets_by_year[year]["pnl"] += pnl

        # Totales
        line_total = 22.5 if best_of == 3 else 38.5
        total_games_dist = sim["total_games_distribution"]
        total_sims = sum(total_games_dist.values())
        if total_sims > 0 and pd.notna(row["total_juegos"]):
            prob_over = sum(count for g, count in total_games_dist.items() if g > line_total) / total_sims
            prob_under = 1.0 - prob_over

            # Over
            ev_over = prob_over * 1.90 - 1.0
            ev_over_capped = min(ev_over, max_ev)
            if ev_over_capped >= min_ev:
                win = (float(row["total_juegos"]) > line_total)
                pnl = 1.90 - 1.0 if win else -1.0
                bet_record = {
                    "fecha": row["fecha"],
                    "superficie": row["surface"],
                    "mercado": "Totals",
                    "seleccion": "Over",
                    "linea": line_total,
                    "cuota": 1.90,
                    "prob": prob_over,
                    "ev": ev_over_capped,
                    "ev_crudo": ev_over,
                    "win": win,
                    "pnl": pnl
                }
                all_bets.append(bet_record)
                total_bets_placed += 1
                total_amount_wagered += 1.0
                total_pnl += pnl
                if win:
                    total_bets_won += 1
                if year not in bets_by_year:
                    bets_by_year[year] = {"bets": 0, "wins": 0, "pnl": 0.0}
                bets_by_year[year]["bets"] += 1
                if win:
                    bets_by_year[year]["wins"] += 1
                bets_by_year[year]["pnl"] += pnl

            # Under
            ev_under = prob_under * 1.90 - 1.0
            ev_under_capped = min(ev_under, max_ev)
            if ev_under_capped >= min_ev:
                win = (float(row["total_juegos"]) < line_total)
                pnl = 1.90 - 1.0 if win else -1.0
                bet_record = {
                    "fecha": row["fecha"],
                    "superficie": row["surface"],
                    "mercado": "Totals",
                    "seleccion": "Under",
                    "linea": line_total,
                    "cuota": 1.90,
                    "prob": prob_under,
                    "ev": ev_under_capped,
                    "ev_crudo": ev_under,
                    "win": win,
                    "pnl": pnl
                }
                all_bets.append(bet_record)
                total_bets_placed += 1
                total_amount_wagered += 1.0
                total_pnl += pnl
                if win:
                    total_bets_won += 1
                if year not in bets_by_year:
                    bets_by_year[year] = {"bets": 0, "wins": 0, "pnl": 0.0}
                bets_by_year[year]["bets"] += 1
                if win:
                    bets_by_year[year]["wins"] += 1
                bets_by_year[year]["pnl"] += pnl

        # Spreads
        line_hc = 3.5 if best_of == 3 else 5.5
        if prob_a >= 0.5:
            L_A = -line_hc
            L_B = line_hc
        else:
            L_A = line_hc
            L_B = -line_hc

        diff_dist = sim["game_diff_distribution"]
        total_diff_sims = sum(diff_dist.values())
        if total_diff_sims > 0 and pd.notna(row["diferencia_juegos"]):
            prob_spread_a = sum(count for d, count in diff_dist.items() if d > -L_A) / total_diff_sims
            prob_spread_b = sum(count for d, count in diff_dist.items() if d < L_B) / total_diff_sims

            # Spread A
            ev_spread_a = prob_spread_a * 1.90 - 1.0
            ev_spread_a_capped = min(ev_spread_a, max_ev)
            if ev_spread_a_capped >= min_ev:
                win = (real_game_diff > -L_A)
                pnl = 1.90 - 1.0 if win else -1.0
                bet_record = {
                    "fecha": row["fecha"],
                    "superficie": row["surface"],
                    "mercado": "Spreads",
                    "seleccion": f"Handicap {p_a_name}",
                    "linea": L_A,
                    "cuota": 1.90,
                    "prob": prob_spread_a,
                    "ev": ev_spread_a_capped,
                    "ev_crudo": ev_spread_a,
                    "win": win,
                    "pnl": pnl
                }
                all_bets.append(bet_record)
                total_bets_placed += 1
                total_amount_wagered += 1.0
                total_pnl += pnl
                if win:
                    total_bets_won += 1
                if year not in bets_by_year:
                    bets_by_year[year] = {"bets": 0, "wins": 0, "pnl": 0.0}
                bets_by_year[year]["bets"] += 1
                if win:
                    bets_by_year[year]["wins"] += 1
                bets_by_year[year]["pnl"] += pnl

            # Spread B
            ev_spread_b = prob_spread_b * 1.90 - 1.0
            ev_spread_b_capped = min(ev_spread_b, max_ev)
            if ev_spread_b_capped >= min_ev:
                win = (real_game_diff < L_B)
                pnl = 1.90 - 1.0 if win else -1.0
                bet_record = {
                    "fecha": row["fecha"],
                    "superficie": row["surface"],
                    "mercado": "Spreads",
                    "seleccion": f"Handicap {p_b_name}",
                    "linea": L_B,
                    "cuota": 1.90,
                    "prob": prob_spread_b,
                    "ev": ev_spread_b_capped,
                    "ev_crudo": ev_spread_b,
                    "win": win,
                    "pnl": pnl
                }
                all_bets.append(bet_record)
                total_bets_placed += 1
                total_amount_wagered += 1.0
                total_pnl += pnl
                if win:
                    total_bets_won += 1
                if year not in bets_by_year:
                    bets_by_year[year] = {"bets": 0, "wins": 0, "pnl": 0.0}
                bets_by_year[year]["bets"] += 1
                if win:
                    bets_by_year[year]["wins"] += 1
                bets_by_year[year]["pnl"] += pnl

    # Guardar el historial de apuestas a un archivo CSV
    if all_bets:
        df_bets = pd.DataFrame(all_bets)
        output_path = BASE_DIR / "data" / "processed" / "backtest_bets_history.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_bets.to_csv(output_path, index=False)
        print(f"[Info] Historial de {len(df_bets)} apuestas guardado en {output_path}")

    # Reporte de resultados
    win_rate = total_bets_won / total_bets_placed if total_bets_placed > 0 else 0.0
    roi = total_pnl / total_amount_wagered if total_amount_wagered > 0 else 0.0

    print(f"\n==================================================")
    print(f"               RESUMEN DE RESULTADOS              ")
    print(f"==================================================")
    print(f"Partidos con datos suficientes: {total_matches_evaluated:,}")
    print(f"Apuestas realizadas (EV >= {min_ev:.1%}): {total_bets_placed:,}")
    print(f"Apuestas ganadas: {total_bets_won:,}")
    print(f"Tasa de Acierto (Win Rate): {win_rate:.2%}")
    print(f"Dinero apostado total: {total_amount_wagered:.2f} ud")
    print(f"Beneficio Neto (PnL): {total_pnl:+.2f} ud")
    print(f"Retorno sobre Inversión (ROI): {roi:+.2%}")
    print(f"==================================================")

    if bets_by_year:
        print(f"\nDesglose por Año:")
        print(f"Año  | Apuestas | Aciertos | Win Rate | PnL (ud)  | ROI")
        print(f"-----|----------|----------|----------|-----------|------")
        for y in sorted(bets_by_year.keys()):
            stats = bets_by_year[y]
            y_wr = stats["wins"] / stats["bets"]
            y_roi = stats["pnl"] / stats["bets"]
            print(f"{y} | {stats['bets']:8d} | {stats['wins']:8d} | {y_wr:8.2%} | {stats['pnl']:+9.2f} | {y_roi:+6.2%}")
        print(f"==================================================\n")

    return {
        "total_matches_evaluated": total_matches_evaluated,
        "total_bets_placed": total_bets_placed,
        "total_bets_won": total_bets_won,
        "win_rate": win_rate,
        "total_amount_wagered": total_amount_wagered,
        "total_pnl": total_pnl,
        "roi": roi,
        "bets_by_year": bets_by_year,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtester histórico del pipeline de apuestas de tenis.")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH, help="Path a atp_features_avanzadas.csv")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Path a la carpeta de Sackmann raw")
    parser.add_argument("--min-ev", type=float, default=0.02, help="Límite mínimo de EV (default: 0.02)")
    parser.add_argument("--max-ev", type=float, default=0.30, help="Límite máximo/capping de EV (default: 0.30)")
    parser.add_argument("--n-sims", type=int, default=1000, help="Simulaciones Monte Carlo por partido (default: 1000)")
    parser.add_argument("--year", type=int, default=None, help="Filtrar por un año específico (ej. 2024)")
    args = parser.parse_args()

    run_backtest(
        features_path=args.features,
        raw_dir=args.raw_dir,
        min_ev=args.min_ev,
        max_ev=args.max_ev,
        n_sims=args.n_sims,
        target_year=args.year,
    )


if __name__ == "__main__":
    main()
