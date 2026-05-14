from __future__ import annotations

import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from simulator import calculate_match_serve_probs, run_monte_carlo
    from odds_listener import run as fetch_and_save_odds
except ModuleNotFoundError:
    from tennis.src.simulator import calculate_match_serve_probs, run_monte_carlo
    from tennis.src.odds_listener import run as fetch_and_save_odds


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "lol" / ".env")

FEATURES_PATH = BASE_DIR / "data" / "processed" / "atp_features_avanzadas.csv"
ODDS_PATH = BASE_DIR / "data" / "processed" / "tennis_market_odds.csv"
OUTPUT_PATH = BASE_DIR / "data" / "predictions" / "upcoming_value_bets_tennis.csv"
MIN_NAME_SCORE = 0.72
TENNIS_PICKS_TABLE = "tennis_picks"
PICK_ODDS_HISTORY_TABLE = "pick_odds_history"


class TennisEVCalculatorError(RuntimeError):
    """Raised when tennis value bets cannot be calculated."""


class SupabaseResponse:
    def __init__(self, data: list[dict[str, Any]] | dict[str, Any] | None) -> None:
        self.data = data


class SupabaseQuery:
    def __init__(self, base_url: str, service_role_key: str, schema_name: str, table_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_role_key = service_role_key
        self.schema_name = schema_name
        self.table_name = table_name
        self.method = "GET"
        self.params: dict[str, str] = {}
        self.payload: dict[str, Any] | None = None

    def select(self, columns: str) -> "SupabaseQuery":
        self.method = "GET"
        self.params["select"] = columns
        return self

    def eq(self, column: str, value: Any) -> "SupabaseQuery":
        self.params[column] = f"eq.{value}"
        return self

    def order(self, column: str, desc: bool = False) -> "SupabaseQuery":
        direction = "desc" if desc else "asc"
        self.params["order"] = f"{column}.{direction}"
        return self

    def limit(self, count: int) -> "SupabaseQuery":
        self.params["limit"] = str(count)
        return self

    def insert(self, data: dict[str, Any]) -> "SupabaseQuery":
        self.method = "POST"
        self.payload = data
        return self

    def execute(self) -> SupabaseResponse:
        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Accept-Profile": self.schema_name,
            "Content-Profile": self.schema_name,
        }
        url = f"{self.base_url}/rest/v1/{self.table_name}"
        if self.method == "POST":
            headers["Content-Type"] = "application/json"
            headers["Prefer"] = "return=representation"
            response = requests.post(url, headers=headers, params=self.params, json=self.payload, timeout=30)
        else:
            response = requests.get(url, headers=headers, params=self.params, timeout=30)

        if response.status_code >= 400:
            raise TennisEVCalculatorError(
                f"Supabase {self.method} failed for {self.schema_name}.{self.table_name} "
                f"({response.status_code}): {response.text}"
            )
        if not response.text:
            return SupabaseResponse([])
        return SupabaseResponse(response.json())


class SupabaseSchemaClient:
    def __init__(self, base_url: str, service_role_key: str, schema_name: str) -> None:
        self.base_url = base_url
        self.service_role_key = service_role_key
        self.schema_name = schema_name

    def table(self, table_name: str) -> SupabaseQuery:
        return SupabaseQuery(self.base_url, self.service_role_key, self.schema_name, table_name)


class SupabaseRestClient:
    def __init__(self, base_url: str, service_role_key: str) -> None:
        self.base_url = base_url
        self.service_role_key = service_role_key

    def schema(self, schema_name: str) -> SupabaseSchemaClient:
        return SupabaseSchemaClient(self.base_url, self.service_role_key, schema_name)


def get_supabase_client() -> SupabaseRestClient:
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not supabase_key:
        raise TennisEVCalculatorError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment.")
    return SupabaseRestClient(supabase_url, supabase_key)


def normalize_name(name: object) -> str:
    text = "" if pd.isna(name) else str(name)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace(".", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text)


def name_score(api_name: object, sackmann_name: object) -> float:
    api = normalize_name(api_name)
    sackmann = normalize_name(sackmann_name)
    if not api or not sackmann:
        return 0.0
    if api == sackmann:
        return 1.0

    api_tokens = api.split()
    sackmann_tokens = sackmann.split()
    api_set = set(api_tokens)
    sackmann_set = set(sackmann_tokens)
    token_overlap = len(api_set & sackmann_set) / max(len(api_set | sackmann_set), 1)

    api_last = api_tokens[-1]
    sackmann_last = sackmann_tokens[-1]
    last_name_score = 0.86 if api_last == sackmann_last else 0.0

    initial_score = 0.0
    if len(api_tokens) == 2 and len(sackmann_tokens) >= 2:
        if api_tokens[0][:1] == sackmann_tokens[0][:1] and api_tokens[-1] == sackmann_tokens[-1]:
            initial_score = 0.95
        if api_tokens[-1][:1] == sackmann_tokens[0][:1] and api_tokens[0] == sackmann_tokens[-1]:
            initial_score = 0.95

    return max(token_overlap, last_name_score, initial_score)


def build_latest_player_state(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = features.copy()
    features["fecha"] = pd.to_datetime(features["fecha"], errors="coerce")
    features = features.sort_values("fecha")

    for _, row in features.iterrows():
        rows.append(
            {
                "fecha": row["fecha"],
                "player_name": row["jugador_a"],
                "serve_points_won_pct_last20": row["jugador_a_serve_points_won_pct_last20"],
                "return_points_won_pct_last20": row["jugador_a_return_points_won_pct_last20"],
                "dias_desde_ultimo_partido": row["jugador_a_dias_desde_ultimo_partido"],
                "returning_from_injury": row["jugador_a_returning_from_injury"],
            }
        )
        rows.append(
            {
                "fecha": row["fecha"],
                "player_name": row["jugador_b"],
                "serve_points_won_pct_last20": row["jugador_b_serve_points_won_pct_last20"],
                "return_points_won_pct_last20": row["jugador_b_return_points_won_pct_last20"],
                "dias_desde_ultimo_partido": row["jugador_b_dias_desde_ultimo_partido"],
                "returning_from_injury": row["jugador_b_returning_from_injury"],
            }
        )

    player_state = pd.DataFrame(rows)
    player_state["normalized_name"] = player_state["player_name"].map(normalize_name)
    return (
        player_state.sort_values("fecha")
        .drop_duplicates(subset=["normalized_name"], keep="last")
        .reset_index(drop=True)
    )


def find_player_state(api_name: str, latest_state: pd.DataFrame) -> tuple[pd.Series | None, float]:
    best_row: pd.Series | None = None
    best_score = 0.0
    for _, row in latest_state.iterrows():
        score = name_score(api_name, row["player_name"])
        if score > best_score:
            best_score = score
            best_row = row
    if best_score < MIN_NAME_SCORE:
        return None, best_score
    return best_row, best_score


def build_simulation_row(player_a: pd.Series, player_b: pd.Series) -> dict[str, Any]:
    return {
        "jugador_a_serve_points_won_pct_last20": player_a["serve_points_won_pct_last20"],
        "jugador_a_return_points_won_pct_last20": player_a["return_points_won_pct_last20"],
        "jugador_a_returning_from_injury": player_a["returning_from_injury"],
        "jugador_b_serve_points_won_pct_last20": player_b["serve_points_won_pct_last20"],
        "jugador_b_return_points_won_pct_last20": player_b["return_points_won_pct_last20"],
        "jugador_b_returning_from_injury": player_b["returning_from_injury"],
    }


def probability_from_distribution(distribution: dict[int, int], predicate: Any) -> float:
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    wins = sum(count for value, count in distribution.items() if predicate(int(value)))
    return wins / total


def calculate_ev(probability: float, decimal_odds: float) -> float:
    return probability * decimal_odds - 1.0


def insert_value_bets_to_supabase(value_bets: pd.DataFrame) -> int:
    if value_bets.empty or "EV" not in value_bets.columns:
        return 0

    positives = value_bets.loc[pd.to_numeric(value_bets["EV"], errors="coerce") > 0].copy()
    if positives.empty:
        return 0

    supabase = get_supabase_client()
    inserted = 0
    for _, row in positives.iterrows():
        partido = str(row["partido"])
        mercado = str(row["mercado"])
        linea = float(row["linea"])
        casa_apuestas = str(row["casa_apuestas"])
        cuota_actual = float(row["cuota_casa"])
        ev_actual = float(row["EV"])

        response = (
            supabase.schema("tennis")
            .table(TENNIS_PICKS_TABLE)
            .select("id")
            .eq("partido", partido)
            .eq("mercado", mercado)
            .eq("linea", linea)
            .eq("casa_apuestas", casa_apuestas)
            .execute()
        )

        if len(response.data or []) == 0:
            payload = {
                "partido": partido,
                "mercado": mercado,
                "linea": linea,
                "casa_apuestas": casa_apuestas,
                "cuota_inicial": cuota_actual,
                "nuestra_prob": float(row["nuestra_prob"]),
                "ev_inicial": ev_actual,
            }
            insert_response = (
                supabase.schema("tennis")
                .table(TENNIS_PICKS_TABLE)
                .insert(payload)
                .execute()
            )
            inserted_rows = insert_response.data or []
            if not inserted_rows or "id" not in inserted_rows[0]:
                raise TennisEVCalculatorError(f"Supabase did not return id for inserted pick: {partido}")
            pick_id = inserted_rows[0]["id"]
            print(f"[Supabase] Pick insertado: {mercado} {linea} {partido} ({casa_apuestas})")
            inserted += 1
        else:
            pick_id = response.data[0]["id"]

        latest_history = (
            supabase.schema("tennis")
            .table(PICK_ODDS_HISTORY_TABLE)
            .select("cuota")
            .eq("pick_id", pick_id)
            .order("fecha_registro", desc=True)
            .limit(1)
            .execute()
        )
        latest_rows = latest_history.data or []
        latest_cuota = float(latest_rows[0]["cuota"]) if latest_rows else None

        if latest_cuota is not None and abs(latest_cuota - cuota_actual) < 1e-9:
            print(f"[Supabase] La cuota se mantiene en {cuota_actual:.2f}, no se registra duplicado historico.")
            continue

        history_payload = {
            "pick_id": pick_id,
            "cuota": cuota_actual,
            "ev_momento": ev_actual,
        }
        (
            supabase.schema("tennis")
            .table(PICK_ODDS_HISTORY_TABLE)
            .insert(history_payload)
            .execute()
        )
        print(f"[Supabase] Historico actualizado: {mercado} {linea} {partido} cuota {cuota_actual:.2f}")
    return inserted


def evaluate_event(
    event_odds: pd.DataFrame,
    player_a_state: pd.Series,
    player_b_state: pd.Series,
    player_a_name: str,
    player_b_name: str,
    best_of: int,
    n_simulations: int,
) -> list[dict[str, Any]]:
    simulation_row = build_simulation_row(player_a_state, player_b_state)
    p_serve_a, p_serve_b = calculate_match_serve_probs(simulation_row)
    simulation = run_monte_carlo(p_serve_a, p_serve_b, best_of=best_of, n_simulations=n_simulations)
    total_distribution = simulation["total_games_distribution"]
    diff_distribution = simulation["game_diff_distribution"]
    rows: list[dict[str, Any]] = []

    for _, odds_row in event_odds.iterrows():
        market = str(odds_row["market"])
        selection = str(odds_row["selection"])
        line = odds_row["line"]
        odds = float(odds_row["odds"])
        probability: float | None = None
        market_label: str | None = None

        if market == "totals" and pd.notna(line):
            line_value = float(line)
            if selection.lower() == "over":
                probability = probability_from_distribution(total_distribution, lambda total: total > line_value)
                market_label = "Over"
            elif selection.lower() == "under":
                probability = probability_from_distribution(total_distribution, lambda total: total < line_value)
                market_label = "Under"

        elif market == "spreads" and pd.notna(line):
            line_value = float(line)
            if name_score(selection, player_a_name) >= name_score(selection, player_b_name):
                probability = probability_from_distribution(diff_distribution, lambda diff: diff > -line_value)
                market_label = f"Handicap {player_a_name}"
            else:
                probability = probability_from_distribution(diff_distribution, lambda diff: diff < line_value)
                market_label = f"Handicap {player_b_name}"

        if probability is None or market_label is None:
            continue

        rows.append(
            {
                "partido": f"{player_a_name} vs {player_b_name}",
                "mercado": market_label,
                "linea": float(line),
                "casa_apuestas": odds_row.get("casa_apuestas") or odds_row.get("bookmaker_title"),
                "cuota_casa": odds,
                "nuestra_prob": probability,
                "EV": calculate_ev(probability, odds),
                "p_serve_A": p_serve_a,
                "p_serve_B": p_serve_b,
                "mean_total_games": simulation["mean_total_games"],
                "mean_game_diff": simulation["mean_game_diff"],
            }
        )

    return rows


def calculate_value_bets(
    features_path: Path = FEATURES_PATH,
    output_path: Path = OUTPUT_PATH,
    best_of: int = 3,
    n_simulations: int = 1000,
) -> pd.DataFrame:
    if not features_path.exists():
        raise TennisEVCalculatorError(f"Features file not found: {features_path}")

    random.seed(42)
    odds = fetch_and_save_odds(ODDS_PATH)
    features = pd.read_csv(features_path)
    latest_state = build_latest_player_state(features)

    output_rows: list[dict[str, Any]] = []
    event_columns = ["event_id", "jugador_a_api", "jugador_b_api"]
    for _, event in odds[event_columns].drop_duplicates().iterrows():
        player_a_state, score_a = find_player_state(str(event["jugador_a_api"]), latest_state)
        player_b_state, score_b = find_player_state(str(event["jugador_b_api"]), latest_state)
        if player_a_state is None or player_b_state is None:
            print(
                "Skipping event due to low name match:",
                event["jugador_a_api"],
                score_a,
                event["jugador_b_api"],
                score_b,
            )
            continue

        event_odds = odds.loc[odds["event_id"].eq(event["event_id"])]
        output_rows.extend(
            evaluate_event(
                event_odds=event_odds,
                player_a_state=player_a_state,
                player_b_state=player_b_state,
                player_a_name=str(player_a_state["player_name"]),
                player_b_name=str(player_b_state["player_name"]),
                best_of=best_of,
                n_simulations=n_simulations,
            )
        )

    output = pd.DataFrame(output_rows)
    if not output.empty:
        output["Kelly Recomendado"] = 1000.0 * output["EV"] / (output["cuota_casa"] - 1.0)
        final_columns = [
            "partido",
            "mercado",
            "linea",
            "casa_apuestas",
            "cuota_casa",
            "nuestra_prob",
            "EV",
            "Kelly Recomendado",
            "p_serve_A",
            "p_serve_B",
            "mean_total_games",
            "mean_game_diff",
        ]
        output = output.sort_values("cuota_casa", ascending=False)
        output = output.drop_duplicates(subset=["partido", "mercado", "linea"], keep="first")
        output = output.sort_values("EV", ascending=False).reset_index(drop=True)
        output = output[[column for column in final_columns if column in output.columns]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    inserted = insert_value_bets_to_supabase(output)
    print(f"[Supabase] Total picks EV+ insertados: {inserted}")
    return output


if __name__ == "__main__":
    value_bets = calculate_value_bets()
    print(f"Saved {len(value_bets)} rows to {OUTPUT_PATH}")
    if value_bets.empty:
        print("No value-bet rows generated.")
    else:
        print(value_bets[["partido", "mercado", "linea", "casa_apuestas", "cuota_casa", "nuestra_prob", "EV", "Kelly Recomendado"]].to_string(index=False))
