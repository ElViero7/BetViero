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
    from simulator import calculate_match_serve_probs, run_monte_carlo, clamp_probability
    from odds_listener import run as fetch_and_save_odds
except ModuleNotFoundError:
    from tennis.src.simulator import calculate_match_serve_probs, run_monte_carlo, clamp_probability
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
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_KEY", "").strip()
    if not supabase_url or not supabase_key:
        raise TennisEVCalculatorError("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY in environment.")
    return SupabaseRestClient(supabase_url, supabase_key)


def get_player_id_from_mapping(api_name: str) -> int | None:
    """
    Busca el ID numérico de Sackmann para un jugador en la tabla player_mappings de Supabase.
    """
    try:
        supabase = get_supabase_client()
        response = (
            supabase.schema("tennis")
            .table("player_mappings")
            .select("sackmann_player_id")
            .eq("api_name", api_name)
            .execute()
        )
        if response.data and len(response.data) > 0:
            return int(response.data[0]["sackmann_player_id"])
    except Exception as exc:
        print(f"[Supabase] Error al consultar player_mappings para {api_name}: {exc}")
    return None


def save_player_mapping(api_name: str, sackmann_player_id: int) -> None:
    """
    Guarda la asociación entre el nombre de la API y el ID de Sackmann de forma persistente.
    """
    try:
        supabase = get_supabase_client()
        payload = {
            "api_name": api_name,
            "sackmann_player_id": int(sackmann_player_id),
        }
        supabase.schema("tennis").table("player_mappings").insert(payload).execute()
        print(f"[Supabase] Nuevo mapeo guardado: {api_name} -> {sackmann_player_id}")
    except Exception as exc:
        print(f"[Supabase] Error al guardar player_mappings para {api_name} -> {sackmann_player_id}: {exc}")


def first_name_conflict(left_text: str, right_text: str) -> bool:
    """
    Verifica si hay un conflicto en el primer nombre o las iniciales del nombre de los jugadores.
    Evita falsas coincidencias entre familiares con el mismo apellido.
    """
    left_tokens = left_text.split()
    right_tokens = right_text.split()
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    f1 = left_tokens[0]
    f2 = right_tokens[0]
    if f1[0] != f2[0]:
        return True
    if len(f1) > 1 and len(f2) > 1:
        if f1 != f2 and not f1.startswith(f2) and not f2.startswith(f1):
            return True
    return False


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
    if first_name_conflict(api, sackmann):
        return 0.0

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


def infer_surface(commence_time_str: str) -> str:
    """
    Infiere la superficie basándose en la fecha del evento según la gira de la ATP.
    Retorna: 'Hard', 'Clay' o 'Grass'.
    """
    try:
        dt = pd.to_datetime(commence_time_str)
        month = dt.month
        day = dt.day

        # Temporada de hierba (Grass): Junio entero y primera quincena de Julio
        if month == 6 or (month == 7 and day <= 15):
            return "Grass"
        # Gira europea de arcilla y Roland Garros: Abril (después del día 10), Mayo completo
        elif (month == 4 and day >= 10) or month == 5:
            return "Clay"
        else:
            return "Hard"
    except Exception:
        return "Hard"


def calibrate_serve_probs_to_market(
    p_serve_a: float,
    p_serve_b: float,
    best_of: int,
    market_prob_a: float,
    steps: int = 9,
) -> tuple[float, float]:
    """
    Calibra p_serve_a y p_serve_b de forma que la probabilidad de victoria simulada para A
    sea igual a market_prob_a. Utiliza búsqueda binaria de `steps` pasos sobre delta en [-0.25, 0.25].
    """
    low = -0.25
    high = 0.25

    best_p_a = p_serve_a
    best_p_b = p_serve_b

    for _ in range(steps):
        mid = (low + high) / 2.0
        p_a_adjusted = clamp_probability(p_serve_a + mid)
        p_b_adjusted = clamp_probability(p_serve_b - mid)

        sim = run_monte_carlo(p_a_adjusted, p_b_adjusted, best_of=best_of, n_simulations=5000)
        prob_a = sim["win_probability_A"]

        if prob_a < market_prob_a:
            low = mid
        else:
            high = mid

        best_p_a = p_a_adjusted
        best_p_b = p_b_adjusted

    return best_p_a, best_p_b


def build_latest_player_state(
    features: pd.DataFrame,
    surface: str | None = None,
    *,
    by_surface: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = features.copy()
    features["fecha"] = pd.to_datetime(features["fecha"], errors="coerce")
    if surface is not None and "surface" in features.columns:
        features = features.loc[features["surface"].astype(str).eq(surface)].copy()
    features = features.sort_values("fecha")

    for _, row in features.iterrows():
        rows.append(
            {
                "fecha": row["fecha"],
                "surface": row.get("surface"),
                "player_id": int(row["jugador_a_id"]),
                "player_name": row["jugador_a"],
                "serve_points_won_pct_last20": row["jugador_a_serve_points_won_pct_last20"],
                "return_points_won_pct_last20": row["jugador_a_return_points_won_pct_last20"],
                "dias_desde_ultimo_partido": row["jugador_a_dias_desde_ultimo_partido"],
                "returning_from_injury": row["jugador_a_returning_from_injury"],
                "minutos_acumulados_torneo": row.get("jugador_a_minutos_acumulados_torneo", 0),
            }
        )
        rows.append(
            {
                "fecha": row["fecha"],
                "surface": row.get("surface"),
                "player_id": int(row["jugador_b_id"]),
                "player_name": row["jugador_b"],
                "serve_points_won_pct_last20": row["jugador_b_serve_points_won_pct_last20"],
                "return_points_won_pct_last20": row["jugador_b_return_points_won_pct_last20"],
                "dias_desde_ultimo_partido": row["jugador_b_dias_desde_ultimo_partido"],
                "returning_from_injury": row["jugador_b_returning_from_injury"],
                "minutos_acumulados_torneo": row.get("jugador_b_minutos_acumulados_torneo", 0),
            }
        )

    player_state = pd.DataFrame(rows)
    if player_state.empty:
        return player_state
    player_state["normalized_name"] = player_state["player_name"].map(normalize_name)
    subset = ["player_id", "surface"] if by_surface and "surface" in player_state.columns else ["player_id"]
    return (
        player_state.sort_values("fecha")
        .drop_duplicates(subset=subset, keep="last")
        .reset_index(drop=True)
    )


def find_player_state_by_api_name(
    api_name: str,
    latest_state: pd.DataFrame
) -> tuple[pd.Series | None, float, int | None]:
    """
    Busca el estado de un jugador de la API en latest_state usando primero
    el mapeo persistente de Supabase, y si no existe, la coincidencia por nombres.
    Si se encuentra vía nombres, se guarda el mapeo automáticamente en Supabase.
    """
    # 1. Intentar buscar por mapeo de base de datos
    mapped_id = get_player_id_from_mapping(api_name)
    if mapped_id is not None:
        state_rows = latest_state.loc[latest_state["player_id"] == mapped_id]
        if not state_rows.empty:
            return state_rows.iloc[0], 1.0, mapped_id

    # 2. Si no hay mapeo, buscar por coincidencia de nombre
    best_row: pd.Series | None = None
    best_score = 0.0
    for _, row in latest_state.iterrows():
        score = name_score(api_name, row["player_name"])
        if score > best_score:
            best_score = score
            best_row = row

    if best_score < MIN_NAME_SCORE or best_row is None:
        return None, best_score, None

    # Encontrado por coincidencia de nombre. Guardar mapeo en Supabase para el futuro.
    sackmann_player_id = int(best_row["player_id"])
    save_player_mapping(api_name, sackmann_player_id)
    return best_row, best_score, sackmann_player_id


def build_simulation_row(
    player_a: pd.Series,
    player_b: pd.Series,
    surface: str = "Hard",
    best_of: int = 3,
) -> dict[str, Any]:
    return {
        "jugador_a_serve_points_won_pct_last20": player_a["serve_points_won_pct_last20"],
        "jugador_a_return_points_won_pct_last20": player_a["return_points_won_pct_last20"],
        "jugador_a_returning_from_injury": player_a["returning_from_injury"],
        "jugador_a_dias_desde_ultimo_partido": player_a.get("dias_desde_ultimo_partido"),
        "jugador_a_minutos_acumulados_torneo": player_a.get("minutos_acumulados_torneo"),
        "jugador_b_serve_points_won_pct_last20": player_b["serve_points_won_pct_last20"],
        "jugador_b_return_points_won_pct_last20": player_b["return_points_won_pct_last20"],
        "jugador_b_returning_from_injury": player_b["returning_from_injury"],
        "jugador_b_dias_desde_ultimo_partido": player_b.get("dias_desde_ultimo_partido"),
        "jugador_b_minutos_acumulados_torneo": player_b.get("minutos_acumulados_torneo"),
        "surface": surface,
        "best_of": best_of,
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
        revisar_manualmente = bool(row.get("revisar_manualmente", False))

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
                "revisar_manualmente": revisar_manualmente,
            }
            try:
                insert_response = (
                    supabase.schema("tennis")
                    .table(TENNIS_PICKS_TABLE)
                    .insert(payload)
                    .execute()
                )
            except Exception as exc:
                if "revisar_manualmente" in str(exc):
                    print("[Supabase] Advertencia: la columna 'revisar_manualmente' no existe en la tabla tennis.tennis_picks.")
                    print("[Supabase] Por favor, ejecuta el script 'tennis/supabase/add_revisar_manualmente_to_picks.sql' en tu editor SQL de Supabase.")
                    print("[Supabase] Reintentando inserción sin la columna 'revisar_manualmente'...")
                    payload_fallback = payload.copy()
                    payload_fallback.pop("revisar_manualmente", None)
                    insert_response = (
                        supabase.schema("tennis")
                        .table(TENNIS_PICKS_TABLE)
                        .insert(payload_fallback)
                        .execute()
                    )
                else:
                    raise

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
    commence_time: str,
    h2h_calibrated_serve_probs: tuple[float, float] | None,
    favorabilidad_invertida: bool,
) -> list[dict[str, Any]]:
    # Usar las probabilidades calibradas contra el mercado si están disponibles; si no, calcular por defecto.
    if h2h_calibrated_serve_probs is not None:
        p_serve_a, p_serve_b = h2h_calibrated_serve_probs
    else:
        surface = infer_surface(commence_time)
        simulation_row = build_simulation_row(player_a_state, player_b_state, surface=surface, best_of=best_of)
        p_serve_a, p_serve_b = calculate_match_serve_probs(simulation_row)

    # Simulación Monte Carlo de 10,000 partidos
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

        if market == "spreads" and pd.notna(line):
            line_value = float(line)
            if abs(line_value) > 8.0:
                continue
            if name_score(selection, player_a_name) >= name_score(selection, player_b_name):
                probability = probability_from_distribution(diff_distribution, lambda diff: diff > -line_value)
                market_label = f"Handicap {player_a_name}"
            else:
                probability = probability_from_distribution(diff_distribution, lambda diff: diff < line_value)
                market_label = f"Handicap {player_b_name}"
        else:
            continue

        if probability is None or market_label is None:
            continue

        raw_ev = calculate_ev(probability, odds)
        revisar_manualmente = False

        # Logger detallado de EV anómalo
        if raw_ev > 0.30:
            revisar_manualmente = True
            prob_impl = 1.0 / odds if odds > 0 else 0.0
            print(f"\n[ADVERTENCIA: EV ANÓMALO DETECTADO (>30%)]")
            print(f"--------------------------------------------------")
            print(f"Partido: {player_a_name} vs {player_b_name}")
            print(f"Mercado: {market_label}")
            print(f"Línea: {line if pd.notna(line) else 'N/A'}")
            print(f"Cuota Casa: {odds:.2f} (Prob. Implícita: {prob_impl:.2%})")
            print(f"Nuestra Prob: {probability:.2%}")
            print(f"EV Crudo: {raw_ev:.2%}")
            print(f"Favorabilidad Invertida: {'SÍ' if favorabilidad_invertida else 'NO'}")
            print(f"Estadísticas de Saque Calibradas:")
            print(f"  - Jugador A (Serve): {p_serve_a:.4f}")
            print(f"  - Jugador B (Serve): {p_serve_b:.4f}")
            print(f"  - Media Juegos: {simulation['mean_total_games']:.2f}")
            print(f"  - Media Diferencia: {simulation['mean_game_diff']:.2f}")
            print(f"--------------------------------------------------")
            print(f"[INFO] Se aplicó capping de EV al 30% como medida de seguridad.\n")
            ev_final = 0.30
        else:
            ev_final = raw_ev

        rows.append(
            {
                "partido": f"{player_a_name} vs {player_b_name}",
                "mercado": market_label,
                "linea": float(line) if pd.notna(line) else 0.0,
                "casa_apuestas": odds_row.get("casa_apuestas") or odds_row.get("bookmaker_title"),
                "cuota_casa": odds,
                "nuestra_prob": probability,
                "EV": ev_final,
                "ev_crudo": raw_ev,
                "revisar_manualmente": revisar_manualmente,
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
    n_simulations: int = 10000,
) -> pd.DataFrame:
    if not features_path.exists():
        raise TennisEVCalculatorError(f"Features file not found: {features_path}")

    random.seed(42)
    # 1. Obtener y guardar cuotas
    odds = fetch_and_save_odds(ODDS_PATH)
    if odds.empty:
        print("No market odds fetched. Exiting.")
        return pd.DataFrame()

    features = pd.read_csv(features_path)

    output_rows: list[dict[str, Any]] = []
    event_columns = ["event_id", "jugador_a_api", "jugador_b_api", "commence_time"]
    events_to_process = odds[event_columns].drop_duplicates().copy()

    for _, event in events_to_process.iterrows():
        event_id = str(event["event_id"])
        api_a = str(event["jugador_a_api"])
        api_b = str(event["jugador_b_api"])
        commence_time = str(event["commence_time"])

        # Inferir superficie basada en la fecha del evento
        surface = infer_surface(commence_time)

        # Generar latest_state filtrado por la superficie del evento
        latest_state = build_latest_player_state(features, surface=surface)
        if latest_state.empty:
            # Fallback a sin filtro de superficie
            latest_state = build_latest_player_state(features)

        player_a_state, score_a, id_a = find_player_state_by_api_name(api_a, latest_state)
        player_b_state, score_b, id_b = find_player_state_by_api_name(api_b, latest_state)

        if player_a_state is None or player_b_state is None:
            print(
                f"Skipping event due to low name match: "
                f"{api_a} (score: {score_a:.2f}) vs {api_b} (score: {score_b:.2f})"
            )
            continue

        player_a_name = str(player_a_state["player_name"])
        player_b_name = str(player_b_state["player_name"])

        # Calcular probs iniciales (antes de calibrar contra el H2H)
        simulation_row = build_simulation_row(player_a_state, player_b_state, surface=surface, best_of=best_of)
        p_serve_a_init, p_serve_b_init = calculate_match_serve_probs(simulation_row)

        # Simular modelo inicial para determinar el favorito inicial
        sim_init = run_monte_carlo(p_serve_a_init, p_serve_b_init, best_of=best_of, n_simulations=1000)
        win_prob_a_init = sim_init["win_probability_A"]
        win_prob_b_init = sim_init["win_probability_B"]

        # Buscar cuotas H2H para calibración (Pinnacle primero, cualquier casa después)
        event_odds = odds.loc[odds["event_id"].eq(event_id)]
        pinnacle_h2h = event_odds.loc[
            event_odds["market"].eq("h2h") &
            event_odds["bookmaker_key"].eq("pinnacle")
        ]

        cuota_h2h_A = None
        cuota_h2h_B = None

        if len(pinnacle_h2h) >= 2:
            for _, row in pinnacle_h2h.iterrows():
                sel = row["selection"]
                score_sel_a = name_score(sel, player_a_name)
                score_sel_b = name_score(sel, player_b_name)
                if score_sel_a > score_sel_b:
                    cuota_h2h_A = float(row["odds"])
                else:
                    cuota_h2h_B = float(row["odds"])

        # Fallback a otra casa de apuestas si Pinnacle H2H no está disponible
        if cuota_h2h_A is None or cuota_h2h_B is None:
            any_h2h = event_odds.loc[event_odds["market"].eq("h2h")]
            if not any_h2h.empty:
                for _, grp in any_h2h.groupby("bookmaker_key"):
                    if len(grp) >= 2:
                        temp_a, temp_b = None, None
                        for _, row in grp.iterrows():
                            sel = row["selection"]
                            score_sel_a = name_score(sel, player_a_name)
                            score_sel_b = name_score(sel, player_b_name)
                            if score_sel_a > score_sel_b:
                                temp_a = float(row["odds"])
                            else:
                                temp_b = float(row["odds"])
                        if temp_a is not None and temp_b is not None:
                            cuota_h2h_A = temp_a
                            cuota_h2h_B = temp_b
                            break

        h2h_calibrated_serve_probs = None
        favorabilidad_invertida = False

        if cuota_h2h_A is not None and cuota_h2h_B is not None:
            # 1. Quitar overround
            impl_A = 1.0 / cuota_h2h_A
            impl_B = 1.0 / cuota_h2h_B
            market_prob_a = impl_A / (impl_A + impl_B)

            # 2. Calibración H2H por búsqueda binaria
            h2h_calibrated_serve_probs = calibrate_serve_probs_to_market(
                p_serve_a_init,
                p_serve_b_init,
                best_of,
                market_prob_a,
                steps=9
            )

            # 3. Detectar si la favorabilidad está invertida
            pinnacle_favors_A = cuota_h2h_A < cuota_h2h_B
            model_favors_A = win_prob_a_init > win_prob_b_init
            favorabilidad_invertida = pinnacle_favors_A != model_favors_A

            print(
                f"[Calibración] {player_a_name} vs {player_b_name} | "
                f"Cuotas: {cuota_h2h_A:.2f} - {cuota_h2h_B:.2f} | "
                f"Prob. Mercado: {market_prob_a:.2%} | "
                f"Ajuste Saques: {p_serve_a_init:.4f}/{p_serve_b_init:.4f} -> "
                f"{h2h_calibrated_serve_probs[0]:.4f}/{h2h_calibrated_serve_probs[1]:.4f}"
            )

        # Evaluar el evento completo usando los saques calibrados
        event_picks = evaluate_event(
            event_odds=event_odds,
            player_a_state=player_a_state,
            player_b_state=player_b_state,
            player_a_name=player_a_name,
            player_b_name=player_b_name,
            best_of=best_of,
            n_simulations=n_simulations,
            commence_time=commence_time,
            h2h_calibrated_serve_probs=h2h_calibrated_serve_probs,
            favorabilidad_invertida=favorabilidad_invertida,
        )
        output_rows.extend(event_picks)

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
            "ev_crudo",
            "revisar_manualmente",
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
        print(value_bets[["partido", "mercado", "linea", "casa_apuestas", "cuota_casa", "nuestra_prob", "EV", "ev_crudo", "Kelly Recomendado"]].to_string(index=False))
