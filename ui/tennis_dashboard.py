from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from tennis.src.db import update_tennis_row
from tennis.src.ev_calculator import get_supabase_client


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENNIS_DIR = PROJECT_ROOT / "tennis"

DISABLED_LOCAL_PROXY_VALUES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
}
for proxy_env_name in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    if os.environ.get(proxy_env_name) in DISABLED_LOCAL_PROXY_VALUES:
        os.environ.pop(proxy_env_name, None)

TENNIS_VALUE_BETS_PATH = TENNIS_DIR / "data" / "predictions" / "upcoming_value_bets_tennis.csv"
TENNIS_SACKMANN_PATH = TENNIS_DIR / "data" / "processed" / "sackmann_atp_clean.csv"
TENNIS_FEATURES_PATH = TENNIS_DIR / "data" / "processed" / "atp_features_avanzadas.csv"
TENNIS_SECRET_KEYS = [
    "ODDS_API_KEY",
    "THE_ODDS_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_KEY",
    "THE_ODDS_API_BASE_URL",
    "THE_ODDS_API_TIMEOUT_SECONDS",
    "TENNIS_ODDS_SPORT_KEY",
    "TENNIS_ODDS_REGIONS",
    "TENNIS_ODDS_MARKETS",
    "TENNIS_ODDS_FORMAT",
]


def get_streamlit_secret(key: str) -> str | None:
    try:
        value = st.secrets[key]
    except Exception:
        return None
    return str(value) if value is not None else None


def build_tennis_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in TENNIS_SECRET_KEYS:
        secret_value = get_streamlit_secret(key)
        if secret_value:
            env[key] = secret_value
    return env


def run_tennis_script(script_name: str, *args: str, timeout_seconds: int = 300) -> subprocess.CompletedProcess[str]:
    script_path = TENNIS_DIR / "src" / script_name
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=PROJECT_ROOT,
        env=build_tennis_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


@st.cache_data(ttl=30)
def load_tennis_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def style_tennis_value_rows(row: pd.Series) -> list[str]:
    ev_value = pd.to_numeric(row.get("EV Actual", row.get("EV", 0)), errors="coerce")
    if pd.notna(ev_value) and float(ev_value) > 0:
        return ["background-color: #d8f5df; color: #102a18;" for _ in row]
    return ["" for _ in row]


def add_tennis_kelly_column(frame: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    if frame.empty or not {"EV", "cuota_casa"}.issubset(frame.columns):
        return frame
    output = frame.copy()
    ev_values = pd.to_numeric(output["EV"], errors="coerce")
    odds_values = pd.to_numeric(output["cuota_casa"], errors="coerce")
    denominator = odds_values - 1
    output["Kelly Recomendado"] = (bankroll * ev_values / denominator).where(denominator > 0)
    return output


def tennis_file_status(path: Path) -> dict[str, object]:
    status: dict[str, object] = {
        "archivo": str(path.relative_to(PROJECT_ROOT)),
        "existe": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if path.exists() and path.suffix.lower() == ".csv":
        try:
            status["filas"] = len(pd.read_csv(path))
        except Exception as exc:
            status["filas"] = f"Error leyendo CSV: {exc}"
    return status


def render_tennis_diagnostics() -> None:
    with st.expander("Diagnostico Tenis"):
        secret_status = {
            key: ("configurado" if get_streamlit_secret(key) or os.getenv(key) else "faltante")
            for key in ["ODDS_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_KEY"]
        }
        st.write("Secrets / entorno")
        st.json(secret_status)

        st.write("Archivos")
        st.dataframe(
            pd.DataFrame(
                [
                    tennis_file_status(TENNIS_FEATURES_PATH),
                    tennis_file_status(TENNIS_VALUE_BETS_PATH),
                    tennis_file_status(TENNIS_SACKMANN_PATH),
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        result = st.session_state.get("last_tennis_script_result")
        if not result:
            st.info("Todavia no se ha ejecutado ningun script en esta sesion.")
            return

        st.write(f"Ultimo script: `{result['script']}` | returncode: `{result['returncode']}`")
        if result.get("stdout"):
            st.write("stdout")
            st.code(result["stdout"])
        if result.get("stderr"):
            st.write("stderr")
            st.code(result["stderr"])


@st.cache_data(ttl=30)
def load_tennis_picks_history() -> pd.DataFrame:
    response = (
        get_supabase_client()
        .schema("tennis")
        .table("tennis_picks")
        .select("*")
        .order("fecha_registro", desc=True)
        .execute()
    )
    return pd.DataFrame(response.data or [])


@st.cache_data(ttl=30)
def load_tennis_pick_odds_history() -> pd.DataFrame:
    response = (
        get_supabase_client()
        .schema("tennis")
        .table("pick_odds_history")
        .select("*")
        .order("fecha_registro", desc=False)
        .execute()
    )
    return pd.DataFrame(response.data or [])


@st.cache_data(ttl=30)
def load_tennis_upcoming_odds() -> pd.DataFrame:
    response = (
        get_supabase_client()
        .schema("tennis")
        .table("upcoming_odds")
        .select("*")
        .order("commence_time", desc=False)
        .execute()
    )
    frame = pd.DataFrame(response.data or [])
    if frame.empty:
        return frame

    for column in ["price", "implied_probability", "line"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["commence_time", "collected_at", "created_at", "updated_at"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["match_label"] = frame["player_1"].fillna("") + " vs " + frame["player_2"].fillna("")
    frame["line_label"] = frame["line"].map(lambda value: "-" if pd.isna(value) else f"{value:g}")
    frame["market_label"] = (
        frame["market_key"].fillna("")
        + " | "
        + frame["selection"].fillna("")
        + " | "
        + frame["line_label"]
    )
    return frame


@st.cache_data(ttl=30)
def load_tennis_bets_log() -> pd.DataFrame:
    response = (
        get_supabase_client()
        .schema("tennis")
        .table("bets_log")
        .select("*")
        .order("placed_at", desc=True)
        .execute()
    )
    frame = pd.DataFrame(response.data or [])
    if frame.empty:
        return frame

    for column in ["line", "my_odd", "model_probability", "model_fair_odd", "edge", "ev", "stake", "kelly_fraction", "pnl"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["placed_at", "event_time", "settled_at", "created_at", "updated_at"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["partido"] = frame["player_1"].fillna("") + " vs " + frame["player_2"].fillna("")
    frame["line_label"] = frame["line"].map(lambda value: "-" if pd.isna(value) else f"{value:g}")
    frame["bet_label"] = (
        frame["placed_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("-")
        + " | "
        + frame["partido"].fillna("-")
        + " | "
        + frame["market_key"].fillna("-")
        + " "
        + frame["selection"].fillna("-")
        + " "
        + frame["line_label"]
        + " | "
        + frame["my_odd"].map(lambda value: "-" if pd.isna(value) else f"{value:.2f}")
    )
    return frame


def settle_tennis_bet(bet_id: str, status: str, result_selection: str | None = None) -> None:
    bets = load_tennis_bets_log()
    if bets.empty:
        raise ValueError("No hay apuestas cargadas.")

    row = bets.loc[bets["id"].eq(bet_id)]
    if row.empty:
        raise ValueError("No encontre la apuesta seleccionada.")

    bet = row.iloc[0]
    stake = float(bet["stake"])
    odd = float(bet["my_odd"])
    if status == "won":
        pnl = stake * (odd - 1.0)
    elif status == "lost":
        pnl = -stake
    elif status == "void":
        pnl = 0.0
    else:
        raise ValueError(f"Estado no soportado: {status}")

    payload = {
        "status": status,
        "result_selection": result_selection or None,
        "pnl": pnl,
        "settled_at": pd.Timestamp.utcnow().isoformat(),
    }
    update_tennis_row("bets_log", bet_id, payload)


def build_tennis_best_market_odds(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    sort_columns = ["external_event_id", "market_key", "selection", "line", "price"]
    best_rows = (
        frame.sort_values(sort_columns, ascending=[True, True, True, True, False], na_position="first")
        .drop_duplicates(subset=["external_event_id", "market_key", "selection", "line"], keep="first")
        .copy()
    )
    grouped = frame.groupby(["external_event_id", "market_key", "selection", "line"], dropna=False)
    summary = grouped.agg(
        casas=("bookmaker_key", "nunique"),
        min_cuota=("price", "min"),
        max_cuota=("price", "max"),
        ultima_actualizacion=("collected_at", "max"),
    ).reset_index()

    output = best_rows.merge(
        summary,
        on=["external_event_id", "market_key", "selection", "line"],
        how="left",
        suffixes=("", "_summary"),
    )
    output["rango_cuotas"] = output["max_cuota"] - output["min_cuota"]
    return output.sort_values(["commence_time", "match_label", "market_key", "selection", "line"]).reset_index(drop=True)


def parse_tennis_value_market(mercado: object) -> tuple[str | None, str | None]:
    market_text = "" if pd.isna(mercado) else str(mercado).strip()
    lowered = market_text.lower()
    if lowered == "over":
        return "totals", "Over"
    if lowered == "under":
        return "totals", "Under"
    if lowered.startswith("handicap "):
        return "spreads", market_text[len("Handicap ") :].strip()
    if market_text:
        return "h2h", market_text
    return None, None


def enrich_tennis_value_bets_with_market(value_bets: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    if value_bets.empty:
        return value_bets

    output = value_bets.copy()
    if "linea" in output.columns:
        output["linea"] = pd.to_numeric(output["linea"], errors="coerce")
    if "nuestra_prob" in output.columns:
        output["nuestra_prob"] = pd.to_numeric(output["nuestra_prob"], errors="coerce")

    parsed_markets = output["mercado"].map(parse_tennis_value_market)
    output["market_key_actual"] = parsed_markets.map(lambda value: value[0])
    output["selection_actual"] = parsed_markets.map(lambda value: value[1])

    try:
        picks = load_tennis_picks_history()
    except Exception:
        picks = pd.DataFrame()
    if not picks.empty:
        picks = picks.copy()
        picks["linea"] = pd.to_numeric(picks["linea"], errors="coerce")
        picks["fecha_registro"] = pd.to_datetime(picks["fecha_registro"], errors="coerce")
        latest_picks = (
            picks.sort_values("fecha_registro", ascending=False)
            .drop_duplicates(subset=["partido", "mercado", "linea"], keep="first")
            [["id", "partido", "mercado", "linea", "fecha_registro"]]
            .rename(columns={"id": "tennis_pick_id", "fecha_registro": "tennis_pick_fecha"})
        )
        output = output.merge(
            latest_picks,
            on=["partido", "mercado", "linea"],
            how="left",
        )

    try:
        market_odds = load_tennis_upcoming_odds()
    except Exception:
        return output
    if market_odds.empty:
        return output

    best_odds = build_tennis_best_market_odds(market_odds)
    if best_odds.empty:
        return output

    join_columns_left = ["partido", "market_key_actual", "selection_actual", "linea"]
    join_columns_right = ["match_label", "market_key", "selection", "line"]
    current_odds = best_odds[
        [
            "match_label",
            "market_key",
            "selection",
            "line",
            "price",
            "bookmaker_key",
            "bookmaker_title",
            "casas",
            "min_cuota",
            "max_cuota",
            "rango_cuotas",
            "ultima_actualizacion",
        ]
    ].rename(
        columns={
            "price": "mejor_cuota_actual",
            "bookmaker_key": "mejor_bookmaker_key_actual",
            "bookmaker_title": "mejor_casa_actual",
            "casas": "casas_actuales",
            "min_cuota": "min_cuota_actual",
            "max_cuota": "max_cuota_actual",
            "rango_cuotas": "rango_cuotas_actual",
            "ultima_actualizacion": "ultima_actualizacion_cuota",
        }
    )

    output = output.merge(
        current_odds,
        left_on=join_columns_left,
        right_on=join_columns_right,
        how="left",
    )
    output["delta_cuota_actual"] = output["mejor_cuota_actual"] - pd.to_numeric(output.get("cuota_casa"), errors="coerce")
    output["EV Actual"] = output["nuestra_prob"] * output["mejor_cuota_actual"] - 1.0
    denominator = output["mejor_cuota_actual"] - 1.0
    output["Kelly Actual"] = (bankroll * output["EV Actual"] / denominator).where(denominator > 0)
    return output


def split_tennis_match_name(match_name: object) -> tuple[str, str]:
    text = "" if pd.isna(match_name) else str(match_name)
    if " vs " not in text:
        return text.strip(), ""
    player_1, player_2 = text.split(" vs ", 1)
    return player_1.strip(), player_2.strip()


def calculate_tennis_kelly_values(probability: float, decimal_odd: float, bankroll: float, multiplier: float) -> tuple[float, float, float]:
    ev = probability * decimal_odd - 1.0
    full_kelly_fraction = ev / (decimal_odd - 1.0) if decimal_odd > 1.0 else 0.0
    applied_fraction = max(0.0, full_kelly_fraction * multiplier)
    stake = bankroll * applied_fraction
    return ev, applied_fraction, stake


def insert_tennis_bet_log(row: pd.Series, *, bet_type: str, odd: float, stake: float, kelly_fraction: float, notes: str) -> None:
    player_1, player_2 = split_tennis_match_name(row.get("partido"))
    market_key, selection = parse_tennis_value_market(row.get("mercado"))
    probability = float(row.get("nuestra_prob"))
    ev = probability * odd - 1.0
    payload = {
        "tennis_pick_id": None if pd.isna(row.get("tennis_pick_id")) else str(row.get("tennis_pick_id")),
        "source": "streamlit_value_bet",
        "bet_type": bet_type,
        "player_1": player_1,
        "player_2": player_2,
        "market_key": market_key or str(row.get("mercado") or ""),
        "selection": selection or str(row.get("mercado") or ""),
        "line": None if pd.isna(row.get("linea")) else float(row.get("linea")),
        "bookmaker_key": str(row.get("mejor_bookmaker_key_actual") or row.get("mejor_casa_actual") or row.get("casa_apuestas") or ""),
        "my_odd": float(odd),
        "model_probability": probability,
        "model_fair_odd": 1.0 / probability if probability > 0 else None,
        "edge": probability - (1.0 / odd),
        "ev": ev,
        "stake": float(stake),
        "kelly_fraction": float(kelly_fraction),
        "status": "pending",
        "notes": notes.strip() or None,
        "raw_payload": {
            "partido": row.get("partido"),
            "mercado": row.get("mercado"),
            "cuota_casa_csv": None if pd.isna(row.get("cuota_casa")) else float(row.get("cuota_casa")),
            "casa_apuestas_csv": row.get("casa_apuestas"),
            "mejor_cuota_actual": None if pd.isna(row.get("mejor_cuota_actual")) else float(row.get("mejor_cuota_actual")),
            "mejor_casa_actual": row.get("mejor_casa_actual"),
            "tennis_pick_id": None if pd.isna(row.get("tennis_pick_id")) else str(row.get("tennis_pick_id")),
            "tennis_pick_fecha": None if pd.isna(row.get("tennis_pick_fecha")) else str(row.get("tennis_pick_fecha")),
            "ev_csv": None if pd.isna(row.get("EV")) else float(row.get("EV")),
            "ev_actual": None if pd.isna(row.get("EV Actual")) else float(row.get("EV Actual")),
        },
    }
    (
        get_supabase_client()
        .schema("tennis")
        .table("bets_log")
        .insert(payload)
        .execute()
    )


def build_tennis_odds_tracking_frame() -> pd.DataFrame:
    picks = load_tennis_picks_history()
    history = load_tennis_pick_odds_history()
    if picks.empty or history.empty:
        return pd.DataFrame()

    tracking = history.merge(
        picks,
        left_on="pick_id",
        right_on="id",
        how="left",
        suffixes=("_snapshot", "_pick"),
    )
    if tracking.empty:
        return tracking

    for column in ["cuota", "ev_momento", "linea", "cuota_inicial", "nuestra_prob", "ev_inicial"]:
        if column in tracking.columns:
            tracking[column] = pd.to_numeric(tracking[column], errors="coerce")

    for column in ["fecha_registro_snapshot", "fecha_registro_pick"]:
        if column in tracking.columns:
            tracking[column] = pd.to_datetime(tracking[column], errors="coerce")

    tracking["linea_label"] = tracking["linea"].map(lambda value: "-" if pd.isna(value) else f"{value:g}")
    tracking["pick_label"] = (
        tracking["partido"].fillna("Sin partido")
        + " | "
        + tracking["mercado"].fillna("Sin mercado")
        + " "
        + tracking["linea_label"]
        + " | "
        + tracking["casa_apuestas"].fillna("Sin casa")
    )
    return tracking.sort_values(["pick_label", "fecha_registro_snapshot"]).reset_index(drop=True)


def render_tennis_live_hunter() -> None:
    col_download, col_features, col_ev = st.columns(3)
    download_sackmann = col_download.button("Descargar Historico ATP", use_container_width=True)
    process_features = col_features.button("Procesar Cansancio y Features", use_container_width=True)
    hunt_odds = col_ev.button("Cazar Cuotas (Calcular EV)", use_container_width=True)

    if download_sackmann:
        with st.spinner("Descargando y limpiando Sackmann..."):
            result = run_tennis_script("data_fetcher.py", timeout_seconds=300)
        st.session_state["last_tennis_script_result"] = {
            "script": "data_fetcher.py",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode == 0:
            load_tennis_csv.clear()
            st.success(result.stdout.strip() or "Datos de Sackmann actualizados.")
        else:
            st.error(result.stderr.strip() or "No se pudieron descargar los datos de Sackmann.")

    if process_features:
        with st.spinner("Procesando features avanzadas sin leakage..."):
            result = run_tennis_script("data_prep.py", timeout_seconds=300)
        st.session_state["last_tennis_script_result"] = {
            "script": "data_prep.py",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode == 0:
            load_tennis_csv.clear()
            st.success(result.stdout.strip() or "Features avanzadas actualizadas.")
        else:
            st.error(result.stderr.strip() or "No se pudieron procesar las features avanzadas.")

    if hunt_odds:
        if not TENNIS_FEATURES_PATH.exists():
            st.warning(f"Falta el dataset de features avanzadas: {TENNIS_FEATURES_PATH}.")
        else:
            with st.spinner("Cazando cuotas y calculando EV con Monte Carlo..."):
                result = run_tennis_script("ev_calculator.py", timeout_seconds=300)
            st.session_state["last_tennis_script_result"] = {
                "script": "ev_calculator.py",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            if result.returncode == 0:
                load_tennis_csv.clear()
                load_tennis_picks_history.clear()
                load_tennis_pick_odds_history.clear()
                load_tennis_upcoming_odds.clear()
                st.success(result.stdout.strip() or "Value bets de tenis actualizadas.")
            else:
                st.error(result.stderr.strip() or "No se pudo calcular EV de tenis.")

    value_bets = enrich_tennis_value_bets_with_market(
        add_tennis_kelly_column(load_tennis_csv(str(TENNIS_VALUE_BETS_PATH)))
    )
    sackmann_frame = load_tennis_csv(str(TENNIS_SACKMANN_PATH))
    features_frame = load_tennis_csv(str(TENNIS_FEATURES_PATH))

    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Historico limpio", len(sackmann_frame))
    metric_2.metric("Features avanzadas", len(features_frame))
    metric_3.metric("Value rows", len(value_bets))
    metric_4.metric("EV > 0", int((pd.to_numeric(value_bets.get("EV", pd.Series(dtype=float)), errors="coerce") > 0).sum()))
    metric_5.metric("EV actual > 0", int((pd.to_numeric(value_bets.get("EV Actual", pd.Series(dtype=float)), errors="coerce") > 0).sum()))

    render_tennis_diagnostics()

    st.subheader("Value Bets Tenis")
    if value_bets.empty:
        st.info("No hay value bets calculadas todavia. Usa `Cazar Cuotas (Calcular EV)`.")
        return

    numeric_columns = [
        "linea",
        "cuota_casa",
        "nuestra_prob",
        "EV",
        "Kelly Recomendado",
        "mejor_cuota_actual",
        "EV Actual",
        "Kelly Actual",
        "delta_cuota_actual",
        "casas_actuales",
        "min_cuota_actual",
        "max_cuota_actual",
        "rango_cuotas_actual",
    ]
    for column in numeric_columns:
        if column in value_bets.columns:
            value_bets[column] = pd.to_numeric(value_bets[column], errors="coerce")
    preferred_columns = [
        "partido",
        "mercado",
        "linea",
        "casa_apuestas",
        "cuota_casa",
        "mejor_casa_actual",
        "mejor_cuota_actual",
        "delta_cuota_actual",
        "nuestra_prob",
        "EV",
        "EV Actual",
        "Kelly Recomendado",
        "Kelly Actual",
        "casas_actuales",
        "min_cuota_actual",
        "max_cuota_actual",
        "rango_cuotas_actual",
        "ultima_actualizacion_cuota",
        "tennis_pick_id",
        "p_serve_A",
        "p_serve_B",
        "mean_total_games",
        "mean_game_diff",
    ]
    display_value_bets = value_bets[[column for column in preferred_columns if column in value_bets.columns]]

    styled = display_value_bets.style.apply(style_tennis_value_rows, axis=1).format(
        {
            "linea": "{:.1f}",
            "cuota_casa": "{:.2f}",
            "nuestra_prob": "{:.2%}",
            "EV": "{:.2%}",
            "Kelly Recomendado": "{:.2f} EUR",
            "mejor_cuota_actual": "{:.2f}",
            "EV Actual": "{:.2%}",
            "Kelly Actual": "{:.2f} EUR",
            "delta_cuota_actual": "{:+.2f}",
            "casas_actuales": "{:.0f}",
            "min_cuota_actual": "{:.2f}",
            "max_cuota_actual": "{:.2f}",
            "rango_cuotas_actual": "{:.2f}",
            "p_serve_A": "{:.2%}",
            "p_serve_B": "{:.2%}",
            "mean_total_games": "{:.2f}",
            "mean_game_diff": "{:.2f}",
        },
        na_rep="-",
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    render_tennis_bet_registration(value_bets)


def render_tennis_bet_registration(value_bets: pd.DataFrame) -> None:
    st.subheader("Registrar apuesta")
    success_message = st.session_state.pop("tennis_bet_registration_success", None)
    if success_message:
        st.success(success_message)

    candidate_bets = value_bets.copy()
    if "EV Actual" in candidate_bets.columns:
        candidate_bets["ev_para_registro"] = pd.to_numeric(candidate_bets["EV Actual"], errors="coerce")
    else:
        candidate_bets["ev_para_registro"] = pd.to_numeric(candidate_bets.get("EV"), errors="coerce")
    candidate_bets = candidate_bets.loc[candidate_bets["ev_para_registro"].fillna(-1) > 0].copy()
    if candidate_bets.empty:
        st.info("No hay value bets con EV actual positivo para registrar.")
        return

    candidate_bets = candidate_bets.reset_index(drop=True)
    candidate_bets["registro_label"] = candidate_bets.apply(
        lambda row: (
            f"{int(row.name) + 1}. {row.get('partido', '-')} | {row.get('mercado', '-')} "
            f"{row.get('linea', '-')} | cuota {row.get('mejor_cuota_actual', row.get('cuota_casa', '-'))}"
        ),
        axis=1,
    )
    selected_label = st.selectbox("Value bet", candidate_bets["registro_label"].tolist())
    selected_row = candidate_bets.loc[candidate_bets["registro_label"].eq(selected_label)].iloc[0]

    default_odd = selected_row.get("mejor_cuota_actual")
    if pd.isna(default_odd):
        default_odd = selected_row.get("cuota_casa")
    default_odd = float(default_odd)
    probability = float(selected_row.get("nuestra_prob"))

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Prob modelo", f"{probability:.2%}")
    col_2.metric("Cuota por defecto", f"{default_odd:.2f}")
    col_3.metric("Casa", str(selected_row.get("mejor_casa_actual") or selected_row.get("casa_apuestas") or "-"))
    col_4.metric("EV actual", f"{(probability * default_odd - 1.0):.2%}")

    with st.form("tennis_bet_registration_form", clear_on_submit=False):
        form_col_1, form_col_2, form_col_3, form_col_4 = st.columns(4)
        bet_type = form_col_1.selectbox("Tipo", ["real", "paper"])
        bankroll = form_col_2.number_input("Bankroll", min_value=0.0, value=1000.0, step=50.0)
        kelly_multiplier = form_col_3.selectbox("Kelly", [0.25, 0.5, 1.0], format_func=lambda value: f"{value:g}x")
        manual_odd = form_col_4.number_input("Cuota", min_value=1.01, value=default_odd, step=0.01, format="%.2f")

        ev, applied_kelly_fraction, recommended_stake = calculate_tennis_kelly_values(
            probability,
            manual_odd,
            bankroll,
            kelly_multiplier,
        )
        use_recommended_stake = st.checkbox("Usar stake Kelly recomendado", value=True)
        if use_recommended_stake:
            stake = float(round(recommended_stake, 2))
            st.metric("Stake a registrar", f"{stake:.2f}")
        else:
            stake = st.number_input("Stake manual", min_value=0.0, value=float(round(recommended_stake, 2)), step=1.0, format="%.2f")
        notes = st.text_area("Notas", value="")

        calc_col_1, calc_col_2, calc_col_3 = st.columns(3)
        calc_col_1.metric("EV con tu cuota", f"{ev:.2%}")
        calc_col_2.metric("Kelly aplicado", f"{applied_kelly_fraction:.2%}")
        calc_col_3.metric("Stake recomendado", f"{recommended_stake:.2f}")
        submitted = st.form_submit_button("Registrar en bets_log", use_container_width=True)

    if submitted:
        try:
            stored_kelly_fraction = stake / bankroll if bankroll > 0 else applied_kelly_fraction
            insert_tennis_bet_log(
                selected_row,
                bet_type=bet_type,
                odd=manual_odd,
                stake=stake,
                kelly_fraction=stored_kelly_fraction,
                notes=notes,
            )
            load_tennis_bets_log.clear()
            st.session_state["tennis_bet_registration_success"] = "Apuesta registrada en tennis.bets_log."
            st.rerun()
        except Exception as exc:
            st.error(f"No pude registrar la apuesta: {exc}")


def render_tennis_market_odds() -> None:
    if st.button("Refrescar Mercado", use_container_width=False):
        load_tennis_upcoming_odds.clear()

    try:
        odds = load_tennis_upcoming_odds()
    except Exception as exc:
        st.error(f"No pude cargar upcoming_odds: {exc}")
        return

    if odds.empty:
        st.info("No hay cuotas guardadas en upcoming_odds todavia.")
        return

    best_odds = build_tennis_best_market_odds(odds)
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Cuotas activas", len(odds))
    metric_2.metric("Partidos", int(odds["external_event_id"].nunique()))
    metric_3.metric("Casas", int(odds["bookmaker_key"].nunique()))
    metric_4.metric("Mejores mercados", len(best_odds))

    matches = ["Todos"] + odds["match_label"].dropna().drop_duplicates().sort_values().tolist()
    markets = ["Todos"] + odds["market_key"].dropna().drop_duplicates().sort_values().tolist()
    col_match, col_market = st.columns(2)
    selected_match = col_match.selectbox("Partido", matches)
    selected_market = col_market.selectbox("Mercado", markets)

    filtered_best = best_odds.copy()
    filtered_all = odds.copy()
    if selected_match != "Todos":
        filtered_best = filtered_best.loc[filtered_best["match_label"].eq(selected_match)]
        filtered_all = filtered_all.loc[filtered_all["match_label"].eq(selected_match)]
    if selected_market != "Todos":
        filtered_best = filtered_best.loc[filtered_best["market_key"].eq(selected_market)]
        filtered_all = filtered_all.loc[filtered_all["market_key"].eq(selected_market)]

    st.subheader("Mejor cuota por seleccion")
    best_columns = [
        "commence_time",
        "match_label",
        "market_key",
        "selection",
        "line",
        "price",
        "bookmaker_title",
        "casas",
        "min_cuota",
        "max_cuota",
        "rango_cuotas",
        "ultima_actualizacion",
    ]
    st.dataframe(
        filtered_best[[column for column in best_columns if column in filtered_best.columns]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "commence_time": st.column_config.DatetimeColumn("Inicio", format="YYYY-MM-DD HH:mm"),
            "match_label": "Partido",
            "market_key": "Mercado",
            "selection": "Seleccion",
            "line": st.column_config.NumberColumn("Linea", format="%.1f"),
            "price": st.column_config.NumberColumn("Mejor cuota", format="%.2f"),
            "bookmaker_title": "Casa",
            "casas": st.column_config.NumberColumn("Casas", format="%d"),
            "min_cuota": st.column_config.NumberColumn("Min", format="%.2f"),
            "max_cuota": st.column_config.NumberColumn("Max", format="%.2f"),
            "rango_cuotas": st.column_config.NumberColumn("Rango", format="%.2f"),
            "ultima_actualizacion": st.column_config.DatetimeColumn("Ultimo cambio", format="YYYY-MM-DD HH:mm"),
        },
    )

    st.subheader("Detalle por casa")
    detail_columns = [
        "commence_time",
        "match_label",
        "market_key",
        "selection",
        "line",
        "bookmaker_title",
        "price",
        "implied_probability",
        "collected_at",
    ]
    detail = filtered_all.sort_values(["commence_time", "match_label", "market_key", "selection", "line", "price"], ascending=[True, True, True, True, True, False])
    st.dataframe(
        detail[[column for column in detail_columns if column in detail.columns]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "commence_time": st.column_config.DatetimeColumn("Inicio", format="YYYY-MM-DD HH:mm"),
            "match_label": "Partido",
            "market_key": "Mercado",
            "selection": "Seleccion",
            "line": st.column_config.NumberColumn("Linea", format="%.1f"),
            "bookmaker_title": "Casa",
            "price": st.column_config.NumberColumn("Cuota", format="%.2f"),
            "implied_probability": st.column_config.NumberColumn("Prob implicita", format="%.2f"),
            "collected_at": st.column_config.DatetimeColumn("Ultimo cambio", format="YYYY-MM-DD HH:mm"),
        },
    )


def render_tennis_odds_tracking() -> None:
    if st.button("Refrescar Graficas", use_container_width=False):
        load_tennis_picks_history.clear()
        load_tennis_pick_odds_history.clear()

    try:
        tracking = build_tennis_odds_tracking_frame()
    except Exception as exc:
        st.error(f"No pude cargar el tracking de cuotas: {exc}")
        return

    if tracking.empty:
        st.info("No hay historico de cuotas para graficar todavia.")
        return

    tracked_pick_count = int(tracking["pick_id"].nunique())
    snapshot_count = len(tracking)
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Picks trackeados", tracked_pick_count)
    metric_2.metric("Snapshots", snapshot_count)
    metric_3.metric("Media snapshots/pick", f"{snapshot_count / max(tracked_pick_count, 1):.1f}")

    pick_options = tracking["pick_label"].dropna().drop_duplicates().sort_values().tolist()
    selected_pick = st.selectbox("Pick", pick_options)
    show_all = st.checkbox("Comparar todos los picks", value=False)

    chart_frame = tracking if show_all else tracking.loc[tracking["pick_label"].eq(selected_pick)]
    if chart_frame.empty:
        st.info("No hay puntos para el pick seleccionado.")
        return

    selected_summary = chart_frame.groupby("pick_label", as_index=False).agg(
        primera_cuota=("cuota", "first"),
        ultima_cuota=("cuota", "last"),
        max_cuota=("cuota", "max"),
        min_cuota=("cuota", "min"),
        snapshots=("cuota", "count"),
    )
    selected_summary["movimiento"] = selected_summary["ultima_cuota"] - selected_summary["primera_cuota"]

    if not show_all and not selected_summary.empty:
        row = selected_summary.iloc[0]
        col_1, col_2, col_3, col_4 = st.columns(4)
        col_1.metric("Cuota inicial", f"{row['primera_cuota']:.2f}")
        col_2.metric("Cuota actual", f"{row['ultima_cuota']:.2f}", delta=f"{row['movimiento']:.2f}")
        col_3.metric("Max", f"{row['max_cuota']:.2f}")
        col_4.metric("Min", f"{row['min_cuota']:.2f}")

    st.subheader("Evolucion de cuota")
    st.line_chart(
        chart_frame,
        x="fecha_registro_snapshot",
        y="cuota",
        color="pick_label" if show_all else None,
        use_container_width=True,
    )

    if "ev_momento" in chart_frame.columns and chart_frame["ev_momento"].notna().any():
        st.subheader("Evolucion de EV")
        st.line_chart(
            chart_frame,
            x="fecha_registro_snapshot",
            y="ev_momento",
            color="pick_label" if show_all else None,
            use_container_width=True,
        )

    st.subheader("Ultimos puntos")
    table_columns = [
        "fecha_registro_snapshot",
        "partido",
        "mercado",
        "linea",
        "casa_apuestas",
        "cuota",
        "ev_momento",
    ]
    visible_columns = [column for column in table_columns if column in chart_frame.columns]
    st.dataframe(
        chart_frame.sort_values("fecha_registro_snapshot", ascending=False)[visible_columns].head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "fecha_registro_snapshot": st.column_config.DatetimeColumn("Fecha", format="YYYY-MM-DD HH:mm"),
            "linea": st.column_config.NumberColumn("Linea", format="%.1f"),
            "cuota": st.column_config.NumberColumn("Cuota", format="%.2f"),
            "ev_momento": st.column_config.NumberColumn("EV momento", format="%.2f"),
        },
    )


def render_tennis_bets_log() -> None:
    if st.button("Refrescar Apuestas", use_container_width=False):
        load_tennis_bets_log.clear()

    try:
        bets = load_tennis_bets_log()
    except Exception as exc:
        st.error(f"No pude cargar bets_log: {exc}")
        return

    if bets.empty:
        st.info("No hay apuestas registradas todavia.")
        return

    settled_bets = bets.loc[bets["status"].isin(["won", "lost", "void"])].copy()
    real_bets = bets.loc[bets["bet_type"].eq("real")].copy()
    pending_bets = bets.loc[bets["status"].eq("pending")].copy()
    total_staked = float(bets["stake"].sum())
    total_pnl = float(settled_bets["pnl"].fillna(0).sum()) if "pnl" in settled_bets.columns else 0.0
    real_pnl = float(real_bets["pnl"].fillna(0).sum()) if "pnl" in real_bets.columns else 0.0
    roi = total_pnl / total_staked if total_staked > 0 else 0.0

    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Apuestas", len(bets))
    metric_2.metric("Pendientes", len(pending_bets))
    metric_3.metric("Stake total", f"{total_staked:.2f}")
    metric_4.metric("PnL total", f"{total_pnl:.2f}")
    metric_5.metric("ROI", f"{roi:.2%}")
    st.caption(f"PnL real: {real_pnl:.2f}")

    type_options = ["Todos"] + bets["bet_type"].dropna().drop_duplicates().sort_values().tolist()
    status_options = ["Todos"] + bets["status"].dropna().drop_duplicates().sort_values().tolist()
    col_type, col_status = st.columns(2)
    selected_type = col_type.selectbox("Tipo", type_options)
    selected_status = col_status.selectbox("Estado", status_options)

    filtered = bets.copy()
    if selected_type != "Todos":
        filtered = filtered.loc[filtered["bet_type"].eq(selected_type)]
    if selected_status != "Todos":
        filtered = filtered.loc[filtered["status"].eq(selected_status)]

    visible_columns = [
        "placed_at",
        "tennis_pick_id",
        "bet_type",
        "status",
        "partido",
        "market_key",
        "selection",
        "line",
        "bookmaker_key",
        "my_odd",
        "model_probability",
        "ev",
        "stake",
        "kelly_fraction",
        "pnl",
        "settled_at",
        "notes",
    ]
    st.dataframe(
        filtered[[column for column in visible_columns if column in filtered.columns]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "placed_at": st.column_config.DatetimeColumn("Fecha", format="YYYY-MM-DD HH:mm"),
            "tennis_pick_id": "Pick ID",
            "bet_type": "Tipo",
            "status": "Estado",
            "partido": "Partido",
            "market_key": "Mercado",
            "selection": "Seleccion",
            "line": st.column_config.NumberColumn("Linea", format="%.1f"),
            "bookmaker_key": "Casa",
            "my_odd": st.column_config.NumberColumn("Cuota", format="%.2f"),
            "model_probability": st.column_config.NumberColumn("Prob modelo", format="%.2f"),
            "ev": st.column_config.NumberColumn("EV", format="%.2f"),
            "stake": st.column_config.NumberColumn("Stake", format="%.2f"),
            "kelly_fraction": st.column_config.NumberColumn("Kelly", format="%.2f"),
            "pnl": st.column_config.NumberColumn("PnL", format="%.2f"),
            "settled_at": st.column_config.DatetimeColumn("Cierre", format="YYYY-MM-DD HH:mm"),
        },
    )

    st.subheader("Cerrar apuesta pendiente")
    if pending_bets.empty:
        st.info("No hay apuestas pendientes.")
        return

    selected_bet_label = st.selectbox("Apuesta pendiente", pending_bets["bet_label"].tolist())
    selected_bet = pending_bets.loc[pending_bets["bet_label"].eq(selected_bet_label)].iloc[0]
    settle_col_1, settle_col_2 = st.columns(2)
    settle_status = settle_col_1.selectbox("Resultado", ["won", "lost", "void"])
    default_result = str(selected_bet.get("selection") or "") if settle_status == "won" else ""
    result_selection = settle_col_2.text_input("Seleccion ganadora / resultado", value=default_result)

    projected_pnl = 0.0
    if settle_status == "won":
        projected_pnl = float(selected_bet["stake"]) * (float(selected_bet["my_odd"]) - 1.0)
    elif settle_status == "lost":
        projected_pnl = -float(selected_bet["stake"])
    st.metric("PnL proyectado", f"{projected_pnl:.2f}")

    if st.button("Cerrar apuesta", use_container_width=True):
        try:
            settle_tennis_bet(str(selected_bet["id"]), settle_status, result_selection)
            load_tennis_bets_log.clear()
            st.success("Apuesta cerrada.")
            st.rerun()
        except Exception as exc:
            st.error(f"No pude cerrar la apuesta: {exc}")


def render_tennis_picks_history() -> None:
    if st.button("Refrescar Historial", use_container_width=False):
        load_tennis_picks_history.clear()

    try:
        picks_history = load_tennis_picks_history()
    except Exception as exc:
        st.error(f"No pude cargar el historial de Supabase: {exc}")
        return

    if picks_history.empty:
        st.info("No hay picks registrados aun.")
        return

    for column in ["linea", "cuota_inicial", "nuestra_prob", "ev_inicial", "cuota_cierre"]:
        if column in picks_history.columns:
            picks_history[column] = pd.to_numeric(picks_history[column], errors="coerce")
    if "fecha_registro" in picks_history.columns:
        picks_history["fecha_registro"] = pd.to_datetime(picks_history["fecha_registro"], errors="coerce")

    preferred_columns = [
        "fecha_registro",
        "partido",
        "mercado",
        "linea",
        "casa_apuestas",
        "cuota_inicial",
        "nuestra_prob",
        "ev_inicial",
        "cuota_cierre",
        "resultado",
    ]
    visible_columns = [column for column in preferred_columns if column in picks_history.columns]
    display_frame = picks_history[visible_columns] if visible_columns else picks_history

    st.dataframe(
        display_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "fecha_registro": st.column_config.DatetimeColumn("Fecha registro", format="YYYY-MM-DD HH:mm"),
            "linea": st.column_config.NumberColumn("Linea", format="%.1f"),
            "cuota_inicial": st.column_config.NumberColumn("Cuota inicial", format="%.2f"),
            "nuestra_prob": st.column_config.NumberColumn("Nuestra prob", format="%.2f"),
            "ev_inicial": st.column_config.NumberColumn("EV inicial", format="%.2f"),
            "cuota_cierre": st.column_config.NumberColumn("Cuota cierre", format="%.2f"),
        },
    )


def render_tennis_dashboard_legacy() -> None:
    st.title("BetViero Tenis")
    st.caption("Backend Quant: histórico ATP, features sin leakage, Monte Carlo Markov y EV contra mercado.")

    col_download, col_features, col_ev = st.columns(3)
    download_sackmann = col_download.button("Descargar Histórico ATP", use_container_width=True)
    process_features = col_features.button("Procesar Cansancio y Features", use_container_width=True)
    hunt_odds = col_ev.button("Cazar Cuotas (Calcular EV)", use_container_width=True)

    if download_sackmann:
        with st.spinner("Descargando y limpiando Sackmann..."):
            result = run_tennis_script("data_fetcher.py", timeout_seconds=300)
        if result.returncode == 0:
            load_tennis_csv.clear()
            st.success(result.stdout.strip() or "Datos de Sackmann actualizados.")
        else:
            st.error(result.stderr.strip() or "No se pudieron descargar los datos de Sackmann.")

    if process_features:
        with st.spinner("Procesando features avanzadas sin leakage..."):
            result = run_tennis_script("data_prep.py", timeout_seconds=300)
        if result.returncode == 0:
            load_tennis_csv.clear()
            st.success(result.stdout.strip() or "Features avanzadas actualizadas.")
        else:
            st.error(result.stderr.strip() or "No se pudieron procesar las features avanzadas.")

    if hunt_odds:
        if not TENNIS_FEATURES_PATH.exists():
            st.warning(f"Falta el dataset de features avanzadas: {TENNIS_FEATURES_PATH}.")
        else:
            with st.spinner("Cazando cuotas y calculando EV con Monte Carlo..."):
                result = run_tennis_script("ev_calculator.py", timeout_seconds=300)
            if result.returncode == 0:
                load_tennis_csv.clear()
                st.success(result.stdout.strip() or "Value bets de tenis actualizadas.")
            else:
                st.error(result.stderr.strip() or "No se pudo calcular EV de tenis.")

    value_bets = add_tennis_kelly_column(load_tennis_csv(str(TENNIS_VALUE_BETS_PATH)))
    sackmann_frame = load_tennis_csv(str(TENNIS_SACKMANN_PATH))
    features_frame = load_tennis_csv(str(TENNIS_FEATURES_PATH))

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Histórico limpio", len(sackmann_frame))
    metric_2.metric("Features avanzadas", len(features_frame))
    metric_3.metric("Value rows", len(value_bets))
    metric_4.metric("EV > 0", int((pd.to_numeric(value_bets.get("EV", pd.Series(dtype=float)), errors="coerce") > 0).sum()))

    st.subheader("Value Bets Tenis")
    if value_bets.empty:
        st.info("No hay value bets calculadas todavía. Usa `Cazar Cuotas (Calcular EV)`.")
        return

    numeric_columns = ["linea", "cuota_casa", "nuestra_prob", "EV", "Kelly Recomendado"]
    for column in numeric_columns:
        if column in value_bets.columns:
            value_bets[column] = pd.to_numeric(value_bets[column], errors="coerce")

    styled = value_bets.style.apply(style_tennis_value_rows, axis=1).format(
        {
            "linea": "{:.1f}",
            "cuota_casa": "{:.2f}",
            "nuestra_prob": "{:.2%}",
            "EV": "{:.2%}",
            "Kelly Recomendado": "{:.2f} €",
            "p_serve_A": "{:.2%}",
            "p_serve_B": "{:.2%}",
            "mean_total_games": "{:.2f}",
            "mean_game_diff": "{:.2f}",
        },
        na_rep="-",
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_tennis_dashboard() -> None:
    st.title("BetViero Tenis")
    st.caption("Backend Quant: historico ATP, features sin leakage, Monte Carlo Markov y EV contra mercado.")
    tab1, tab2 = st.tabs(["🎯 Cazador en Vivo", "🗄️ Historial de Picks (Supabase)"])
    with tab1:
        render_tennis_live_hunter()
    with tab2:
        render_tennis_picks_history()


def render_tennis_dashboard() -> None:
    st.title("BetViero Tenis")
    st.caption("Backend Quant: historico ATP, features sin leakage, Monte Carlo Markov y EV contra mercado.")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cazador en Vivo", "Mercado", "Graficas de Cuotas", "Apuestas", "Historial de Picks"])
    with tab1:
        render_tennis_live_hunter()
    with tab2:
        render_tennis_market_odds()
    with tab3:
        render_tennis_odds_tracking()
    with tab4:
        render_tennis_bets_log()
    with tab5:
        render_tennis_picks_history()
