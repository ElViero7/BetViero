from __future__ import annotations

import subprocess
import sys
from datetime import date
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from lol.src.base import LoLPipelineError
from lol.src.dashboard import (
    apply_filters,
    build_oracle_results_frame,
    build_unified_records,
    compute_current_bankroll,
    compute_summary_metrics,
    download_oracle_2026_csv,
    fetch_table,
    load_dashboard_config,
    load_predictions_snapshot,
    settle_legacy_bets,
    settle_picks,
    update_table_row,
)
from lol.src.fair_odds_bot import (
    BotConfig,
    build_alert_message,
    FairOddsEngine,
    insert_pick_to_supabase,
    load_config as load_bot_config,
)
from lol.src.multimarket import (
    calculate_ev,
    calculate_fair_odd,
    calculate_implied_probability,
    calculate_kelly_fraction,
    estimate_over_under_probability,
    train_multimarket_pipeline,
    MultiMarketTrainingConfig,
)
from lol.src.vision_parser import detect_matchups_from_ocr_lines, extract_matchups_from_image
from lol.src.vision_parser import apply_ocr_team_corrections, clean_ocr_team_text
from tennis.src.ev_calculator import get_supabase_client
from tennis.src.db import update_tennis_row


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
LOL_DIR = PROJECT_ROOT / "lol"
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

os.environ.setdefault("PREDICTIONS_INPUT_PATH", str(LOL_DIR / "data" / "predictions" / "upcoming_multimarket_predictions.csv"))
os.environ.setdefault("ORACLES_ELIXIR_RAW_DIR", str(LOL_DIR / "data" / "raw" / "oracles_elixir"))
os.environ.setdefault("MODEL_ARTIFACT_PATH", str(LOL_DIR / "models" / "modelo_multimarket.pkl"))
os.environ.setdefault(
    "ORACLES_ELIXIR_CANONICAL_PARQUET_PATH",
    str(LOL_DIR / "data" / "processed" / "oracles_elixir_team_rows_training.parquet"),
)
os.environ.setdefault("TEAM_NAME_OVERRIDES_PATH", str(LOL_DIR / "data" / "config" / "team_name_overrides.json"))

st.set_page_config(
    page_title="BetViero Dashboard",
    page_icon="🎮",
    layout="wide",
)


if "pending_pick" not in st.session_state:
    st.session_state["pending_pick"] = None
if "register_match_name" not in st.session_state:
    st.session_state["register_match_name"] = None
if "dashboard_page" not in st.session_state:
    st.session_state["dashboard_page"] = "Resumen"
if "force_register_page" not in st.session_state:
    st.session_state["force_register_page"] = False


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


@st.cache_data(ttl=60)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_dashboard_config()
    bets_frame = fetch_table(config, "bets")
    picks_frame = fetch_table(config, "picks")
    predictions_frame = load_predictions_snapshot(config)
    results_frame = build_oracle_results_frame(config.raw_data_dir)
    return bets_frame, picks_frame, predictions_frame, results_frame


def format_percentage(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2%}"


def format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}"


def add_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .bet-card {
            background: linear-gradient(180deg, #141824 0%, #10131d 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }
        .bet-card-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 0.88rem;
            color: #b6bfd6;
        }
        .bet-card-match {
            font-size: 1.02rem;
            font-weight: 700;
            color: #f7f9ff;
            margin-bottom: 8px;
        }
        .bet-card-market {
            font-size: 0.95rem;
            color: #dfe6fb;
            margin-bottom: 10px;
        }
        .bet-card-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }
        .bet-chip {
            background: rgba(255,255,255,0.04);
            border-radius: 12px;
            padding: 10px;
        }
        .bet-chip-label {
            font-size: 0.75rem;
            color: #97a3bf;
            margin-bottom: 4px;
        }
        .bet-chip-value {
            font-size: 0.98rem;
            color: #ffffff;
            font-weight: 700;
        }
        .bet-status {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .bet-status-won { background: rgba(39,174,96,0.18); color: #6ee7a1; }
        .bet-status-lost { background: rgba(231,76,60,0.18); color: #ff8f8f; }
        .bet-status-pending { background: rgba(241,196,15,0.18); color: #ffd86b; }
        .bet-status-void { background: rgba(149,165,166,0.18); color: #d0d6db; }
        .control-box {
            background: linear-gradient(180deg, #141824 0%, #10131d 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def filter_by_date(records: pd.DataFrame, start_date: date | None, end_date: date | None) -> pd.DataFrame:
    if records.empty:
        return records
    filtered = records.copy()
    created_dates = filtered["created_at"].dt.date
    if start_date is not None:
        filtered = filtered.loc[created_dates >= start_date]
        created_dates = filtered["created_at"].dt.date
    if end_date is not None:
        filtered = filtered.loc[created_dates <= end_date]
    return filtered.reset_index(drop=True)


def filter_by_investment_mode(records: pd.DataFrame, investment_mode: str) -> pd.DataFrame:
    if records.empty or investment_mode == "Combinado":
        return records
    target_bet_type = "paper" if investment_mode == "Paper" else "real"
    bet_types = records.get("bet_type", pd.Series("real", index=records.index)).fillna("real").astype(str).str.lower()
    return records.loc[bet_types.eq(target_bet_type)].reset_index(drop=True)


def render_summary_metrics(summary: dict[str, object]) -> None:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Registros", summary["total_records"])
    col2.metric("Pendientes", summary["pending_records"])
    col3.metric("Liquidadas", summary["settled_records"])
    col4.metric("Stake total", format_currency(summary["total_stake"]))
    col5.metric("PnL", format_currency(summary["total_pnl"]))
    col6.metric("Bankroll actual", format_currency(summary["current_bankroll"]))

    col6, col7, col8, col9 = st.columns(4)
    col6.metric("ROI", format_percentage(summary["roi"]))
    col7.metric("Avg Edge", format_percentage(summary["avg_edge"]))
    col8.metric("Avg EV", format_percentage(summary["avg_ev"]))
    col9.metric(
        "W/L/V",
        f"{summary['won_records']}/{summary['lost_records']}/{summary['void_records']}",
    )


def render_cards(records: pd.DataFrame) -> None:
    if records.empty:
        st.info("No hay apuestas para los filtros seleccionados.")
        return
    for _, row in records.iterrows():
        status = str(row.get("status", "pending")).lower()
        status_class = f"bet-status bet-status-{status if status in {'won', 'lost', 'pending', 'void'} else 'pending'}"
        event_at = row.get("event_date")
        if pd.isna(event_at):
            event_at = row.get("created_at")
        created_text = event_at.strftime("%Y-%m-%d %H:%M") if pd.notna(event_at) else "-"
        line_value = "-" if pd.isna(row.get("line")) else f"{float(row['line']):.1f}"
        pnl_value = format_currency(row.get("pnl"))
        st.markdown(
            f"""
            <div class="bet-card">
              <div class="bet-card-top">
                <div>{row.get('source_table', '-').upper()} | {str(row.get('bet_type', 'real')).upper()} | {row.get('league', '-')}</div>
                <div class="{status_class}">{status.upper()}</div>
              </div>
              <div class="bet-card-match">{row.get('match_name', '-')}</div>
              <div class="bet-card-market">{row.get('market', '-')} | {row.get('selection', '-')}</div>
              <div class="bet-card-grid">
                <div class="bet-chip"><div class="bet-chip-label">Partido</div><div class="bet-chip-value">{created_text}</div></div>
                <div class="bet-chip"><div class="bet-chip-label">Cuota</div><div class="bet-chip-value">{format_currency(row.get('my_odd'))}</div></div>
                <div class="bet-chip"><div class="bet-chip-label">Fair</div><div class="bet-chip-value">{format_currency(row.get('model_fair_odd'))}</div></div>
                <div class="bet-chip"><div class="bet-chip-label">Stake</div><div class="bet-chip-value">{format_currency(row.get('recommended_stake'))}</div></div>
                <div class="bet-chip"><div class="bet-chip-label">Linea</div><div class="bet-chip-value">{line_value}</div></div>
                <div class="bet-chip"><div class="bet-chip-label">Edge</div><div class="bet-chip-value">{format_percentage(row.get('edge'))}</div></div>
                <div class="bet-chip"><div class="bet-chip-label">EV</div><div class="bet-chip-value">{format_percentage(row.get('ev'))}</div></div>
                <div class="bet-chip"><div class="bet-chip-label">PnL</div><div class="bet-chip-value">{pnl_value}</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def send_alerts_to_telegram(config: BotConfig, frame: pd.DataFrame) -> int:
    sent = 0
    for league_name, league_frame in frame.groupby("league", sort=False):
        requests.post(
            f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
            data={"chat_id": config.telegram_chat_id, "text": f"{league_name}\n"},
            timeout=30,
        )
        for _, row in league_frame.iterrows():
            requests.post(
                f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
                data={"chat_id": config.telegram_chat_id, "text": build_alert_message(row)},
                timeout=30,
            )
            sent += 1
    return sent


def save_predictions_snapshot(frame: pd.DataFrame, output_path: Path, merge: bool = True) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if merge and output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["match_name"], keep="last")
    else:
        combined = frame.copy()
    combined = combined.sort_values(["league", "scheduled_at", "match_name"], na_position="last").reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    load_dashboard_data.clear()
    return combined


def enrich_event_dates(records: pd.DataFrame, predictions_frame: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    enriched = records.copy()
    if "event_date" not in enriched.columns:
        enriched["event_date"] = pd.NaT
    enriched["event_date"] = pd.to_datetime(enriched["event_date"], utc=True, errors="coerce")
    if predictions_frame.empty or "match_name" not in predictions_frame.columns or "scheduled_at" not in predictions_frame.columns:
        return enriched
    schedule_map = (
        predictions_frame[["match_name", "scheduled_at"]]
        .dropna(subset=["match_name"])
        .drop_duplicates(subset=["match_name"], keep="last")
        .copy()
    )
    schedule_map["scheduled_at"] = pd.to_datetime(schedule_map["scheduled_at"], utc=True, errors="coerce")
    merged = enriched.merge(schedule_map, on="match_name", how="left")
    merged["event_date"] = merged["event_date"].fillna(merged["scheduled_at"])
    merged = merged.drop(columns=["scheduled_at"])
    return merged


def parse_manual_matches_text(text: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            league_part, match_part = [part.strip() for part in line.split("|", 1)]
            if " vs " not in match_part:
                raise LoLPipelineError(f"Linea invalida: {line}")
            team_a, team_b = [part.strip() for part in match_part.split(" vs ", 1)]
            rows.append({"league": league_part, "team_a": team_a, "team_b": team_b})
        else:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                raise LoLPipelineError(f"Linea invalida: {line}")
            rows.append({"league": parts[0], "team_a": parts[1], "team_b": parts[2]})
    if not rows:
        raise LoLPipelineError("No se detectaron partidos manuales.")
    return pd.DataFrame(rows)


def build_valid_team_list(predictions_frame: pd.DataFrame) -> list[str]:
    if predictions_frame.empty:
        return []
    team_values: list[str] = []
    for column in ["team_a", "team_b"]:
        if column in predictions_frame.columns:
            team_values.extend(predictions_frame[column].dropna().astype(str).tolist())
    cleaned_teams = [apply_ocr_team_corrections(team_name) for team_name in team_values]
    cleaned_teams = [team_name for team_name in cleaned_teams if team_name]
    return sorted(set(cleaned_teams))


def find_prediction_match_name(predictions_frame: pd.DataFrame, team_a: str, team_b: str) -> str | None:
    if predictions_frame.empty or not {"team_a", "team_b", "match_name"}.issubset(predictions_frame.columns):
        return None
    target_pair = {clean_ocr_team_text(team_a).lower(), clean_ocr_team_text(team_b).lower()}
    for _, row in predictions_frame.iterrows():
        row_pair = {
            clean_ocr_team_text(str(row["team_a"])).lower(),
            clean_ocr_team_text(str(row["team_b"])).lower(),
        }
        if row_pair == target_pair:
            return str(row["match_name"])
    return None


def render_magic_prediction_capture(predictions_frame: pd.DataFrame) -> None:
    with st.container(border=True):
        st.markdown("**Prediccion Magica por Captura**")
        uploaded_images = st.file_uploader(
            "Pega o sube una o varias capturas de cuotas Winamax",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if not uploaded_images:
            return

        valid_teams = build_valid_team_list(predictions_frame)
        st.write(f"Debug equipos validos para fuzzy: {len(valid_teams)}")
        st.write("Muestra de equipos validos:", valid_teams[:5])
        print("Equipos validos para fuzzy:", valid_teams)
        if not valid_teams:
            st.warning("No hay equipos validos en el snapshot actual para hacer fuzzy matching.")
            return

        raw_matchups: list[str] = []
        for uploaded_image in uploaded_images:
            try:
                raw_matchups.extend(extract_matchups_from_image(uploaded_image.getvalue()))
            except Exception as exc:
                st.error(f"No pude procesar `{uploaded_image.name}`: {exc}")

        if not raw_matchups:
            st.warning("No se detectaron lineas de equipos en las capturas subidas.")
            return

        with st.expander("Ver texto bruto detectado (OCR Raw)"):
            st.write(raw_matchups)

        raw_matchups = list(dict.fromkeys(raw_matchups))
        detected_matchups = detect_matchups_from_ocr_lines(raw_matchups, valid_teams, minimum_score=70)
        if not detected_matchups:
            st.warning("No hay equipos con confianza alta. Revisa el OCR Raw o corrige manualmente en el siguiente ajuste.")
            return

        st.caption(f"OCR procesado sobre {len(uploaded_images)} captura(s). Valida los emparejamientos antes de continuar.")
        selector_options = ["Desconocido (Seleccionar...)"] + valid_teams
        match_options: list[tuple[str, str]] = []
        for row_index, detected_matchup in enumerate(detected_matchups):
            raw_line = str(detected_matchup["raw_line"])
            team_a = str(detected_matchup["team_a"])
            team_b = str(detected_matchup["team_b"])
            score_a = int(detected_matchup["score_a"])
            score_b = int(detected_matchup["score_b"])

            st.markdown(f"**Linea OCR:** {raw_line}")
            col_detected_a, col_confidence_a, col_selector_a = st.columns([2, 1, 3])
            col_detected_a.write(team_a)
            col_confidence_a.write(f"{score_a}%")
            default_a = selector_options.index(team_a) if team_a in selector_options else 0
            selected_team_a = col_selector_a.selectbox(
                "Selector equipo 1",
                selector_options,
                index=default_a,
                key=f"magic_team_a_selector_{row_index}_{team_a}",
            )

            col_detected_b, col_confidence_b, col_selector_b = st.columns([2, 1, 3])
            col_detected_b.write(team_b or "Desconocido")
            col_confidence_b.write(f"{score_b}%")
            default_b = selector_options.index(team_b) if team_b in selector_options else 0
            selected_team_b = col_selector_b.selectbox(
                "Selector equipo 2",
                selector_options,
                index=default_b,
                key=f"magic_team_b_selector_{row_index}_{team_b or 'unknown'}",
            )

            if selected_team_a.startswith("Desconocido") or selected_team_b.startswith("Desconocido"):
                continue

            team_a = selected_team_a
            team_b = selected_team_b
            match_name = find_prediction_match_name(predictions_frame, team_a, team_b)
            label = f"{team_a} vs {team_b}"
            if match_name is None:
                label = f"{label} (no encontrado en snapshot)"
            match_options.append((label, match_name or ""))

        if not match_options:
            st.info("Selecciona al menos dos equipos para formar un partido.")
            return

        labels = [label for label, _ in match_options]
        selected_label = st.selectbox("Partido detectado para cargar", labels)
        selected_match_name = dict(match_options).get(selected_label)
        if st.button("Cargar en Calculadora", use_container_width=True):
            if not selected_match_name:
                st.warning("Ese partido no existe en el snapshot actual. Primero genera predicciones para ese matchup.")
            else:
                st.session_state["register_match_name"] = selected_match_name
                st.session_state["lol_register_match_name"] = selected_match_name
                st.session_state["force_register_page"] = True
                st.success(f"Partido cargado en calculadora: {selected_match_name}")
                st.rerun()


def compute_pick_payload_from_row(
    config: BotConfig,
    row: pd.Series,
    market: str,
    selection_key: str,
    my_odd: float,
    bankroll_amount: float,
    line: float | None = None,
) -> dict[str, object]:
    if market in {"winner", "firstblood", "firsttower"}:
        if market == "firstblood" and str(row["league"]) == "LPL":
            raise LoLPipelineError("First Blood queda desactivado para LPL.")
        team_name = str(row["team_a"]) if selection_key == "1" else str(row["team_b"])
        probability = float(row[f"{market}_{'team_a' if selection_key == '1' else 'team_b'}_prob"])
        fair_odd = float(row[f"{market}_{'team_a' if selection_key == '1' else 'team_b'}_fair_odd"])
        selection = team_name
        predicted_value = None
    else:
        if line is None:
            raise LoLPipelineError("La linea es obligatoria para kills.")
        probability = estimate_over_under_probability(
            predicted_total=float(row["kills_prediction"]),
            line=float(line),
            residual_std=float(row.get("kills_residual_std", 6.0)),
            direction=selection_key,
        )
        fair_odd = calculate_fair_odd(probability)
        selection = selection_key
        predicted_value = float(row["kills_prediction"])

    implied_probability = calculate_implied_probability(my_odd)
    edge = probability - implied_probability
    ev = calculate_ev(probability, my_odd)
    full_kelly = calculate_kelly_fraction(probability, my_odd)
    recommended_stake = min(
        bankroll_amount * full_kelly * config.kelly_fraction,
        config.max_stake_amount,
    )
    return {
        "match_name": str(row["match_name"]),
        "league": str(row["league"]),
        "market": market,
        "selection": selection,
        "line": line,
        "my_odd": my_odd,
        "model_probability": probability,
        "model_fair_odd": fair_odd,
        "predicted_value": predicted_value,
        "implied_probability": implied_probability,
        "edge": edge,
        "ev": ev,
        "full_kelly": full_kelly,
        "kelly_fraction": config.kelly_fraction,
        "recommended_stake": recommended_stake,
        "bankroll_snapshot": bankroll_amount,
        "status": "pending",
    }


def render_overview(records: pd.DataFrame) -> None:
    st.subheader("Pendientes")
    pending_view = records.loc[
        records["status"].eq("pending"),
        [
            "created_at",
            "source_table",
            "bet_type",
            "league",
            "match_name",
            "market",
            "selection",
            "line",
            "my_odd",
            "model_fair_odd",
            "recommended_stake",
            "edge",
            "ev",
        ],
    ].copy()
    st.dataframe(pending_view, use_container_width=True, hide_index=True)

    st.subheader("Desgloses")
    left, right = st.columns(2)
    with left:
        by_market = (
            records.groupby("market", dropna=False)
            .agg(records=("record_id", "count"), avg_edge=("edge", "mean"), avg_ev=("ev", "mean"), pnl=("pnl", "sum"))
            .reset_index()
        )
        st.dataframe(by_market, use_container_width=True, hide_index=True)
    with right:
        by_status = (
            records.groupby("status", dropna=False)
            .agg(records=("record_id", "count"), pnl=("pnl", "sum"))
            .reset_index()
        )
        st.dataframe(by_status, use_container_width=True, hide_index=True)


def render_history_table(records: pd.DataFrame) -> None:
    if records.empty:
        st.info("No hay historial para mostrar.")
        return
    history_view = records[
        [
            "created_at",
            "source_table",
            "bet_type",
            "league",
            "match_name",
            "market",
            "selection",
            "line",
            "my_odd",
            "model_fair_odd",
            "model_probability",
            "predicted_value",
            "edge",
            "ev",
            "recommended_stake",
            "status",
            "pnl",
        ]
    ].copy()
    st.dataframe(history_view, use_container_width=True, hide_index=True)


def render_predictions(predictions_frame: pd.DataFrame) -> None:
    if predictions_frame.empty:
        st.info("No hay snapshot de predicciones disponible.")
        return
    preferred_cols = [
        "league",
        "scheduled_at",
        "match_name",
        "winner_team_a_fair_odd",
        "winner_team_b_fair_odd",
        "firstblood_team_a_fair_odd",
        "firstblood_team_b_fair_odd",
        "firsttower_team_a_fair_odd",
        "firsttower_team_b_fair_odd",
        "kills_prediction",
    ]
    keep_cols = [col for col in preferred_cols if col in predictions_frame.columns]
    st.dataframe(predictions_frame[keep_cols], use_container_width=True, hide_index=True)


def render_control_page(bot_config: BotConfig, dashboard_config, predictions_frame: pd.DataFrame) -> None:
    st.subheader("Control")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Oracle**")
            if st.button("Actualizar CSV Oracle 2026", use_container_width=True):
                try:
                    output_path = download_oracle_2026_csv(
                        drive_url=dashboard_config.oracle_2026_drive_url,
                        output_path=dashboard_config.raw_data_dir / "2026_LoL_esports_match_data_from_OraclesElixir.csv",
                    )
                    st.success(f"CSV 2026 actualizado: {output_path.name}")
                    st.cache_data.clear()
                except Exception as exc:
                    st.error(str(exc))
            if st.button("Sincronizar legacy bets", use_container_width=True):
                try:
                    bets_frame = fetch_table(dashboard_config, "bets")
                    if not bets_frame.empty:
                        bets_frame = bets_frame.loc[
                            bets_frame.get("status").fillna("pending").astype(str).str.lower().eq("pending")
                        ].copy()
                    settled = settle_legacy_bets(bets_frame, build_oracle_results_frame(dashboard_config.raw_data_dir))
                    updated = 0
                    for _, row in settled.iterrows():
                        update_table_row(
                            dashboard_config,
                            "bets",
                            str(row["id"]),
                            {"status": str(row["settled_status"])},
                        )
                        updated += 1
                    st.success(f"Legacy bets sincronizadas: {updated}")
                    load_dashboard_data.clear()
                except Exception as exc:
                    st.error(str(exc))
            if st.button("Sincronizar picks", use_container_width=True):
                try:
                    picks_frame = fetch_table(dashboard_config, "picks")
                    if not picks_frame.empty:
                        picks_frame = picks_frame.loc[
                            picks_frame.get("status").fillna("pending").astype(str).str.lower().eq("pending")
                        ].copy()
                    settled = settle_picks(picks_frame, build_oracle_results_frame(dashboard_config.raw_data_dir))
                    updated = 0
                    for _, row in settled.iterrows():
                        update_table_row(
                            dashboard_config,
                            "picks",
                            str(row["id"]),
                            {"status": str(row["settled_status"])},
                        )
                        updated += 1
                    st.success(f"Picks sincronizados: {updated}")
                    load_dashboard_data.clear()
                except Exception as exc:
                    st.error(str(exc))
    with col2:
        with st.container(border=True):
            st.markdown("**Modelo**")
            if st.button("Reentrenar modelo", use_container_width=True):
                try:
                    summary = train_multimarket_pipeline(
                        input_path=dashboard_config.raw_data_dir,
                        model_output_path=LOL_DIR / "models" / "modelo_multimarket.pkl",
                        team_rows_output_path=LOL_DIR / "data" / "processed" / "oracles_elixir_team_rows_training.parquet",
                        matchup_output_path=LOL_DIR / "data" / "processed" / "multimarket_matchups.parquet",
                        config=MultiMarketTrainingConfig(),
                    )
                    st.success(
                        f"Modelo reentrenado. Matchups: {summary['matchup_rows']}. Winner AUC: {summary['market_metrics']['winner'].get('roc_auc'):.3f}"
                    )
                except Exception as exc:
                    st.error(str(exc))
            if st.button("Scan PandaScore y guardar snapshot", use_container_width=True):
                try:
                    frame = FairOddsEngine(bot_config).score_upcoming_matches()
                    save_predictions_snapshot(frame, dashboard_config.predictions_output_path, merge=False)
                    st.success(f"Snapshot actualizado con {len(frame)} partidos.")
                except Exception as exc:
                    st.error(str(exc))

    render_magic_prediction_capture(predictions_frame)

    with st.container(border=True):
        st.markdown("**Partidos manuales**")
        manual_text = st.text_area(
            "Pega partidos manuales",
            placeholder="LCK | T1 vs Gen.G\nLPL | Top Esports vs JD Gaming",
            height=160,
        )
        col_a, col_b = st.columns(2)
        send_telegram = col_a.checkbox("Enviar a Telegram", value=False)
        merge_snapshot = col_b.checkbox("Mezclar con snapshot actual", value=True)
        if st.button("Score manual y actualizar snapshot", use_container_width=True):
            try:
                manual_frame = parse_manual_matches_text(manual_text)
                scored = FairOddsEngine(bot_config).score_manual_matches(manual_frame)
                combined = save_predictions_snapshot(
                    scored,
                    dashboard_config.predictions_output_path,
                    merge=merge_snapshot,
                )
                sent = 0
                if send_telegram:
                    sent = send_alerts_to_telegram(bot_config, scored)
                st.success(
                    f"Snapshot actualizado con {len(scored)} partidos manuales. Total snapshot: {len(combined)}. Telegram: {sent} alertas."
                )
            except Exception as exc:
                st.error(str(exc))


def render_pending_pick_decision(bot_config: BotConfig) -> None:
    pending_pick = st.session_state.get("pending_pick")
    if pending_pick is None:
        return

    st.info("Pick calculado pendiente de accion. Confirma si quieres registrarlo como real, guardarlo como paper trading o descartarlo.")

    with st.container(border=True):
        st.markdown(
            f"**{pending_pick.get('match_name', '-')}**  \n"
            f"{pending_pick.get('league', '-')} | {pending_pick.get('market', '-')} | {pending_pick.get('selection', '-')}"
        )
        metric_cols = st.columns(5)
        metric_cols[0].metric("Probabilidad", format_percentage(pending_pick.get("model_probability")))
        metric_cols[1].metric("Cuota justa", format_currency(pending_pick.get("model_fair_odd")))
        metric_cols[2].metric("Edge", format_percentage(pending_pick.get("edge")))
        metric_cols[3].metric("EV", format_percentage(pending_pick.get("ev")))
        metric_cols[4].metric("Stake Kelly", format_currency(pending_pick.get("recommended_stake")))

    real_col, paper_col, cancel_col = st.columns(3)
    if real_col.button("💰 Registrar Apuesta Real", use_container_width=True):
        payload = dict(pending_pick)
        payload["bet_type"] = "real"
        try:
            insert_pick_to_supabase(bot_config, payload)
            st.session_state["pending_pick"] = None
            load_dashboard_data.clear()
            st.success("Apuesta real registrada.")
        except Exception as exc:
            st.error(str(exc))

    if paper_col.button("👁️ Solo Seguimiento (Paper)", use_container_width=True):
        payload = dict(pending_pick)
        payload["bet_type"] = "paper"
        try:
            insert_pick_to_supabase(bot_config, payload)
            st.session_state["pending_pick"] = None
            load_dashboard_data.clear()
            st.success("Pick paper registrado.")
        except Exception as exc:
            st.error(str(exc))

    if cancel_col.button("❌ Cancelar / Nuevo Calculo", use_container_width=True):
        st.session_state["pending_pick"] = None
        st.rerun()


def render_lol_pick_metrics(payload: dict[str, object]) -> None:
    metric_cols = st.columns(5)
    metric_cols[0].metric("Probabilidad", format_percentage(payload.get("model_probability")))
    metric_cols[1].metric("Cuota justa", format_currency(payload.get("model_fair_odd")))
    metric_cols[2].metric("Edge", format_percentage(payload.get("edge")))
    metric_cols[3].metric("EV", format_percentage(payload.get("ev")))
    metric_cols[4].metric("Stake Kelly", format_currency(payload.get("recommended_stake")))


def register_lol_pick(bot_config: BotConfig, payload: dict[str, object], bet_type: str) -> None:
    stored_payload = dict(payload)
    stored_payload["bet_type"] = bet_type
    insert_pick_to_supabase(bot_config, stored_payload)
    load_dashboard_data.clear()


def render_register_page(bot_config: BotConfig, predictions_frame: pd.DataFrame, current_bankroll: float) -> None:
    st.subheader("Registrar pick")
    st.caption(f"Bankroll dinamico actual: {current_bankroll:.2f} EUR")

    if predictions_frame.empty:
        st.info("No hay snapshot de predicciones. Primero haz un scan.")
        return

    match_options = predictions_frame["match_name"].dropna().astype(str).tolist()
    preselected_match = st.session_state.get("register_match_name")
    if "lol_register_match_name" not in st.session_state:
        st.session_state["lol_register_match_name"] = preselected_match if preselected_match in match_options else match_options[0]
    if st.session_state["lol_register_match_name"] not in match_options:
        st.session_state["lol_register_match_name"] = match_options[0]
    match_name = st.selectbox("Partido", match_options, key="lol_register_match_name")
    st.session_state["register_match_name"] = match_name
    row = predictions_frame.loc[predictions_frame["match_name"].astype(str) == match_name].iloc[0]

    st.markdown(
        f"""
        <div class="control-box">
        <strong>{row['match_name']}</strong><br/>
        Liga: {row['league']}<br/>
        Winner: Equipo 1 @{row['winner_team_a_fair_odd']:.2f} | Equipo 2 @{row['winner_team_b_fair_odd']:.2f}<br/>
        {'First Blood: Equipo 1 @%.2f | Equipo 2 @%.2f<br/>' % (row['firstblood_team_a_fair_odd'], row['firstblood_team_b_fair_odd']) if row['league'] != 'LPL' else 'First Blood: desactivado en LPL<br/>'}
        First Tower: Equipo 1 @{row['firsttower_team_a_fair_odd']:.2f} | Equipo 2 @{row['firsttower_team_b_fair_odd']:.2f}<br/>
        Total Kills: {row['kills_prediction']:.1f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    pick_mode = st.radio(
        "Modo",
        ["Registrar mercado binario", "Evaluar kills", "Registrar kills"],
        horizontal=True,
        key="lol_register_mode",
    )

    if pick_mode == "Registrar mercado binario":
        available_markets = ["winner", "firsttower"] + ([] if row["league"] == "LPL" else ["firstblood"])
        if st.session_state.get("lol_register_market") not in available_markets:
            st.session_state["lol_register_market"] = available_markets[0]
        market = st.selectbox("Mercado", available_markets, key="lol_register_market")
        selection_options = [f"1 - {row['team_a']}", f"2 - {row['team_b']}"]
        if st.session_state.get("lol_register_selection") not in selection_options:
            st.session_state["lol_register_selection"] = selection_options[0]
        selection_label = st.radio(
            "Seleccion",
            selection_options,
            horizontal=True,
            key="lol_register_selection",
        )
        odd = st.number_input("Cuota", min_value=1.01, step=0.01, format="%.2f", key="lol_register_odd")
        payload: dict[str, object] | None = None
        try:
            selection_key = "1" if selection_label.startswith("1") else "2"
            payload = compute_pick_payload_from_row(
                bot_config,
                row,
                market,
                selection_key,
                float(odd),
                current_bankroll,
            )
            render_lol_pick_metrics(payload)
            if float(payload["recommended_stake"]) <= 0:
                st.warning("Sin valor")
        except Exception as exc:
            st.error(str(exc))

        real_col, paper_col = st.columns(2)
        can_register = payload is not None and float(payload["recommended_stake"]) > 0
        if real_col.button("Registrar real", use_container_width=True, disabled=not can_register):
            try:
                register_lol_pick(bot_config, payload, "real")
                st.success("Apuesta real registrada.")
            except Exception as exc:
                st.error(str(exc))
        if paper_col.button("Registrar paper", use_container_width=True, disabled=not can_register):
            try:
                register_lol_pick(bot_config, payload, "paper")
                st.success("Pick paper registrado.")
            except Exception as exc:
                st.error(str(exc))

    elif pick_mode == "Evaluar kills":
        st.caption("Introduce varias lineas con formato: linea over under")
        kills_text = st.text_area(
            "Lineas kills",
            placeholder="28.5 1.68 2.00\n29.5 1.84 1.86\n30.5 2.00 1.70",
            height=160,
            key="lol_evaluate_kills_text",
        )
        if st.button("Evaluar lineas kills", use_container_width=True):
            try:
                evaluations: list[dict[str, object]] = []
                for raw_line in kills_text.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.replace(",", ".").split()
                    if len(parts) != 3:
                        raise LoLPipelineError("Cada linea debe ser: linea over under")
                    line_value, over_odd, under_odd = map(float, parts)
                    evaluations.append(compute_pick_payload_from_row(bot_config, row, "kills", "Over", over_odd, current_bankroll, line_value))
                    evaluations.append(compute_pick_payload_from_row(bot_config, row, "kills", "Under", under_odd, current_bankroll, line_value))
                eval_frame = pd.DataFrame(evaluations)
                eval_frame = eval_frame.loc[eval_frame["recommended_stake"] > 0].sort_values(
                    ["recommended_stake", "ev"], ascending=False
                )
                if eval_frame.empty:
                    st.info("Sin valor en ninguna linea")
                else:
                    st.dataframe(
                        eval_frame[
                            ["selection", "line", "my_odd", "model_fair_odd", "edge", "ev", "recommended_stake"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as exc:
                st.error(str(exc))

    else:
        side = st.radio("Lado", ["Over", "Under"], horizontal=True, key="lol_kills_side")
        line_value = st.number_input("Linea", min_value=1.0, step=0.5, format="%.1f", key="lol_kills_line")
        odd = st.number_input("Cuota kills", min_value=1.01, step=0.01, format="%.2f", key="kills_odd")
        payload = None
        try:
            payload = compute_pick_payload_from_row(
                bot_config,
                row,
                "kills",
                side,
                float(odd),
                current_bankroll,
                float(line_value),
            )
            render_lol_pick_metrics(payload)
            if float(payload["recommended_stake"]) <= 0:
                st.warning("Sin valor")
        except Exception as exc:
            st.error(str(exc))

        real_col, paper_col = st.columns(2)
        can_register = payload is not None and float(payload["recommended_stake"]) > 0
        if real_col.button("Registrar kills real", use_container_width=True, disabled=not can_register):
            try:
                register_lol_pick(bot_config, payload, "real")
                st.success("Apuesta real registrada.")
            except Exception as exc:
                st.error(str(exc))
        if paper_col.button("Registrar kills paper", use_container_width=True, disabled=not can_register):
            try:
                register_lol_pick(bot_config, payload, "paper")
                st.success("Pick paper registrado.")
            except Exception as exc:
                st.error(str(exc))


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

    if st.button("Registrar en bets_log", use_container_width=True):
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
            st.success("Apuesta registrada en tennis.bets_log.")
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


def render_lol_dashboard() -> None:
    add_dashboard_styles()
    st.title("BetViero LoL Dashboard")
    st.caption("Centro de mando: panel, control y registro sin depender de Telegram.")

    try:
        bets_frame, picks_frame, predictions_frame, results_frame = load_dashboard_data()
        bot_config = load_bot_config()
        dashboard_config = load_dashboard_config()
    except Exception as exc:
        st.error(f"No pude cargar datos: {exc}")
        return

    unified_records = build_unified_records(bets_frame, picks_frame, results_frame)
    unified_records = enrich_event_dates(unified_records, predictions_frame)

    if st.session_state.get("force_register_page"):
        st.session_state["dashboard_page"] = "Registrar"
        st.session_state["force_register_page"] = False

    with st.sidebar:
        st.header("Navegacion")
        page = st.radio(
            "Pantalla",
            ["Resumen", "Tarjetas", "Historial", "Predicciones", "Control", "Registrar"],
            key="dashboard_page",
        )

        st.header("Filtros")
        investment_mode = st.radio(
            "Modo de Inversion",
            ["Real", "Paper", "Combinado"],
            index=0,
        )
        source_options = sorted(unified_records["source_table"].dropna().unique().tolist()) if not unified_records.empty else []
        league_options = sorted([league for league in unified_records["league"].dropna().unique().tolist() if league and league != "<NA>"]) if not unified_records.empty else []
        market_options = sorted(unified_records["market"].dropna().unique().tolist()) if not unified_records.empty else []
        status_options = sorted(unified_records["status"].dropna().unique().tolist()) if not unified_records.empty else []

        source_filter = st.multiselect("Tabla", source_options, default=source_options)
        league_filter = st.multiselect("Liga", league_options, default=league_options)
        market_filter = st.multiselect("Mercado", market_options, default=market_options)
        status_filter = st.multiselect("Estado", status_options, default=status_options)

        if not unified_records.empty and unified_records["created_at"].notna().any():
            min_date = unified_records["created_at"].min().date()
            max_date = unified_records["created_at"].max().date()
            date_range = st.date_input(
                "Rango de fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = min_date
                end_date = max_date
        else:
            start_date = None
            end_date = None

    filtered_records = apply_filters(
        unified_records,
        source_filter=source_filter,
        league_filter=league_filter,
        market_filter=market_filter,
        status_filter=status_filter,
    )
    filtered_records = filter_by_date(filtered_records, start_date, end_date)
    filtered_records = filter_by_investment_mode(filtered_records, investment_mode)
    current_bankroll = compute_current_bankroll(filtered_records, bot_config.bankroll_amount)
    register_bankroll_records = filter_by_investment_mode(unified_records, investment_mode)
    register_bankroll = compute_current_bankroll(register_bankroll_records, bot_config.bankroll_amount)
    summary_metrics = compute_summary_metrics(filtered_records)
    summary_metrics["current_bankroll"] = current_bankroll

    if page == "Resumen":
        render_summary_metrics(summary_metrics)
        render_overview(filtered_records)
    elif page == "Tarjetas":
        render_cards(filtered_records)
    elif page == "Historial":
        render_history_table(filtered_records)
    elif page == "Predicciones":
        render_predictions(predictions_frame)
    elif page == "Control":
        render_control_page(bot_config, dashboard_config, predictions_frame)
    else:
        render_register_page(bot_config, predictions_frame, register_bankroll)

    st.subheader("Notas")
    st.write("- `bets` muestra solo legacy value bets: `my_odd > model_fair_odd`.")
    st.write("- Las `bets` legacy se liquidan contra Oracle 2026 con stake fijo de `1.0`.")
    st.write("- `picks` se liquidan contra Oracle 2026 para `winner`, `firstblood`, `firsttower` y `kills`.")
    st.write("- El bankroll ya es dinamico: `bankroll inicial + PnL liquidado`.")
    st.write("- Con tu key actual de PandaScore, `games` devuelve `403`, así que no podemos cerrar automaticamente `firstblood/firsttower/kills` desde PandaScore.")
    st.write("- Para movil y escritorio, una web es mejor que una APK ahora mismo. Si luego quieres, se puede empaquetar como PWA o wrapper.")


def check_password() -> bool:
    def get_expected_password() -> str | None:
        try:
            return st.secrets["APP_PASSWORD"]
        except Exception:
            return None

    def password_entered() -> None:
        expected_password = get_expected_password()
        if not expected_password:
            st.session_state["password_correct"] = False
            return

        if st.session_state.get("app_password") == expected_password:
            st.session_state["password_correct"] = True
            del st.session_state["app_password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct"):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="app_password")

    if "password_correct" in st.session_state:
        if not get_expected_password():
            st.error("Falta configurar APP_PASSWORD en Streamlit secrets.")
        else:
            st.error("Password incorrecto.")

    return False


def main() -> None:
    if not check_password():
        return

    sport = st.sidebar.selectbox(
        "Deporte",
        ["League of Legends", "Tenis"],
        index=0,
    )

    if sport == "Tenis":
        render_tennis_dashboard()
        return

    render_lol_dashboard()


if __name__ == "__main__":
    main()
