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


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
LOL_DIR = PROJECT_ROOT / "lol"
TENNIS_DIR = PROJECT_ROOT / "tennis"

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


def render_register_page(bot_config: BotConfig, predictions_frame: pd.DataFrame, current_bankroll: float) -> None:
    st.subheader("Registrar pick")
    st.caption(f"Bankroll dinamico actual: {current_bankroll:.2f} EUR")

    if st.session_state.get("pending_pick") is not None:
        render_pending_pick_decision(bot_config)
        return

    if predictions_frame.empty:
        st.info("No hay snapshot de predicciones. Primero haz un scan.")
        return

    match_options = predictions_frame["match_name"].dropna().astype(str).tolist()
    preselected_match = st.session_state.get("register_match_name")
    match_index = match_options.index(preselected_match) if preselected_match in match_options else 0
    match_name = st.selectbox("Partido", match_options, index=match_index)
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

    pick_mode = st.radio("Modo", ["Registrar mercado binario", "Evaluar kills", "Registrar kills"], horizontal=True)

    if pick_mode == "Registrar mercado binario":
        available_markets = ["winner", "firsttower"] + ([] if row["league"] == "LPL" else ["firstblood"])
        market = st.selectbox("Mercado", available_markets)
        selection_label = st.radio(
            "Seleccion",
            [f"1 - {row['team_a']}", f"2 - {row['team_b']}"],
            horizontal=True,
        )
        odd = st.number_input("Cuota", min_value=1.01, step=0.01, format="%.2f")
        if st.button("Calcular Valor", use_container_width=True):
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
                if payload["recommended_stake"] <= 0:
                    st.warning("Sin valor")
                else:
                    st.session_state["pending_pick"] = payload
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))

    elif pick_mode == "Evaluar kills":
        st.caption("Introduce varias lineas con formato: linea over under")
        kills_text = st.text_area(
            "Lineas kills",
            placeholder="28.5 1.68 2.00\n29.5 1.84 1.86\n30.5 2.00 1.70",
            height=160,
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
        side = st.radio("Lado", ["Over", "Under"], horizontal=True)
        line_value = st.number_input("Linea", min_value=1.0, step=0.5, format="%.1f")
        odd = st.number_input("Cuota kills", min_value=1.01, step=0.01, format="%.2f", key="kills_odd")
        if st.button("Calcular Valor", use_container_width=True, key="calculate_kills_value"):
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
                if payload["recommended_stake"] <= 0:
                    st.warning("Sin valor")
                else:
                    st.session_state["pending_pick"] = payload
                    st.rerun()
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
    ev_value = pd.to_numeric(row.get("EV", 0), errors="coerce")
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
                st.success(result.stdout.strip() or "Value bets de tenis actualizadas.")
            else:
                st.error(result.stderr.strip() or "No se pudo calcular EV de tenis.")

    value_bets = add_tennis_kelly_column(load_tennis_csv(str(TENNIS_VALUE_BETS_PATH)))
    sackmann_frame = load_tennis_csv(str(TENNIS_SACKMANN_PATH))
    features_frame = load_tennis_csv(str(TENNIS_FEATURES_PATH))

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Historico limpio", len(sackmann_frame))
    metric_2.metric("Features avanzadas", len(features_frame))
    metric_3.metric("Value rows", len(value_bets))
    metric_4.metric("EV > 0", int((pd.to_numeric(value_bets.get("EV", pd.Series(dtype=float)), errors="coerce") > 0).sum()))

    render_tennis_diagnostics()

    st.subheader("Value Bets Tenis")
    if value_bets.empty:
        st.info("No hay value bets calculadas todavia. Usa `Cazar Cuotas (Calcular EV)`.")
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
            "Kelly Recomendado": "{:.2f} EUR",
            "p_serve_A": "{:.2%}",
            "p_serve_B": "{:.2%}",
            "mean_total_games": "{:.2f}",
            "mean_game_diff": "{:.2f}",
        },
        na_rep="-",
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


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
