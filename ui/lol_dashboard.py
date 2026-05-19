from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from lol.src.dashboard import (
    apply_filters,
    build_unified_records,
    compute_current_bankroll,
    compute_summary_metrics,
    load_dashboard_config,
)
from lol.src.fair_odds_bot import load_config as load_bot_config
from ui.lol_pages.predicciones import render_history_table, render_predictions
from ui.lol_pages.registro import render_register_page
from ui.lol_pages.resumen import render_overview, render_summary_metrics
from ui.lol_pages.resultados import render_lol_results_page
from ui.lol_pages.shared import enrich_event_dates, load_dashboard_data
from ui.lol_pages.tarjetas import render_cards
from ui.lol_pages.control import render_control_page
from ui.shared import add_dashboard_styles


if "pending_pick" not in st.session_state:
    st.session_state["pending_pick"] = None
if "register_match_name" not in st.session_state:
    st.session_state["register_match_name"] = None
if "dashboard_page" not in st.session_state:
    st.session_state["dashboard_page"] = "Resumen"
if "force_register_page" not in st.session_state:
    st.session_state["force_register_page"] = False
if "force_predictions_page" not in st.session_state:
    st.session_state["force_predictions_page"] = False

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
    if st.session_state.get("force_predictions_page"):
        st.session_state["dashboard_page"] = "Predicciones"
        st.session_state["force_predictions_page"] = False

    with st.sidebar:
        st.header("Navegacion")
        page = st.radio(
            "Pantalla",
            ["Resumen", "Tarjetas", "Historial", "Predicciones", "Control", "Registrar", "Resultados"],
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
        render_cards(filtered_records, dashboard_config)
    elif page == "Historial":
        render_history_table(filtered_records)
    elif page == "Predicciones":
        render_predictions(predictions_frame, dashboard_config)
    elif page == "Control":
        render_control_page(bot_config, dashboard_config, predictions_frame)
    elif page == "Registrar":
        render_register_page(bot_config, predictions_frame, register_bankroll)
    else:
        render_lol_results_page(dashboard_config)

    st.subheader("Notas")
    st.write("- `bets` muestra solo legacy value bets: `my_odd > model_fair_odd`.")
    st.write("- Las `bets` legacy se liquidan contra Oracle 2026 con stake fijo de `1.0`.")
    st.write("- `picks` se liquidan contra Oracle 2026 para `winner`, `firstblood`, `firsttower` y `kills`.")
    st.write("- El bankroll ya es dinamico: `bankroll inicial + PnL liquidado`.")
    st.write("- Con tu key actual de PandaScore, `games` devuelve `403`, así que no podemos cerrar automaticamente `firstblood/firsttower/kills` desde PandaScore.")
    st.write("- Para movil y escritorio, una web es mejor que una APK ahora mismo. Si luego quieres, se puede empaquetar como PWA o wrapper.")
