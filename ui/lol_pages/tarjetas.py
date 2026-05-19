from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from lol.src.dashboard import update_table_row
from ui.lol_pages.shared import load_dashboard_data
from ui.shared import format_currency, format_percentage

def render_cards(records: pd.DataFrame, dashboard_config=None) -> None:
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
        source_table_safe = html.escape(str(row.get("source_table", "-")).upper())
        bet_type_safe = html.escape(str(row.get("bet_type", "real")).upper())
        league_safe = html.escape(str(row.get("league", "-")))
        status_safe = html.escape(status.upper())
        match_name_safe = html.escape(str(row.get("match_name", "-")))
        market_safe = html.escape(str(row.get("market", "-")))
        selection_safe = html.escape(str(row.get("selection", "-")))
        st.markdown(
            f"""
            <div class="bet-card">
              <div class="bet-card-top">
                <div>{source_table_safe} | {bet_type_safe} | {league_safe}</div>
                <div class="{status_class}">{status_safe}</div>
              </div>
              <div class="bet-card-match">{match_name_safe}</div>
              <div class="bet-card-market">{market_safe} | {selection_safe}</div>
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
        if (
            dashboard_config is not None
            and str(row.get("source_table", "")).lower() == "picks"
            and status == "pending"
        ):
            action_cols = st.columns(3)
            for result_status, column in zip(["won", "lost", "void"], action_cols):
                if column.button(result_status.upper(), key=f"card_settle_{row.get('record_id')}_{result_status}", use_container_width=True):
                    try:
                        update_table_row(
                            dashboard_config,
                            "picks",
                            str(row["record_id"]),
                            {"status": result_status},
                        )
                        load_dashboard_data.clear()
                        st.success(f"Pick marcado como {result_status}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"No pude actualizar el pick: {exc}")
