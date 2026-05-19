from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.lol_pages.shared import filter_prediction_view, load_dashboard_data
from ui.shared import format_currency

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

def render_predictions(predictions_frame: pd.DataFrame, dashboard_config) -> None:
    if predictions_frame.empty:
        st.info("No hay snapshot de predicciones disponible.")
        return
    upcoming = filter_prediction_view(predictions_frame)
    if upcoming.empty:
        st.info("No hay predicciones de partidos próximos. Genera nuevas predicciones desde Control.")
        return

    st.caption(f"Mostrando {len(upcoming)} partidos próximos.")
    for index, row in upcoming.iterrows():
        scheduled_at = row.get("scheduled_at")
        scheduled_text = scheduled_at.strftime("%Y-%m-%d %H:%M") if pd.notna(scheduled_at) else "-"
        with st.container(border=True):
            top_col, load_col, remove_col = st.columns([4, 1, 1])
            top_col.markdown(
                f"**{row.get('match_name', '-')}**  \n"
                f"{row.get('league', '-')} | {scheduled_text}"
            )
            if load_col.button("Cargar", key=f"load_lol_prediction_{index}", use_container_width=True):
                st.session_state["register_match_name"] = str(row.get("match_name", ""))
                st.session_state["lol_register_match_name"] = str(row.get("match_name", ""))
                st.session_state["force_register_page"] = True
                st.rerun()
            if remove_col.button("Quitar", key=f"remove_lol_prediction_{index}", use_container_width=True):
                remaining = predictions_frame.copy()
                if "source_match_id" in remaining.columns and pd.notna(row.get("source_match_id")):
                    remaining = remaining.loc[remaining["source_match_id"].astype(str) != str(row.get("source_match_id"))]
                else:
                    remaining = remaining.loc[remaining["match_name"].astype(str) != str(row.get("match_name", ""))]
                remaining.reset_index(drop=True).to_csv(dashboard_config.predictions_output_path, index=False)
                load_dashboard_data.clear()
                st.rerun()

            metric_cols = st.columns(4)
            metric_cols[0].metric("Winner 1", format_currency(row.get("winner_team_a_fair_odd")))
            metric_cols[1].metric("Winner 2", format_currency(row.get("winner_team_b_fair_odd")))
            metric_cols[2].metric("First Tower 1/2", f"{format_currency(row.get('firsttower_team_a_fair_odd'))} / {format_currency(row.get('firsttower_team_b_fair_odd'))}")
            metric_cols[3].metric("Kills", format_currency(row.get("kills_prediction")))
