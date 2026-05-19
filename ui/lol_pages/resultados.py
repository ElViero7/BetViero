from __future__ import annotations

import pandas as pd
import streamlit as st

from lol.src.dashboard import fetch_table, update_table_row
from ui.lol_pages.shared import _sync_picks_and_bets, load_dashboard_data
from ui.shared import format_currency

def render_lol_results_page(dashboard_config) -> None:
    st.subheader("Actualizar resultados LoL")

    with st.container(border=True):
        st.markdown("**Sincronización con Oracle**")
        _sync_picks_and_bets(dashboard_config)

    with st.container(border=True):
        st.markdown("**Cierre manual de picks**")
        try:
            picks_frame = fetch_table(dashboard_config, "picks")
        except Exception as exc:
            st.error(f"No pude cargar picks: {exc}")
            return

        if picks_frame.empty:
            st.info("No hay picks registrados.")
            return

        pending_picks = picks_frame.loc[
            picks_frame.get("status").fillna("pending").astype(str).str.lower().eq("pending")
        ].copy()
        if pending_picks.empty:
            st.info("No hay picks pendientes.")
            return

        pending_picks["pick_label"] = pending_picks.apply(
            lambda row: (
                f"{row.get('created_at', '-')} | {row.get('match_name', '-')} | "
                f"{row.get('market', '-')} {row.get('selection', '-')}"
            ),
            axis=1,
        )
        selected_label = st.selectbox("Pick pendiente", pending_picks["pick_label"].tolist())
        selected_pick = pending_picks.loc[pending_picks["pick_label"].eq(selected_label)].iloc[0]
        st.dataframe(
            pd.DataFrame([selected_pick.drop(labels=["pick_label"], errors="ignore")]),
            use_container_width=True,
            hide_index=True,
        )

        settle_status = st.selectbox("Resultado", ["won", "lost", "void"])
        stake_value = pd.to_numeric(selected_pick.get("recommended_stake"), errors="coerce")
        odd_value = pd.to_numeric(selected_pick.get("my_odd"), errors="coerce")
        stake = 0.0 if pd.isna(stake_value) else float(stake_value)
        odd = 0.0 if pd.isna(odd_value) else float(odd_value)
        projected_pnl = 0.0
        if settle_status == "won":
            projected_pnl = stake * (odd - 1.0)
        elif settle_status == "lost":
            projected_pnl = -stake
        st.metric("PnL proyectado", format_currency(projected_pnl))

        if st.button("Cerrar pick", use_container_width=True):
            try:
                update_table_row(
                    dashboard_config,
                    "picks",
                    str(selected_pick["id"]),
                    {"status": settle_status},
                )
                load_dashboard_data.clear()
                st.success("Pick cerrado.")
                st.rerun()
            except Exception as exc:
                st.error(f"No pude cerrar el pick: {exc}")
