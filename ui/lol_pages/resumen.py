from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.shared import format_currency, format_percentage

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

def render_ev_vs_pnl_chart(records: pd.DataFrame) -> None:
    settled = records.loc[records["status"].isin(["won", "lost", "void"])].copy()
    if settled.empty:
        st.info("No hay apuestas liquidadas para comparar EV teorico y PnL real.")
        return

    date_column = "event_date" if "event_date" in settled.columns and settled["event_date"].notna().any() else "created_at"
    settled[date_column] = pd.to_datetime(settled[date_column], utc=True, errors="coerce")
    settled = settled.loc[settled[date_column].notna()].sort_values(date_column).copy()
    if settled.empty:
        st.info("No hay fechas validas para pintar la evolucion de EV y PnL.")
        return

    settled["bet_type"] = settled.get("bet_type", "real")
    settled["bet_type"] = settled["bet_type"].fillna("real").astype(str).str.lower()
    settled["pnl_numeric"] = pd.to_numeric(
        settled["pnl"],
        errors="coerce",
    ).fillna(0.0)
    settled["expected_profit"] = pd.to_numeric(settled["ev"], errors="coerce").fillna(0.0) * pd.to_numeric(
        settled["recommended_stake"],
        errors="coerce",
    ).fillna(0.0)
    settled["PnL Acumulado"] = settled.groupby("bet_type")["pnl_numeric"].cumsum()
    settled["EV Teorico Acumulado"] = settled.groupby("bet_type")["expected_profit"].cumsum()

    chart_parts: list[pd.DataFrame] = []
    for bet_type, group in settled.groupby("bet_type", sort=True):
        label = "Real" if bet_type == "real" else "Paper" if bet_type == "paper" else bet_type.title()
        chart_parts.append(
            group.set_index(date_column)[["EV Teorico Acumulado", "PnL Acumulado"]].rename(
                columns={
                    "EV Teorico Acumulado": f"EV Teorico {label}",
                    "PnL Acumulado": f"PnL {label}",
                }
            )
        )

    chart_frame = pd.concat(chart_parts, axis=1).sort_index()
    chart_frame["Break-even"] = 0.0

    st.subheader("EV teorico vs PnL real")
    st.line_chart(chart_frame)

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

    render_ev_vs_pnl_chart(records)
