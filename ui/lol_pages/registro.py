from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from lol.src.base import LoLPipelineError
from lol.src.fair_odds_bot import BotConfig, insert_pick_to_supabase
from lol.src.multimarket import (
    KILLS_LINES,
    calculate_ev,
    calculate_fair_odd,
    calculate_implied_probability,
    calculate_kelly_fraction,
)
from ui.lol_pages.shared import filter_prediction_view, load_dashboard_data
from ui.shared import format_currency, format_percentage

def _kills_prob_column(line: float) -> str:
    return f"prob_over_{str(float(line)).replace('.', '_')}"

def get_kills_over_probability_from_row(row: pd.Series, line: float) -> float:
    line_value = float(line)
    trained_lines = sorted(float(value) for value in KILLS_LINES)
    if line_value in trained_lines:
        column = _kills_prob_column(line_value)
        if column not in row or pd.isna(row[column]):
            raise LoLPipelineError(f"El snapshot no tiene probabilidad de kills para linea {line_value}.")
        return float(row[column])

    if line_value < trained_lines[0]:
        lower_line, upper_line = trained_lines[0], trained_lines[1]
    elif line_value > trained_lines[-1]:
        lower_line, upper_line = trained_lines[-2], trained_lines[-1]
    else:
        lower_line = max(value for value in trained_lines if value < line_value)
        upper_line = min(value for value in trained_lines if value > line_value)

    lower_prob = get_kills_over_probability_from_row(row, lower_line)
    upper_prob = get_kills_over_probability_from_row(row, upper_line)
    interpolated = lower_prob + (upper_prob - lower_prob) * (
        (line_value - lower_line) / (upper_line - lower_line)
    )
    return min(1.0, max(0.0, float(interpolated)))

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
        over_probability = get_kills_over_probability_from_row(row, float(line))
        probability = over_probability if selection_key.lower() == "over" else 1.0 - over_probability
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

def _escape_html_value(value: object) -> str:
    return html.escape(str(value))

def _format_html_float(value: object, decimals: int = 2) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "-"
    return html.escape(f"{float(numeric_value):.{decimals}f}")

def render_register_page(bot_config: BotConfig, predictions_frame: pd.DataFrame, current_bankroll: float) -> None:
    st.subheader("Registrar pick")
    st.caption(f"Bankroll dinamico actual: {current_bankroll:.2f} EUR")

    predictions_frame = filter_prediction_view(predictions_frame)
    if predictions_frame.empty:
        st.info("No hay predicciones de partidos próximos. Primero genera partidos desde Control.")
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
    firstblood_summary = (
        "First Blood: desactivado en LPL<br/>"
        if row["league"] == "LPL"
        else (
            "First Blood: Equipo 1 @"
            f"{_format_html_float(row['firstblood_team_a_fair_odd'])} | Equipo 2 @"
            f"{_format_html_float(row['firstblood_team_b_fair_odd'])}<br/>"
        )
    )

    st.markdown(
        f"""
        <div class="control-box">
        <strong>{_escape_html_value(row['match_name'])}</strong><br/>
        Liga: {_escape_html_value(row['league'])}<br/>
        Winner: Equipo 1 @{_format_html_float(row['winner_team_a_fair_odd'])} | Equipo 2 @{_format_html_float(row['winner_team_b_fair_odd'])}<br/>
        {firstblood_summary}
        First Tower: Equipo 1 @{_format_html_float(row['firsttower_team_a_fair_odd'])} | Equipo 2 @{_format_html_float(row['firsttower_team_b_fair_odd'])}<br/>
        Total Kills: {_format_html_float(row['kills_prediction'], 1)}
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
