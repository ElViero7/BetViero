from __future__ import annotations

from datetime import date
import html
import os
from pathlib import Path
import re

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
    KILLS_LINES,
    calculate_ev,
    calculate_fair_odd,
    calculate_implied_probability,
    calculate_kelly_fraction,
    train_multimarket_pipeline,
    MultiMarketTrainingConfig,
)
from lol.src.vision_parser import detect_matchups_from_ocr_lines, extract_matchups_from_image
from lol.src.vision_parser import apply_ocr_team_corrections, clean_ocr_team_text
from ui.shared import add_dashboard_styles, format_currency, format_percentage


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOL_DIR = PROJECT_ROOT / "lol"

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


@st.cache_data(ttl=60)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_dashboard_config()
    bets_frame = fetch_table(config, "bets")
    picks_frame = fetch_table(config, "picks")
    predictions_frame = load_predictions_snapshot(config)
    results_frame = build_oracle_results_frame(config.raw_data_dir)
    return bets_frame, picks_frame, predictions_frame, results_frame


@st.cache_resource
def get_fair_odds_engine(_config: BotConfig, _cache_version: str = "kills-line-fallback-v2") -> FairOddsEngine:
    return FairOddsEngine(_config)


def get_lol_manual_options(bot_config: BotConfig) -> dict[str, list[str]]:
    frame = get_fair_odds_engine(bot_config).feature_store.team_frame
    if frame.empty:
        return {}
    options: dict[str, list[str]] = {}
    for league, league_frame in frame.groupby("league", dropna=True):
        teams = sorted(league_frame["team_name"].dropna().astype(str).unique().tolist())
        if teams:
            options[str(league)] = teams
    return dict(sorted(options.items()))


def build_manual_source_match_id(league: str, team_a: str, team_b: str, scheduled_at: str) -> str:
    raw_key = f"{league}::{team_a}::{team_b}::{scheduled_at}"
    slug = re.sub(r"[^a-z0-9]+", "-", raw_key.lower()).strip("-")
    return f"manual::{slug}"


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
    frame_to_save = frame.copy()
    if not frame_to_save.empty and "source_match_id" in frame_to_save.columns:
        manual_mask = frame_to_save["source_match_id"].astype(str).str.startswith("manual::")
        if manual_mask.any() and "snapshot_source" not in frame_to_save.columns:
            frame_to_save["snapshot_source"] = "manual_remesa"
    if merge and output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, frame_to_save], ignore_index=True)
        combined = combined.drop_duplicates(subset=["match_name"], keep="last")
    else:
        combined = frame_to_save
    combined = combined.sort_values(["league", "scheduled_at", "match_name"], na_position="last").reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    load_dashboard_data.clear()
    return combined


def filter_upcoming_predictions(predictions_frame: pd.DataFrame) -> pd.DataFrame:
    if predictions_frame.empty or "scheduled_at" not in predictions_frame.columns:
        return predictions_frame
    output = predictions_frame.copy()
    output["scheduled_at"] = pd.to_datetime(output["scheduled_at"], utc=True, errors="coerce")
    today_utc = pd.Timestamp.now(tz="UTC").normalize()
    output = output.loc[output["scheduled_at"].isna() | (output["scheduled_at"] >= today_utc)].copy()
    return output.sort_values(["scheduled_at", "league", "match_name"], na_position="last").reset_index(drop=True)


def filter_prediction_view(predictions_frame: pd.DataFrame) -> pd.DataFrame:
    if predictions_frame.empty:
        return predictions_frame
    if "source_match_id" in predictions_frame.columns:
        manual_mask = predictions_frame["source_match_id"].astype(str).str.startswith("manual::")
        snapshot_is_manual = False
        if "snapshot_source" in predictions_frame.columns:
            snapshot_is_manual = predictions_frame["snapshot_source"].astype(str).eq("manual_remesa").any()
        if snapshot_is_manual or manual_mask.all():
            output = predictions_frame.loc[manual_mask].copy()
            if "scheduled_at" in output.columns:
                output["scheduled_at"] = pd.to_datetime(output["scheduled_at"], utc=True, errors="coerce")
                output = output.sort_values(["scheduled_at", "league", "match_name"], na_position="last")
            return output.reset_index(drop=True)
    return filter_upcoming_predictions(predictions_frame)


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


def render_manual_prediction_builder(
    bot_config: BotConfig,
    dashboard_config,
    predictions_frame: pd.DataFrame,
) -> None:
    st.markdown("**Remesa manual de partidos**")
    options = get_lol_manual_options(bot_config)
    if not options:
        st.warning("No hay histórico de equipos disponible para construir partidos manuales.")
        return

    if "lol_manual_matches" not in st.session_state:
        st.session_state["lol_manual_matches"] = []
    success_message = st.session_state.pop("lol_manual_snapshot_success", None)
    if success_message:
        st.success(success_message)

    league_options = list(options)
    league = st.selectbox("Liga", league_options, key="lol_manual_league")
    team_options = options.get(league, [])
    col_a, col_b, col_c = st.columns(3)
    team_a = col_a.selectbox("Equipo 1", team_options, key=f"lol_manual_team_a_{league}")
    team_b_default_index = 1 if len(team_options) > 1 else 0
    team_b = col_b.selectbox(
        "Equipo 2",
        team_options,
        index=team_b_default_index,
        key=f"lol_manual_team_b_{league}",
    )
    scheduled_date = col_c.date_input("Fecha", value=date.today(), key="lol_manual_date")
    submitted = st.button("Guardar en remesa", use_container_width=True)

    if submitted:
        if team_a == team_b:
            st.error("Selecciona dos equipos distintos.")
        else:
            scheduled_at = pd.Timestamp(scheduled_date).tz_localize("UTC").isoformat()
            match_payload = {
                "league": league,
                "team_a": team_a,
                "team_b": team_b,
                "scheduled_at": scheduled_at,
                "source_match_id": build_manual_source_match_id(league, team_a, team_b, scheduled_at),
            }
            existing_keys = {
                item["source_match_id"]
                for item in st.session_state["lol_manual_matches"]
            }
            match_key = match_payload["source_match_id"]
            if match_key not in existing_keys:
                st.session_state["lol_manual_matches"].append(match_payload)
                st.success(f"Partido guardado en remesa: {team_a} vs {team_b}")

    selected_matches = pd.DataFrame(st.session_state["lol_manual_matches"])
    if selected_matches.empty:
        st.info("Guarda partidos en la remesa para previsualizarlos antes de actualizar el snapshot.")
        return

    st.markdown("**Previsualización de la remesa**")
    for index, row in selected_matches.iterrows():
        with st.container(border=True):
            top_col, action_col = st.columns([4, 1])
            top_col.markdown(
                f"**{row['team_a']} vs {row['team_b']}**  \n"
                f"{row['league']} | {row['scheduled_at']}"
            )
            if action_col.button("Quitar", key=f"lol_manual_remove_{index}", use_container_width=True):
                st.session_state["lol_manual_matches"].pop(index)
                st.rerun()

    col_score, col_clear = st.columns(2)
    send_telegram = st.checkbox("Enviar alertas a Telegram", value=False)

    if col_score.button(
        "Actualizar snapshot con esta remesa",
        key="lol_manual_update_snapshot",
        use_container_width=True,
    ):
        try:
            if selected_matches.empty:
                st.error("La remesa esta vacia. Guarda al menos un partido antes de actualizar el snapshot.")
                return

            with st.spinner("Generando predicciones de la remesa..."):
                get_fair_odds_engine.clear()
                scored = get_fair_odds_engine(bot_config).score_manual_matches(selected_matches)

            if scored.empty:
                st.error("El motor no ha generado ninguna prediccion para esta remesa. Revisa liga y equipos.")
                return

            if len(scored) != len(selected_matches):
                st.warning(
                    f"Se han podido generar {len(scored)} predicciones de {len(selected_matches)} partidos en la remesa."
                )
            scored = scored.copy()
            scored["snapshot_source"] = "manual_remesa"
            combined = save_predictions_snapshot(
                scored,
                dashboard_config.predictions_output_path,
                merge=False,
            )
            if not dashboard_config.predictions_output_path.exists():
                st.error(f"No se ha creado el CSV de predicciones: {dashboard_config.predictions_output_path}")
                return
            sent = send_alerts_to_telegram(bot_config, scored) if send_telegram else 0
            st.cache_data.clear()
            st.session_state["lol_manual_snapshot_success"] = (
                f"Snapshot reemplazado con {len(combined)} partidos de la remesa. Telegram: {sent} alertas."
            )
            st.session_state["lol_manual_matches"] = []
            st.session_state["force_predictions_page"] = True
            st.rerun()
        except Exception as exc:
            st.error(f"No pude actualizar el snapshot: {exc}")

    if col_clear.button("Vaciar selección", use_container_width=True):
        st.session_state["lol_manual_matches"] = []
        st.rerun()


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

    settled["PnL Real Acumulado"] = pd.to_numeric(
        settled["pnl"],
        errors="coerce",
    ).fillna(0.0).cumsum()
    expected_profit = pd.to_numeric(settled["ev"], errors="coerce").fillna(0.0) * pd.to_numeric(
        settled["recommended_stake"],
        errors="coerce",
    ).fillna(0.0)
    settled["EV Teórico Acumulado"] = expected_profit.cumsum()
    chart_frame = settled.set_index(date_column)[["EV Teórico Acumulado", "PnL Real Acumulado"]]

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
    upcoming = filter_prediction_view(predictions_frame)
    if upcoming.empty:
        st.info("No hay predicciones de partidos próximos. Genera nuevas predicciones desde Control.")
        return

    st.caption(f"Mostrando {len(upcoming)} partidos próximos.")
    for index, row in upcoming.iterrows():
        scheduled_at = row.get("scheduled_at")
        scheduled_text = scheduled_at.strftime("%Y-%m-%d %H:%M") if pd.notna(scheduled_at) else "-"
        with st.container(border=True):
            top_col, action_col = st.columns([4, 1])
            top_col.markdown(
                f"**{row.get('match_name', '-')}**  \n"
                f"{row.get('league', '-')} | {scheduled_text}"
            )
            if action_col.button("Cargar", key=f"load_lol_prediction_{index}", use_container_width=True):
                st.session_state["register_match_name"] = str(row.get("match_name", ""))
                st.session_state["lol_register_match_name"] = str(row.get("match_name", ""))
                st.session_state["force_register_page"] = True
                st.rerun()

            metric_cols = st.columns(4)
            metric_cols[0].metric("Winner 1", format_currency(row.get("winner_team_a_fair_odd")))
            metric_cols[1].metric("Winner 2", format_currency(row.get("winner_team_b_fair_odd")))
            metric_cols[2].metric("First Tower 1/2", f"{format_currency(row.get('firsttower_team_a_fair_odd'))} / {format_currency(row.get('firsttower_team_b_fair_odd'))}")
            metric_cols[3].metric("Kills", format_currency(row.get("kills_prediction")))


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
                    frame = get_fair_odds_engine(bot_config).score_upcoming_matches()
                    save_predictions_snapshot(frame, dashboard_config.predictions_output_path, merge=False)
                    st.success(f"Snapshot actualizado con {len(frame)} partidos.")
                except Exception as exc:
                    st.error(str(exc))

    render_magic_prediction_capture(predictions_frame)

    with st.container(border=True):
        render_manual_prediction_builder(bot_config, dashboard_config, predictions_frame)


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


def render_lol_results_page(dashboard_config) -> None:
    st.subheader("Actualizar resultados LoL")

    with st.container(border=True):
        st.markdown("**Sincronización con Oracle**")
        col_legacy, col_picks = st.columns(2)
        if col_legacy.button("Sincronizar legacy bets", use_container_width=True):
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
                load_dashboard_data.clear()
                st.success(f"Legacy bets sincronizadas: {updated}")
            except Exception as exc:
                st.error(str(exc))

        if col_picks.button("Sincronizar picks", use_container_width=True):
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
                load_dashboard_data.clear()
                st.success(f"Picks sincronizados: {updated}")
            except Exception as exc:
                st.error(str(exc))

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
        render_predictions(predictions_frame)
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
