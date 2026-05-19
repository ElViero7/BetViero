from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from lol.src.base import LoLPipelineError
from lol.src.dashboard import download_oracle_2026_csv
from lol.src.fair_odds_bot import BotConfig
from lol.src.multimarket import MultiMarketTrainingConfig, train_multimarket_pipeline
from lol.src.vision_parser import detect_matchups_from_ocr_lines, extract_matchups_from_image
from ui.lol_pages.shared import (
    LOL_DIR,
    _sync_picks_and_bets,
    build_manual_source_match_id,
    build_valid_team_list,
    find_prediction_match_name,
    get_fair_odds_engine,
    get_lol_manual_options,
    get_lol_model_mtime,
    load_dashboard_data,
    logger,
    save_predictions_snapshot,
    send_alerts_to_telegram,
)

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
                scored = get_fair_odds_engine(bot_config, get_lol_model_mtime()).score_manual_matches(selected_matches)

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
        logger.debug("Equipos validos para fuzzy: %d equipos", len(valid_teams))
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
            _sync_picks_and_bets(dashboard_config)
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
                    frame = get_fair_odds_engine(bot_config, get_lol_model_mtime()).score_upcoming_matches()
                    save_predictions_snapshot(frame, dashboard_config.predictions_output_path, merge=False)
                    st.success(f"Snapshot actualizado con {len(frame)} partidos.")
                except Exception as exc:
                    st.error(str(exc))

    render_magic_prediction_capture(predictions_frame)

    with st.container(border=True):
        render_manual_prediction_builder(bot_config, dashboard_config, predictions_frame)
