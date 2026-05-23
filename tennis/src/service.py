from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from tennis.src.db import TennisDatabaseError, update_tennis_row
from tennis.src.ev_calculator import get_supabase_client


def _load_tennis_bets_log() -> pd.DataFrame:
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

    for column in [
        "line",
        "my_odd",
        "model_probability",
        "model_fair_odd",
        "edge",
        "ev",
        "stake",
        "kelly_fraction",
        "pnl",
        "clv_pinnacle_odd",
        "clv_edge",
        "clv_value",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["placed_at", "event_time", "settled_at", "clv_captured_at", "created_at", "updated_at"]:
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


def load_tennis_clv_pending_bets() -> pd.DataFrame:
    bets = _load_tennis_bets_log()
    if bets.empty:
        return bets

    if "clv_pinnacle_odd" not in bets.columns:
        bets["clv_pinnacle_odd"] = pd.NA

    status_values = bets.get("status", pd.Series("", index=bets.index)).fillna("").astype(str)
    market_values = bets.get("market_key", pd.Series("", index=bets.index)).fillna("").astype(str)

    pending = bets.loc[
        status_values.eq("pending")
        & market_values.isin(["totals", "spreads"])
        & bets["clv_pinnacle_odd"].isna()
    ].copy()
    return pending.sort_values("placed_at", ascending=False).reset_index(drop=True)


def _coerce_raw_payload(raw_payload: Any) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload.copy()
    if isinstance(raw_payload, str) and raw_payload.strip():
        try:
            decoded = json.loads(raw_payload)
        except json.JSONDecodeError:
            return {"raw_payload_text": raw_payload}
        return decoded if isinstance(decoded, dict) else {"raw_payload": decoded}
    return {}


def _update_bets_log_clv(row_id: str, payload: dict[str, Any]) -> None:
    try:
        update_tennis_row("bets_log", row_id, payload)
    except TennisDatabaseError as exc:
        if "clv_source" not in str(exc):
            raise
        fallback_payload = payload.copy()
        fallback_payload.pop("clv_source", None)
        update_tennis_row("bets_log", row_id, fallback_payload)


def capture_pinnacle_clv_for_bet(bet_id: str) -> dict[str, Any]:
    bets = _load_tennis_bets_log()
    if bets.empty:
        raise ValueError("No hay apuestas cargadas.")

    selected = bets.loc[bets["id"].astype(str).eq(str(bet_id))]
    if selected.empty:
        raise ValueError("No encontre la apuesta seleccionada.")

    bet = selected.iloc[0]
    market_key = str(bet.get("market_key") or "")
    if market_key not in {"totals", "spreads"}:
        raise ValueError(f"Mercado no soportado para CLV Pinnacle: {market_key}")

    from tennis.src.odds_listener import fetch_pinnacle_odds_for_pick

    clv_quote = fetch_pinnacle_odds_for_pick(
        player_1=str(bet.get("player_1") or ""),
        player_2=str(bet.get("player_2") or ""),
        market_key=market_key,
        selection=str(bet.get("selection") or ""),
        line=None if pd.isna(bet.get("line")) else float(bet.get("line")),
    )
    clv_odd = float(clv_quote["price"])
    my_odd = float(bet["my_odd"])
    model_probability = bet.get("model_probability")
    clv_edge = None if pd.isna(model_probability) else float(model_probability) * clv_odd - 1.0
    clv_value = my_odd / clv_odd - 1.0 if clv_odd > 0 else None

    raw_payload = _coerce_raw_payload(bet.get("raw_payload"))
    raw_payload["clv_pinnacle"] = clv_quote

    payload = {
        "clv_pinnacle_odd": clv_odd,
        "clv_captured_at": clv_quote["captured_at"],
        "clv_edge": clv_edge,
        "clv_value": clv_value,
        "clv_source_event_id": clv_quote.get("event_id"),
        "clv_source": "pinnacle_api",
        "raw_payload": raw_payload,
    }
    _update_bets_log_clv(str(bet_id), payload)
    return {
        "bet_id": str(bet_id),
        "clv_pinnacle_odd": clv_odd,
        "clv_value": clv_value,
        "clv_edge": clv_edge,
        "quote": clv_quote,
    }


def record_manual_clv_for_bet(
    bet_id: str,
    *,
    closing_odd: float,
    clv_source: str = "manual",
    notes: str = "",
) -> dict[str, Any]:
    bets = _load_tennis_bets_log()
    if bets.empty:
        raise ValueError("No hay apuestas cargadas.")

    selected = bets.loc[bets["id"].astype(str).eq(str(bet_id))]
    if selected.empty:
        raise ValueError("No encontre la apuesta seleccionada.")
    if closing_odd <= 1.0:
        raise ValueError("La cuota de cierre debe ser mayor que 1.0.")

    bet = selected.iloc[0]
    my_odd = float(bet["my_odd"])
    model_probability = bet.get("model_probability")
    clv_edge = None if pd.isna(model_probability) else float(model_probability) * closing_odd - 1.0
    clv_value = my_odd / closing_odd - 1.0
    captured_at = datetime.now(timezone.utc).isoformat()

    raw_payload = _coerce_raw_payload(bet.get("raw_payload"))
    raw_payload["clv_manual"] = {
        "price": float(closing_odd),
        "source": clv_source,
        "captured_at": captured_at,
        "notes": notes.strip() or None,
    }

    payload = {
        "clv_pinnacle_odd": float(closing_odd),
        "clv_captured_at": captured_at,
        "clv_edge": clv_edge,
        "clv_value": clv_value,
        "clv_source": clv_source,
        "raw_payload": raw_payload,
    }
    _update_bets_log_clv(str(bet_id), payload)
    return {
        "bet_id": str(bet_id),
        "clv_pinnacle_odd": float(closing_odd),
        "clv_value": clv_value,
        "clv_edge": clv_edge,
        "clv_source": clv_source,
    }


def _load_tennis_picks_history() -> pd.DataFrame:
    response = (
        get_supabase_client()
        .schema("tennis")
        .table("tennis_picks")
        .select("*")
        .order("fecha_registro", desc=True)
        .execute()
    )
    return pd.DataFrame(response.data or [])


def _load_tennis_pick_odds_history() -> pd.DataFrame:
    response = (
        get_supabase_client()
        .schema("tennis")
        .table("pick_odds_history")
        .select("*")
        .order("fecha_registro", desc=False)
        .execute()
    )
    return pd.DataFrame(response.data or [])


def _split_tennis_match_name(match_name: object) -> tuple[str, str]:
    text = "" if pd.isna(match_name) else str(match_name)
    if " vs " not in text:
        return text.strip(), ""
    player_1, player_2 = text.split(" vs ", 1)
    return player_1.strip(), player_2.strip()


def _parse_tennis_value_market(mercado: object) -> tuple[str | None, str | None]:
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


def _first_non_empty_text(*values: object) -> str:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def settle_tennis_bet(bet_id: str, status: str, result_selection: str | None = None) -> None:
    bets = _load_tennis_bets_log()
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


def calculate_tennis_kelly_values(probability: float, decimal_odd: float, bankroll: float, multiplier: float) -> tuple[float, float, float]:
    ev = probability * decimal_odd - 1.0
    full_kelly_fraction = ev / (decimal_odd - 1.0) if decimal_odd > 1.0 else 0.0
    applied_fraction = max(0.0, full_kelly_fraction * multiplier)
    stake = bankroll * applied_fraction
    return ev, applied_fraction, stake


def insert_tennis_bet_log(
    row: pd.Series,
    *,
    bet_type: str,
    odd: float,
    stake: float,
    kelly_fraction: float,
    notes: str,
    source: str = "streamlit_value_bet",
) -> None:
    player_1, player_2 = _split_tennis_match_name(row.get("partido"))
    market_key, selection = _parse_tennis_value_market(row.get("mercado"))
    probability = float(row.get("nuestra_prob"))
    ev = probability * odd - 1.0
    bookmaker_key = _first_non_empty_text(
        row.get("mejor_bookmaker_key_actual"),
        row.get("mejor_casa_actual"),
        row.get("casa_apuestas"),
    )
    revisar_manualmente = bool(row.get("revisar_manualmente", False))
    event_time = row.get("commence_time") or row.get("event_time")
    if pd.isna(event_time):
        event_time = None
    else:
        event_time = str(event_time)

    upcoming_odds_id = row.get("upcoming_odds_id")
    if pd.isna(upcoming_odds_id):
        upcoming_odds_id = None
    else:
        upcoming_odds_id = str(upcoming_odds_id)

    clv_source_event_id = row.get("clv_source_event_id")
    if pd.isna(clv_source_event_id):
        clv_source_event_id = None
    else:
        clv_source_event_id = str(clv_source_event_id)

    payload = {
        "tennis_pick_id": None if pd.isna(row.get("tennis_pick_id")) else str(row.get("tennis_pick_id")),
        "upcoming_odds_id": upcoming_odds_id,
        "clv_source_event_id": clv_source_event_id,
        "source": source,
        "bet_type": bet_type,
        "event_time": event_time,
        "player_1": player_1,
        "player_2": player_2,
        "market_key": market_key or str(row.get("mercado") or ""),
        "selection": selection or str(row.get("mercado") or ""),
        "line": None if pd.isna(row.get("linea")) else float(row.get("linea")),
        "bookmaker_key": bookmaker_key,
        "my_odd": float(odd),
        "model_probability": probability,
        "model_fair_odd": 1.0 / probability if probability > 0 else None,
        "edge": probability - (1.0 / odd),
        "ev": ev,
        "stake": float(stake),
        "kelly_fraction": float(kelly_fraction),
        "status": "pending",
        "notes": notes.strip() or None,
        "revisar_manualmente": revisar_manualmente,
        "raw_payload": {
            "source": source,
            "partido": row.get("partido"),
            "event_time": event_time,
            "mercado": row.get("mercado"),
            "superficie": row.get("superficie"),
            "cuota_casa_csv": None if pd.isna(row.get("cuota_casa")) else float(row.get("cuota_casa")),
            "casa_apuestas_csv": row.get("casa_apuestas"),
            "mejor_cuota_actual": None if pd.isna(row.get("mejor_cuota_actual")) else float(row.get("mejor_cuota_actual")),
            "mejor_casa_actual": row.get("mejor_casa_actual"),
            "p_serve_A": None if pd.isna(row.get("p_serve_A")) else float(row.get("p_serve_A")),
            "p_serve_B": None if pd.isna(row.get("p_serve_B")) else float(row.get("p_serve_B")),
            "mean_total_games": None if pd.isna(row.get("mean_total_games")) else float(row.get("mean_total_games")),
            "mean_game_diff": None if pd.isna(row.get("mean_game_diff")) else float(row.get("mean_game_diff")),
            "win_probability_A": None if pd.isna(row.get("win_probability_A")) else float(row.get("win_probability_A")),
            "win_probability_B": None if pd.isna(row.get("win_probability_B")) else float(row.get("win_probability_B")),
            "tennis_pick_id": None if pd.isna(row.get("tennis_pick_id")) else str(row.get("tennis_pick_id")),
            "tennis_pick_fecha": None if pd.isna(row.get("tennis_pick_fecha")) else str(row.get("tennis_pick_fecha")),
            "ev_csv": None if pd.isna(row.get("EV")) else float(row.get("EV")),
            "ev_actual": None if pd.isna(row.get("EV Actual")) else float(row.get("EV Actual")),
            "revisar_manualmente": revisar_manualmente,
        },
    }
    try:
        (
            get_supabase_client()
            .schema("tennis")
            .table("bets_log")
            .insert(payload)
            .execute()
        )
    except Exception as exc:
        if "revisar_manualmente" in str(exc):
            print("[Supabase] Advertencia: la columna 'revisar_manualmente' no existe en tennis.bets_log. Reintentando...")
            payload_fallback = payload.copy()
            payload_fallback.pop("revisar_manualmente", None)
            (
                get_supabase_client()
                .schema("tennis")
                .table("bets_log")
                .insert(payload_fallback)
                .execute()
            )
        else:
            raise


def build_tennis_odds_tracking_frame() -> pd.DataFrame:
    picks = _load_tennis_picks_history()
    history = _load_tennis_pick_odds_history()
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
