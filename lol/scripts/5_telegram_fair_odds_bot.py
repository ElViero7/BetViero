from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import telebot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base import LoLPipelineError
from src.fair_odds_bot import (
    build_alert_message,
    evaluate_kills_batch_from_reply,
    build_pick_payload_from_reply,
    FairOddsEngine,
    insert_pick_to_supabase,
    load_config,
)

config = None
engine = None
bot = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Telegram Fair Odds bot in polling mode."
    )
    return parser.parse_args()


def handle_start(message: telebot.types.Message) -> None:
    assert bot is not None
    bot.reply_to(
        message,
        "Comandos disponibles:\n"
        "/scan - Escanea proximos partidos mayores de LoL y envia alertas multi-mercado.\n\n"
        "Responde a una alerta con:\n"
        "1 2.10 winner\n"
        "2 1.85 firstblood\n"
        "Over 24.5 1.90 kills\n\n"
        "O consulta varias lineas de kills asi:\n"
        "kills\n"
        "28.5 1.68 2.00\n"
        "29.5 1.84 1.86\n"
        "30.5 2.00 1.70",
    )


def handle_scan(message: telebot.types.Message) -> None:
    assert bot is not None
    assert engine is not None
    assert config is not None
    try:
        bot.reply_to(message, "Escaneando proximos partidos de LoL...")
        scored_matches = engine.score_upcoming_matches()
        if scored_matches.empty:
            bot.send_message(message.chat.id, "No he encontrado partidos elegibles para escanear.")
            return

        sent = 0
        for league_name, league_frame in scored_matches.groupby("league", sort=False):
            bot.send_message(message.chat.id, f"{league_name}\n")
            for _, row in league_frame.iterrows():
                bot.send_message(message.chat.id, build_alert_message(row))
                sent += 1

        bot.send_message(
            message.chat.id,
            f"Scan completado. Alertas enviadas: {sent}. "
            f"Predicciones guardadas en {config.predictions_output_path}.",
        )
    except Exception as exc:
        traceback.print_exc()
        bot.send_message(message.chat.id, f"Error en /scan: {exc}")


def handle_reply_pick_registration(message: telebot.types.Message) -> None:
    assert bot is not None
    assert config is not None
    if message.text is None:
        return
    if message.text.startswith("/"):
        return
    if message.reply_to_message is None or not message.reply_to_message.text:
        return

    try:
        kills_batch = evaluate_kills_batch_from_reply(
            config=config,
            reply_text=message.text,
            alert_text=message.reply_to_message.text,
        )
    except LoLPipelineError as exc:
        bot.reply_to(message, str(exc))
        return

    if kills_batch is not None:
        positive = [item for item in kills_batch if item["recommended_stake"] > 0]
        if not positive:
            bot.reply_to(message, "Sin valor en ninguna linea de kills")
            return
        lines = ["Valor en kills:"]
        for item in positive[:5]:
            lines.append(
                f"{item['selection']} {item['line']:.1f} @{item['my_odd']:.2f} | "
                f"stake {item['recommended_stake']:.2f} | EV {item['ev']:.2%}"
            )
        lines.append("Si quieres registrar una, responde solo a la alerta original con una linea concreta.")
        bot.reply_to(message, "\n".join(lines))
        return

    try:
        pick_payload = build_pick_payload_from_reply(
            config=config,
            reply_text=message.text,
            alert_text=message.reply_to_message.text,
        )
    except LoLPipelineError:
        return

    if float(pick_payload["recommended_stake"]) <= 0:
        bot.reply_to(message, "Sin valor")
        return

    try:
        insert_pick_to_supabase(config, pick_payload)
        bot.reply_to(
            message,
            "Apuesta registrada\n"
            f"Stake recomendado: {pick_payload['recommended_stake']:.2f}\n"
            f"Edge: {pick_payload['edge']:.2%}\n"
            f"EV: {pick_payload['ev']:.2%}",
        )
    except Exception as exc:
        traceback.print_exc()
        bot.reply_to(message, f"Error al registrar la apuesta: {exc}")


def main() -> None:
    parse_args()
    global config, engine, bot
    config = load_config()
    engine = FairOddsEngine(config)
    bot = telebot.TeleBot(config.telegram_bot_token, parse_mode=None)
    bot.register_message_handler(handle_start, commands=["start", "help"])
    bot.register_message_handler(handle_scan, commands=["scan"])
    bot.register_message_handler(
        handle_reply_pick_registration,
        func=lambda message: True,
        content_types=["text"],
    )
    try:
        bot.infinity_polling(timeout=30, long_polling_timeout=30)
    except LoLPipelineError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
