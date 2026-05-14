from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import telebot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fair_odds_bot import build_alert_message, FairOddsEngine, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score manually curated LoL fixtures and send grouped alerts to Telegram."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/manual/manual_week_matches.csv"),
        help="CSV file with columns league, team_a, team_b, scheduled_at and optional book odds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    engine = FairOddsEngine(config)
    manual_matches = pd.read_csv(args.input_path)
    scored_matches = engine.score_manual_matches(manual_matches)

    bot = telebot.TeleBot(config.telegram_bot_token, parse_mode=None)
    sent = 0
    for league_name, league_frame in scored_matches.groupby("league", sort=False):
        bot.send_message(config.telegram_chat_id, f"{league_name}\n")
        for _, row in league_frame.iterrows():
            bot.send_message(config.telegram_chat_id, build_alert_message(row))
            sent += 1
    bot.send_message(
        config.telegram_chat_id,
        f"Manual scan completado. Alertas enviadas: {sent}. Snapshot: {config.predictions_output_path}",
    )
    print(f"Sent {sent} alerts from {args.input_path}.")


if __name__ == "__main__":
    main()
