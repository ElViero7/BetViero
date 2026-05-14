from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "sackmann"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "sackmann_atp_clean.csv"
SACKMANN_RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
)
SURFACE_MAP = {
    "Clay": "tierra",
    "Grass": "hierba",
    "Hard": "dura",
}
INCOMPLETE_SCORE_TOKENS = {"RET", "W/O", "WO", "DEF", "ABD", "ABN", "Walkover"}


class TennisDataFetcherError(RuntimeError):
    """Raised when Sackmann data cannot be fetched or cleaned."""


@dataclass(frozen=True)
class DownloadResult:
    year: int
    path: Path
    rows: int


def years_to_fetch(reference_year: int | None = None, lookback_years: int = 5) -> list[int]:
    if lookback_years <= 0:
        raise TennisDataFetcherError("lookback_years must be greater than 0.")
    end_year = reference_year or date.today().year
    start_year = end_year - lookback_years + 1
    return list(range(start_year, end_year + 1))


def download_sackmann_csv(
    year: int,
    output_dir: Path = RAW_DIR,
    *,
    timeout_seconds: int = 30,
    strict: bool = False,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"atp_matches_{year}.csv"
    url = SACKMANN_RAW_URL_TEMPLATE.format(year=year)

    response = requests.get(url, timeout=timeout_seconds)
    if response.status_code == 404 and not strict:
        print(f"Skipping {year}: CSV not found at {url}")
        return None
    if response.status_code >= 400:
        raise TennisDataFetcherError(
            f"Failed to download Sackmann ATP matches for {year} "
            f"({response.status_code}): {response.text[:300]}"
        )

    output_path.write_bytes(response.content)
    return output_path


def download_recent_sackmann_csvs(
    *,
    reference_year: int | None = None,
    lookback_years: int = 5,
    output_dir: Path = RAW_DIR,
    strict: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for year in years_to_fetch(reference_year, lookback_years):
        downloaded_path = download_sackmann_csv(year, output_dir, strict=strict)
        if downloaded_path is not None:
            paths.append(downloaded_path)
    if not paths:
        raise TennisDataFetcherError("No Sackmann ATP CSV files were downloaded.")
    return paths


def parse_total_games(score: object) -> int | None:
    if pd.isna(score):
        return None

    score_text = str(score).strip()
    if not score_text:
        return None
    if any(token in score_text for token in INCOMPLETE_SCORE_TOKENS):
        return None

    total_games = 0
    for set_score in score_text.split():
        if "[" in set_score or "]" in set_score:
            continue
        set_score = set_score.split("(", 1)[0]
        if "-" not in set_score:
            continue

        left, right = set_score.split("-", 1)
        if not left.isdigit() or not right.isdigit():
            continue
        total_games += int(left) + int(right)

    return total_games or None


def service_games_won_pct(service_games: object, break_points_saved: object, break_points_faced: object) -> float | None:
    service_games_value = pd.to_numeric(service_games, errors="coerce")
    break_points_saved_value = pd.to_numeric(break_points_saved, errors="coerce")
    break_points_faced_value = pd.to_numeric(break_points_faced, errors="coerce")

    if pd.isna(service_games_value) or service_games_value <= 0:
        return None
    if pd.isna(break_points_saved_value) or pd.isna(break_points_faced_value):
        return None

    breaks_conceded = max(float(break_points_faced_value) - float(break_points_saved_value), 0.0)
    service_games_won = max(float(service_games_value) - breaks_conceded, 0.0)
    return round(service_games_won / float(service_games_value), 6)


def clean_sackmann_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "tourney_date",
        "surface",
        "winner_name",
        "loser_name",
        "score",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise TennisDataFetcherError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned = frame.copy()
    cleaned["superficie"] = cleaned["surface"].map(SURFACE_MAP)
    cleaned["total_juegos"] = cleaned["score"].map(parse_total_games)
    cleaned["porcentaje_juegos_saque_jugador_a"] = cleaned.apply(
        lambda row: service_games_won_pct(row["w_SvGms"], row["w_bpSaved"], row["w_bpFaced"]),
        axis=1,
    )
    cleaned["porcentaje_juegos_saque_jugador_b"] = cleaned.apply(
        lambda row: service_games_won_pct(row["l_SvGms"], row["l_bpSaved"], row["l_bpFaced"]),
        axis=1,
    )
    cleaned["fecha"] = pd.to_datetime(
        cleaned["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    ).dt.date

    output = pd.DataFrame(
        {
            "fecha": cleaned["fecha"],
            "jugador_a": cleaned["winner_name"],
            "jugador_b": cleaned["loser_name"],
            "superficie": cleaned["superficie"],
            "total_juegos": cleaned["total_juegos"],
            "porcentaje_juegos_saque_jugador_a": cleaned["porcentaje_juegos_saque_jugador_a"],
            "porcentaje_juegos_saque_jugador_b": cleaned["porcentaje_juegos_saque_jugador_b"],
        }
    )
    output = output.dropna(
        subset=[
            "fecha",
            "jugador_a",
            "jugador_b",
            "superficie",
            "total_juegos",
            "porcentaje_juegos_saque_jugador_a",
            "porcentaje_juegos_saque_jugador_b",
        ]
    )
    output["total_juegos"] = output["total_juegos"].astype(int)
    return output.sort_values(["fecha", "jugador_a", "jugador_b"]).reset_index(drop=True)


def build_clean_dataset(raw_paths: list[Path], output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in raw_paths]
    combined = pd.concat(frames, ignore_index=True)
    cleaned = clean_sackmann_frame(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def run(
    *,
    reference_year: int | None = None,
    lookback_years: int = 5,
    raw_dir: Path = RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    strict: bool = False,
) -> pd.DataFrame:
    raw_paths = download_recent_sackmann_csvs(
        reference_year=reference_year,
        lookback_years=lookback_years,
        output_dir=raw_dir,
        strict=strict,
    )
    return build_clean_dataset(raw_paths, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and clean recent Jeff Sackmann ATP match CSVs for tennis training."
    )
    parser.add_argument("--reference-year", type=int, default=None)
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested yearly CSV is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned = run(
        reference_year=args.reference_year,
        lookback_years=args.lookback_years,
        raw_dir=args.raw_dir,
        output_path=args.output,
        strict=args.strict,
    )
    print(f"Saved {len(cleaned)} clean rows to {args.output}")


if __name__ == "__main__":
    main()
