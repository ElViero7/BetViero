from __future__ import annotations

import os
import re

import cv2
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import pytesseract
from thefuzz import fuzz

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_CMD_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

OCR_TEAM_CORRECTIONS = {
    "hanuies": "Hanwha Life Esports",
    "heanwie": "Hanwha Life Esports",
    "hanwha life eaperts": "Hanwha Life Esports",
    "mangehim": "Nongshim RedForce",
    "radferee": "Nongshim RedForce",
    "otravings": "OKSavingsBank BRION",
    "otravingsfiank": "OKSavingsBank BRION",
    "okieing": "OKSavingsBank BRION",
    "okieingefenh": "OKSavingsBank BRION",
    "minjas": "Ninjas",
    "pyjaman": "Ninjas in Pyjamas",
    "genb": "Gen.G",
    "ganb": "Gen.G",
    "ged": "God",
    "l800": "LNG",
    "dn soopers": "Kwangdong Freecs",
}

GENERIC_TEAM_TOKENS = {"esports", "gaming", "team", "club", "academy"}


def clean_ocr_team_text(text: str) -> str:
    cleaned = re.sub(r"[\[\]\|=_]", " ", str(text))
    cleaned = re.sub(r"[^\w\sÀ-ÿ.\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def apply_ocr_team_corrections(text: str) -> str:
    cleaned = clean_ocr_team_text(text)
    for wrong_text, corrected_text in OCR_TEAM_CORRECTIONS.items():
        cleaned = re.sub(
            rf"\b{re.escape(wrong_text)}\b",
            corrected_text,
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned


def _normalize_for_fuzzy(text: str) -> str:
    cleaned = apply_ocr_team_corrections(text).lower()
    cleaned = re.sub(r"\b\d+(?:[.,]\d+)?\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _normalize_for_keyword_scan(text: str) -> str:
    corrected = apply_ocr_team_corrections(text).lower()
    corrected = re.sub(r"\b\d+(?:[.,]\d+)?\b", " ", corrected)
    corrected = re.sub(r"[^a-zà-ÿ.\s-]", " ", corrected)
    corrected = re.sub(r"\s+", " ", corrected)
    return corrected.strip()


def _preprocess_for_ocr(image_bytes: bytes) -> Image.Image:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No pude decodificar la imagen para OCR.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    return Image.fromarray(binary)


def _looks_like_team_line(line: str) -> bool:
    compact = apply_ocr_team_corrections(line)
    if not compact:
        return False
    if re.fullmatch(r"[\d\s.,:;+\-/]+", compact):
        return False
    if len(compact) < 2:
        return False
    letters = re.findall(r"[A-Za-zÀ-ÿ]", compact)
    if len(letters) < 2:
        return False
    return True


def extract_matchups_from_image(image_bytes: bytes) -> list[str]:
    processed_image = _preprocess_for_ocr(image_bytes)
    raw_text = pytesseract.image_to_string(processed_image, config="--psm 6")
    lines = [apply_ocr_team_corrections(line) for line in raw_text.splitlines()]
    return [line for line in lines if _looks_like_team_line(line)]


def fuzzy_match_team(
    raw_name: str,
    valid_teams_list: list[str],
    confidence_threshold: int = 60,
) -> tuple[str | None, int]:
    cleaned_raw_name = _normalize_for_fuzzy(raw_name)
    if not valid_teams_list:
        return None, 0
    if not cleaned_raw_name:
        return str(valid_teams_list[0]), 0

    normalized_team_map = {
        _normalize_for_fuzzy(team_name): str(team_name)
        for team_name in valid_teams_list
        if _normalize_for_fuzzy(team_name)
    }
    if not normalized_team_map:
        return None, 0

    try:
        scored_matches = [
            (team_name, fuzz.partial_ratio(normalized_team_name, cleaned_raw_name))
            for normalized_team_name, team_name in normalized_team_map.items()
        ]
    except Exception:
        return None, 0

    if not scored_matches:
        return str(valid_teams_list[0]), 0

    team_name, score = max(scored_matches, key=lambda item: int(item[1]))
    score = int(score)
    if score < 30:
        return "No identificado", score
    return str(team_name), score


def _score_team_in_line(team_name: str, normalized_line: str) -> int:
    normalized_team = _normalize_for_keyword_scan(team_name)
    if not normalized_team or not normalized_line:
        return 0

    scores = [fuzz.partial_ratio(normalized_team, normalized_line)]
    for token in normalized_team.replace(".", " ").split():
        if len(token) >= 3 and token not in GENERIC_TEAM_TOKENS:
            scores.append(fuzz.partial_ratio(token, normalized_line))
    return int(max(scores))


def detect_matchups_from_ocr_lines(
    raw_lines: list[str],
    valid_teams_list: list[str],
    minimum_score: int = 70,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for raw_line in raw_lines:
        cleaned_line = apply_ocr_team_corrections(raw_line)
        normalized_line = _normalize_for_keyword_scan(cleaned_line)
        if not normalized_line:
            continue

        hits = []
        for team_name in valid_teams_list:
            score = _score_team_in_line(team_name, normalized_line)
            if score > minimum_score:
                hits.append({"team": str(team_name), "score": int(score)})

        hits = sorted(hits, key=lambda hit: int(hit["score"]), reverse=True)
        unique_hits: list[dict[str, object]] = []
        for hit in hits:
            if hit["team"] not in {existing["team"] for existing in unique_hits}:
                unique_hits.append(hit)

        if not unique_hits:
            continue

        team_a = unique_hits[0]
        team_b = unique_hits[1] if len(unique_hits) > 1 else {"team": "", "score": 0}
        dedupe_pair = tuple(sorted([str(team_a["team"]), str(team_b["team"])]))
        seen_key = (str(dedupe_pair[0]), str(dedupe_pair[1]), cleaned_line)
        if seen_key in seen_keys:
            continue
        seen_keys.add(seen_key)

        results.append(
            {
                "raw_line": cleaned_line,
                "team_a": team_a["team"],
                "score_a": team_a["score"],
                "team_b": team_b["team"],
                "score_b": team_b["score"],
            }
        )

    return results
