from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LoLPipelineError(RuntimeError):
    """Raised when a LoL pipeline step cannot be executed."""


@dataclass(frozen=True)
class LoLMockConfig:
    mock_dir: Path | None = None
    region: str = "global"
    tournament_slug: str = "worlds"


def load_lol_mock_payload(mock_dir: Path, relative_path: str) -> dict[str, Any]:
    file_path = mock_dir / "lol" / relative_path
    if not file_path.exists():
        raise LoLPipelineError(f"LoL mock file not found: {file_path}")

    import json

    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LoLPipelineError(f"Invalid JSON in LoL mock file: {file_path}") from exc
