from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ODDS_LISTENER = PROJECT_ROOT / "tennis" / "src" / "odds_listener.py"
EV_CALCULATOR = PROJECT_ROOT / "tennis" / "src" / "ev_calculator.py"
SLEEP_SECONDS = 3600


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_script(script_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
        raise RuntimeError(f"{script_path.name} failed with exit code {result.returncode}")


def main() -> None:
    while True:
        print(f"[{timestamp()}] Inicio de ciclo.")
        print(f"[{timestamp()}] Recolectando nuevas cuotas...")
        run_script(ODDS_LISTENER)

        print(f"[{timestamp()}] Calculando EV y actualizando predicciones...")
        run_script(EV_CALCULATOR)

        print(f"[{timestamp()}] Fin de ciclo.")
        print(f"[{timestamp()}] Esperando 60 minutos para la siguiente ejecucion...")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
