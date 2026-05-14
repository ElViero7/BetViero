# BetViero Football

Raiz independiente del proyecto de futbol para detectar value bets (EV+) en mercados secundarios.

## Estructura

- `supabase/schema.sql`: esquema DDL para Supabase/PostgreSQL.
- `src/`: paquete Python del proyecto.
- `.env.example`: variables de entorno base.
- `data/mock/`: respuestas mock para desarrollar sin consumir APIs.

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Flujo Mock

Los modos `mock` evitan consumir `API-Football` y `The Odds API`.
`3_daily_inference.py` y `4_odds_listener.py` siguen usando Supabase para guardar y leer estado.

ETL:

```bash
python scripts/1_api_football_etl.py --mock-dir data/mock --league-id 39 --season 2025 --last-n 8
```

Entrenamiento:

```bash
python scripts/2_train_xgboost.py --mock-dir data/mock --league-id 39 --season 2025 --last-n-matches 8 --min-train-rows 4
```

Inferencia diaria:

```bash
python scripts/3_daily_inference.py --mock-dir data/mock --league-id 39 --season 2025 --prediction-date 2026-05-07
```

Listener de odds:

```bash
python scripts/4_odds_listener.py --mock-dir data/mock --sport-key soccer_epl
```
