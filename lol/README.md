# BetViero LoL

Raiz independiente para experimentar el pipeline de value betting aplicado a League of Legends.

## Estado

La version actual ya soporta una arquitectura multi-mercado para `Winner`, `First Blood`, `First Tower` y `Total Kills`.

- entrenamiento: `2025 all games + 2026 game 1 only`
- prediccion operativa: `solo Map 1`
- bot Telegram: un mensaje por partido y reply uniforme para registrar picks

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Supabase

Para el flujo nuevo, pega primero [supabase/create_picks_table.sql](C:/Users/Javier/Desktop/Viero7/PROYECTOS/BetViero/lol/supabase/create_picks_table.sql:1) en el SQL Editor de Supabase.

Coloca los CSVs de Oracle's Elixir en:

- `data/raw/oracles_elixir/`

Configura variables:

- copia `.env.example` a `.env` si no lo tienes
- rellena `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PANDASCORE_API_KEY`, `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`

## Dataset canonico

Construir el dataset canonico de entrenamiento:

```bash
python scripts/6_build_multimarket_dataset.py --input-path data/raw/oracles_elixir
```

Entrenar los cuatro modelos:

```bash
python scripts/7_train_multimarket_models.py --input-path data/raw/oracles_elixir
```

Salidas por defecto:

- `data/processed/oracles_elixir_team_rows_training.parquet`
- `data/processed/multimarket_matchups.parquet`
- `models/modelo_multimarket.pkl`

## Telegram Fair Odds Bot

Arranque del bot en polling:

```bash
python scripts/5_telegram_fair_odds_bot.py
```

Flujo:

- escribe `/scan` al bot
- el bot consulta PandaScore y filtra ligas mayores
- calcula `Winner`, `First Blood`, `First Tower` y `Total Kills` para `Map 1`
- envia una alerta por partido con cuota justa de ambos equipos en los mercados binarios
- si respondes a una alerta, inserta una fila en `picks` de Supabase

Formato de reply:

- `T1 2.10 winner`
- `Gen.G 1.85 firstblood`
- `G2 Esports 2.00 firsttower`
- `Over 24.5 1.90 kills`

Notas:

- `scripts/5_telegram_fair_odds_bot.py` guarda el ultimo escaneo en `PREDICTIONS_INPUT_PATH`
- como PandaScore no suele traer los lados `Blue/Red` prepartido, el bot estima la probabilidad final promediando ambos escenarios de lado
- el stake recomendado se calcula con `Kelly * KELLY_FRACTION` usando `BANKROLL_AMOUNT`
- el stake recomendado queda capado por `MAX_STAKE_AMOUNT`
- si algun nombre de equipo no casa entre PandaScore y Oracle's Elixir, ajusta `data/config/team_name_overrides.json`

## Manual Scan

Si PandaScore no trae todos los partidos, puedes cargar un CSV manual y enviar las alertas igualmente:

```bash
python scripts/8_manual_match_scan.py --input-path data/manual/manual_week_matches.csv
```

El CSV debe tener como minimo:

- `league`
- `team_a`
- `team_b`

Y opcionalmente:

- `scheduled_at`
- `book_team_a_odd`
- `book_team_b_odd`

## Streamlit

Dashboard local para ver `bets` legacy, `picks` actuales y el ultimo snapshot de predicciones:

```bash
streamlit run streamlit_app.py
```

Muestra:

- registros pendientes
- historial unificado
- stake, edge, EV y PnL cuando existan
- ROI sobre picks ya liquidados
- snapshot del ultimo scan

## Oracle Drive Refresh

Descargar los CSV desde el folder de Google Drive de Oracle's Elixir:

```bash
python scripts/10_download_oracles_elixir_drive.py
```

Flujo completo de refresco:

```bash
python scripts/11_refresh_pipeline_from_oracle_drive.py
```

Ese flujo:

- descarga/actualiza los CSV raw
- reentrena el artefacto multi-market
- resincroniza el estado de las `bets` legacy

## Docker

Construir y arrancar en segundo plano:

```bash
docker compose up -d --build
```

Ver logs:

```bash
docker compose logs -f fair-odds-bot
```

Parar:

```bash
docker compose down
```

## VPS Deploy

Ruta simple:

1. Crear una VPS Linux pequeña
2. Instalar Docker y Docker Compose
3. Copiar este proyecto a la VPS
4. Rellenar `.env`
5. Lanzar `docker compose up -d --build`

Para un despliegue minimo en Ubuntu:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
```

Despues de volver a entrar en la sesion:

```bash
git clone <tu-repo-o-copia-del-proyecto>
cd lol
docker compose up -d --build
docker compose logs -f fair-odds-bot
```
