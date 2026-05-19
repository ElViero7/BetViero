# BetViero

Monorepo Python para analitica de apuestas deportivas, generacion de picks y tracking operativo.

## Modulos

- `lol/`: pipeline de League of Legends con datos de Oracle's Elixir/PandaScore, modelo multimercado, dashboard Streamlit, registro de picks y tracking de cuotas.
- `tennis/`: pipeline de tenis con historico Sackmann, features por superficie, simulacion Monte Carlo, cazador manual, cazador en vivo con Pinnacle y tracking CLV.
- `football/`: pipeline separado de futbol con API-Football/The Odds API y entrenamiento/inferencia XGBoost.
- `ui/`: dashboards Streamlit compartidos y paginas modulares.
- `streamlit_app.py`: entrada principal de la app.

## Tenis

Flujo recomendado:

1. `Descargar Historico ATP`: descarga CSVs recientes de Jeff Sackmann.
2. `Procesar Cansancio y Features`: recalcula features sin leakage, ahora agrupando saque/resto por jugador y superficie.
3. `Cazar Cuotas`: consulta The Odds API usando Pinnacle por defecto (`TENNIS_ODDS_BOOKMAKERS=pinnacle`) y calcula EV con Monte Carlo.
4. `Cazador Manual`: permite seleccionar un partido y pegar cuotas del partido en formato texto:

```text
Over 22.5 | 1.91
Under 22.5 | 1.91
Sinner -2.5 | 1.87
```

5. `Tracker CLV`: registra cuota de cierre. Para picks manuales permite introducir CLV manual; para picks con cobertura API puede capturar Pinnacle automaticamente.

Migraciones relevantes:

- `tennis/supabase/schema.sql`
- `tennis/supabase/add_bets_log_clv_fields.sql`

## LoL

El dashboard de LoL esta modularizado en `ui/lol_pages/`. El flujo incluye:

- resumen de bankroll, ROI, EV y resultados;
- tarjetas de picks;
- historial y predicciones;
- panel de control para reentrenar modelo, descargar datos y generar snapshots;
- registro manual de picks;
- liquidacion de resultados.

La insercion de picks guarda tambien un snapshot inicial en `picks_odds_history`.

Migracion relevante:

- `lol/supabase/create_picks_odds_history.sql`

## Variables Principales

```env
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
THE_ODDS_API_KEY=
TENNIS_ODDS_BOOKMAKERS=pinnacle
TENNIS_ODDS_SPORT_KEY=tennis_atp
TENNIS_ODDS_MARKETS=h2h,totals,spreads
TENNIS_ODDS_FORMAT=decimal
```

## Comandos Utiles

```powershell
streamlit run streamlit_app.py
python tennis\src\data_fetcher.py
python tennis\src\data_prep.py
python tennis\src\ev_calculator.py
```

## Notas Operativas

- El cazador en vivo de tenis depende de que The Odds API/Pinnacle cubra el partido.
- Para previas, Challengers o partidos no cubiertos por la API, usar `Cazador Manual`.
- El CLV es la metrica principal para validar si los picks baten mercado, especialmente antes de usar dinero real.
- Los CSV de predicciones son outputs operativos; no conviene mezclar cambios generados con cambios de codigo salvo que sean necesarios para deploy.
