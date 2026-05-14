import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
if not ODDS_API_KEY:
    raise RuntimeError("Missing ODDS_API_KEY or THE_ODDS_API_KEY in .env")

# Parámetros estrictos para no gastar cuota innecesaria
SPORT_KEY = 'esports_league_of_legends'
REGIONS = 'eu' # Europa
MARKETS = 'h2h' # h2h es el ganador del partido. Luego buscaremos el First Blood.
BOOKMAKERS = 'bet365'

url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"

params = {
    'apiKey': ODDS_API_KEY,
    'regions': REGIONS,
    'markets': MARKETS,
    'bookmakers': BOOKMAKERS,
    'oddsFormat': 'decimal'
}

print("Consultando The Odds API...")
respuesta = requests.get(url, params=params)

if respuesta.status_code == 200:
    datos = respuesta.json()
    # Guardamos el JSON en un archivo para que lo puedas inspeccionar visualmente
    with open('cuotas_lol_test.json', 'w') as f:
        json.dump(datos, f, indent=4)
    print(f"¡Éxito! Se han encontrado {len(datos)} partidos activos.")
    print("Abre el archivo 'cuotas_lol_test.json' para ver la estructura.")
else:
    print(f"Error {respuesta.status_code}:", respuesta.text)
