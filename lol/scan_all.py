import os
import requests
from dotenv import load_dotenv

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
if not ODDS_API_KEY:
    raise RuntimeError("Missing ODDS_API_KEY or THE_ODDS_API_KEY in .env")

url = "https://api.the-odds-api.com/v4/sports"
params = {
    'apiKey': ODDS_API_KEY,
    'all': 'true'  # El truco mágico para ver deportes fuera de temporada o inactivos hoy
}

print("Descargando el catálogo COMPLETO (Activos e Inactivos) de The Odds API...")
respuesta = requests.get(url, params=params)

if respuesta.status_code == 200:
    deportes = respuesta.json()
    print(f"\n✅ Se han encontrado {len(deportes)} deportes en total.\n")
    
    # Imprimimos todo, pero le decimos a Python que nos grite si ve algo de LoL
    for d in deportes:
        key = d.get('key', '')
        title = d.get('title', '')
        group = d.get('group', '')
        active = "🟢 ACTIVO" if d.get('active') else "🔴 INACTIVO"
        
        texto = f"{group} | {title} -> KEY: '{key}' [{active}]"
        
        # Filtro radar para E-sports
        if 'esport' in group.lower() or 'league' in title.lower() or 'lol' in key.lower():
            print(f"👉 🔥 POSIBLE OBJETIVO: {texto}")
        else:
            print(texto)
else:
    print(f"Error {respuesta.status_code}:", respuesta.text)
