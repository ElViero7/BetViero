import os
import requests
from dotenv import load_dotenv

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
if not ODDS_API_KEY:
    raise RuntimeError("Missing ODDS_API_KEY or THE_ODDS_API_KEY in .env")

url = "https://api.the-odds-api.com/v4/sports"
params = {
    'apiKey': ODDS_API_KEY
}

print("Consultando el catálogo de deportes a The Odds API...")
respuesta = requests.get(url, params=params)

if respuesta.status_code == 200:
    deportes = respuesta.json()
    print(f"Total de deportes disponibles en tu plan: {len(deportes)}")
    
    # Filtramos para buscar cualquier cosa que suene a LoL o Esports
    esports = [d for d in deportes if 'esport' in d.get('group', '').lower() 
               or 'league of legends' in d.get('title', '').lower() 
               or 'lol' in d.get('key', '').lower()]
    
    if esports:
        print("\n🎮 ¡Bingo! E-sports encontrados:")
        for e in esports:
            print(f"- Título: {e['title']}")
            print(f"  -> LA KEY QUE DEBES USAR: '{e['key']}'")
            print(f"  -> ¿Está activo ahora mismo?: {e['active']}")
            print("-" * 30)
    else:
        print("\n❌ No se ha encontrado League of Legends en tu catálogo actual.")
else:
    print(f"Error {respuesta.status_code}:", respuesta.text)
