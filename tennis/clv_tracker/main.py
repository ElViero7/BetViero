import os
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
import httpx
from supabase import create_client, Client

app = FastAPI(title="CLV Tracker - Tennis Models")

# Variables de entorno necesarias en Cloud Run
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_LOCATION = os.environ.get("GCP_LOCATION")
QUEUE_NAME = os.environ.get("QUEUE_NAME")
SERVICE_URL = os.environ.get("SERVICE_URL")  # La URL pública de este Cloud Run
API_SECRET = os.environ.get("API_SECRET")  # Para proteger el endpoint /enqueue

# Función para inicializar Supabase de forma perezosa (lazy)
def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise HTTPException(
            status_code=500, 
            detail="Error de configuración: SUPABASE_URL o SUPABASE_KEY no están definidos en el entorno"
        )
    return create_client(url, key)

# --- Auxiliares de Emparejamiento de Nombres (Basado en odds_listener.py) ---

def normalize_player_name(name: object) -> str:
    text = "" if name is None else str(name)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace(".", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text)

def first_name_conflict(left_text: str, right_text: str) -> bool:
    left_tokens = left_text.split()
    right_tokens = right_text.split()
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    f1 = left_tokens[0]
    f2 = right_tokens[0]
    if f1[0] != f2[0]:
        return True
    if len(f1) > 1 and len(f2) > 1:
        if f1 != f2 and not f1.startswith(f2) and not f2.startswith(f1):
            return True
    return False

def player_name_score(left: object, right: object) -> float:
    left_text = normalize_player_name(left)
    right_text = normalize_player_name(right)
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    if first_name_conflict(left_text, right_text):
        return 0.0

    left_tokens = left_text.split()
    right_tokens = right_text.split()
    token_overlap = len(set(left_tokens) & set(right_tokens)) / max(len(set(left_tokens) | set(right_tokens)), 1)
    last_name_score = 0.86 if left_tokens[-1] == right_tokens[-1] else 0.0
    initial_score = 0.0
    if len(left_tokens) == 2 and len(right_tokens) >= 2:
        if left_tokens[0][:1] == right_tokens[0][:1] and left_tokens[-1] == right_tokens[-1]:
            initial_score = 0.95
        if left_tokens[-1][:1] == right_tokens[0][:1] and left_tokens[0] == right_tokens[-1]:
            initial_score = 0.95
    return max(token_overlap, last_name_score, initial_score)

def extract_pinnacle_odd_matching_bet(odds_data, market_key, selection, line):
    try:
        bookmakers = odds_data.get("bookmakers", [])
        pinnacle = None
        for bm in bookmakers:
            if bm.get("key") == "pinnacle":
                pinnacle = bm
                break
        if not pinnacle:
            return None

        markets = pinnacle.get("markets", [])
        target_market = None
        for m in markets:
            if m.get("key") == market_key:
                target_market = m
                break
        if not target_market:
            return None

        outcomes = target_market.get("outcomes", [])
        
        for outcome in outcomes:
            price = float(outcome.get("price"))
            outcome_name = outcome.get("name")
            outcome_point = outcome.get("point")

            if market_key == "totals":
                if outcome_name.lower() == selection.lower():
                    if line is not None and outcome_point is not None:
                        if abs(float(outcome_point) - float(line)) < 1e-9:
                            return price
            elif market_key == "spreads":
                if line is not None and outcome_point is not None:
                    if abs(float(outcome_point) - float(line)) < 1e-9:
                        if player_name_score(outcome_name, selection) >= 0.72:
                            return price
            elif market_key == "h2h":
                if player_name_score(outcome_name, selection) >= 0.72:
                    return price
        return None
    except Exception:
        return None

# --- Función para obtener cuotas gestionando la caché en Supabase ---
async def fetch_odds_with_cache(match_id: str, sport_key: str = "tennis_atp") -> dict:
    supabase_client = get_supabase_client()
    
    # 1. Comprobar si hay caché válido (menos de 2 minutos de antigüedad)
    try:
        response = supabase_client.schema("tennis").table("odds_api_cache").select("*").eq("id", sport_key).execute()
        if response.data:
            cache = response.data[0]
            fetched_at = datetime.fromisoformat(cache["fetched_at"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - fetched_at < timedelta(minutes=2):
                all_events = cache["response_data"]
                for event in all_events:
                    if event.get("id") == match_id:
                        return event
                raise HTTPException(status_code=404, detail=f"Partido no encontrado en el caché de cuotas para {sport_key}")
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[Caché] Error al leer caché en Supabase para {sport_key}: {exc}. Pasando a llamada directa...")

    # 2. Si no hay caché válido, llamar a The Odds API para todos los partidos activos (1 crédito)
    odds_api_key = os.environ.get("ODDS_API_KEY")
    if not odds_api_key:
         raise HTTPException(status_code=500, detail="ODDS_API_KEY no configurado en el entorno")

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": odds_api_key,
        "regions": "eu",
        "bookmakers": "pinnacle",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "decimal"
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Error en The Odds API para {sport_key}: {resp.text}")
        
    all_events = resp.json()
    
    # 3. Guardar el nuevo listado y el timestamp actual en el caché de Supabase
    try:
        payload = {
            "id": sport_key,
            "response_data": all_events,
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }
        supabase_client.schema("tennis").table("odds_api_cache").upsert(payload).execute()
    except Exception as exc:
        print(f"[Caché] No se pudo guardar el nuevo caché en Supabase para {sport_key}: {exc}")
        
    # Buscar el partido en el nuevo listado
    for event in all_events:
        if event.get("id") == match_id:
            return event
            
    raise HTTPException(status_code=404, detail=f"Partido no encontrado en el nuevo listado de cuotas para {sport_key}")

# --- Modelos de Petición ---

class WebhookPayload(BaseModel):
    type: str
    table: str
    record: dict

class ExecutePayload(BaseModel):
    match_id: str
    bet_id: str

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "alive", "service": "clv-tracker-tennis", "cache": "enabled"}

@app.post("/enqueue")
async def enqueue_task(payload: WebhookPayload, x_api_secret: str = Header(None)):
    """ Recibe el Webhook de Supabase y programa la tarea en Cloud Tasks """
    if API_SECRET and x_api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="No autorizado")

    bet_data = payload.record
    match_id = bet_data.get("match_id") or bet_data.get("clv_source_event_id")
    
    if not match_id and "raw_payload" in bet_data:
        raw_payload = bet_data["raw_payload"]
        if isinstance(raw_payload, str):
            try:
                raw_payload = json.loads(raw_payload)
            except Exception:
                raw_payload = {}
        if isinstance(raw_payload, dict):
            match_id = raw_payload.get("event_id")

    commence_time_str = bet_data.get("event_time") or bet_data.get("commence_time")

    if not match_id or not commence_time_str:
        raise HTTPException(status_code=400, detail="Faltan datos de la apuesta (match_id o commence_time)")

    # Calcular commence_time - 90 segundos
    commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
    execution_time = commence_time - timedelta(seconds=90)
    
    if execution_time < datetime.now(timezone.utc):
        execution_time = datetime.now(timezone.utc) + timedelta(seconds=5)

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(GCP_PROJECT, GCP_LOCATION, QUEUE_NAME)

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{SERVICE_URL}/execute",
            "headers": {"Content-type": "application/json"},
            "body": json.dumps({"match_id": match_id, "bet_id": bet_data.get("id")}).encode(),
        }
    }

    timestamp = timestamp_pb2.Timestamp()
    timestamp.FromDatetime(execution_time)
    task["schedule_time"] = timestamp

    response = client.create_task(request={"parent": parent, "task": task})
    
    return {
        "status": "success",
        "task_name": response.name,
        "scheduled_for": execution_time.isoformat()
    }

@app.post("/execute")
async def execute_task(payload: ExecutePayload):
    """ Despertado por Cloud Tasks justo antes del partido para capturar cuotas """
    match_id = payload.match_id
    bet_id = payload.bet_id

    # 1. Obtener la apuesta desde Supabase (schema 'tennis', tabla 'bets_log')
    supabase_client = get_supabase_client()
    try:
        response = (
            supabase_client.schema("tennis")
            .table("bets_log")
            .select("player_1, player_2, market_key, selection, line, my_odd, model_probability, upcoming_odds_id")
            .eq("id", bet_id)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Apuesta {bet_id} no encontrada en Supabase")
        bet = response.data[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error consultando Supabase: {str(exc)}")

    # Obtener el sport_key desde upcoming_odds si existe
    sport_key = "tennis_atp"
    upcoming_odds_id = bet.get("upcoming_odds_id")
    if upcoming_odds_id:
        try:
            odds_resp = (
                supabase_client.schema("tennis")
                .table("upcoming_odds")
                .select("sport_key")
                .eq("id", upcoming_odds_id)
                .execute()
            )
            if odds_resp.data:
                found_sport = odds_resp.data[0].get("sport_key")
                if found_sport:
                    sport_key = found_sport
        except Exception as exc:
            print(f"[Execute] Error al obtener sport_key para upcoming_odds_id {upcoming_odds_id}: {exc}. Usando 'tennis_atp' por defecto.")

    market_key = bet.get("market_key")
    selection = bet.get("selection")
    line = bet.get("line")
    my_odd = float(bet.get("my_odd")) if bet.get("my_odd") is not None else None
    model_probability = float(bet.get("model_probability")) if bet.get("model_probability") is not None else None

    if not market_key or not selection:
        raise HTTPException(status_code=400, detail="La apuesta de Supabase no contiene market_key o selection")

    # 2. Obtener cuotas del evento usando la caché inteligente
    odds_data = await fetch_odds_with_cache(match_id, sport_key)
    
    # 3. Procesar y extraer la cuota de cierre de Pinnacle
    clv_odd = extract_pinnacle_odd_matching_bet(odds_data, market_key, selection, line)
    
    if clv_odd is None:
        raise HTTPException(
            status_code=404, 
            detail=f"No se encontro cuota de Pinnacle para {market_key} | {selection} | Linea: {line}"
        )

    # 4. Calcular métricas CLV
    clv_edge = None
    if model_probability is not None:
        clv_edge = model_probability * clv_odd - 1.0

    clv_value = None
    if my_odd is not None and clv_odd > 0:
        clv_value = my_odd / clv_odd - 1.0

    # 5. Actualizar Supabase en la tabla bets_log
    update_payload = {
        "clv_pinnacle_odd": clv_odd,
        "clv_captured_at": datetime.now(timezone.utc).isoformat(),
        "clv_edge": clv_edge,
        "clv_value": clv_value,
        "clv_source": "pinnacle_api"
    }

    try:
        supabase_client.schema("tennis").table("bets_log").update(update_payload).eq("id", bet_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error actualizando Supabase: {str(exc)}")

    return {
        "status": "CLV registrado con exito",
        "match_id": match_id,
        "bet_id": bet_id,
        "clv_pinnacle_odd": clv_odd,
        "clv_value": clv_value,
        "clv_edge": clv_edge
    }
