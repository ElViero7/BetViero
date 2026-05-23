from __future__ import annotations

import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configurar rutas
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "sackmann"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "serve_regressor.pkl"

# Intentar importar funciones de data_prep
try:
    from data_prep import load_raw_matches, build_player_match_rows, add_no_leakage_player_features
except ModuleNotFoundError:
    from tennis.src.data_prep import load_raw_matches, build_player_match_rows, add_no_leakage_player_features


def merge_player_features_for_training(matches: pd.DataFrame, player_features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "source_row_id",
        "is_winner",
        "player_id",
        "serve_points_won_pct_last20",
        "return_points_won_pct_last20",
        "dias_desde_ultimo_partido",
        "returning_from_injury",
        "minutos_acumulados_torneo",
        "serve_points_won_pct_match",
    ]
    winner_features = (
        player_features.loc[player_features["is_winner"].eq(1), feature_columns]
        .drop(columns=["is_winner"])
        .rename(
            columns={
                "player_id": "jugador_a_id",
                "serve_points_won_pct_last20": "jugador_a_serve_points_won_pct_last20",
                "return_points_won_pct_last20": "jugador_a_return_points_won_pct_last20",
                "dias_desde_ultimo_partido": "jugador_a_dias_desde_ultimo_partido",
                "returning_from_injury": "jugador_a_returning_from_injury",
                "minutos_acumulados_torneo": "jugador_a_minutos_acumulados_torneo",
                "serve_points_won_pct_match": "jugador_a_serve_points_won_pct_match",
            }
        )
    )
    loser_features = (
        player_features.loc[player_features["is_winner"].eq(0), feature_columns]
        .drop(columns=["is_winner"])
        .rename(
            columns={
                "player_id": "jugador_b_id",
                "serve_points_won_pct_last20": "jugador_b_serve_points_won_pct_last20",
                "return_points_won_pct_last20": "jugador_b_return_points_won_pct_last20",
                "dias_desde_ultimo_partido": "jugador_b_dias_desde_ultimo_partido",
                "returning_from_injury": "jugador_b_returning_from_injury",
                "minutos_acumulados_torneo": "jugador_b_minutos_acumulados_torneo",
                "serve_points_won_pct_match": "jugador_b_serve_points_won_pct_match",
            }
        )
    )
    output = matches.merge(winner_features, on="source_row_id", how="left")
    output = output.merge(loser_features, on="source_row_id", how="left")
    return output


def prepare_training_dataset(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in merged_df.iterrows():
        surface = str(row.get("surface", "Hard"))
        best_of = float(row.get("best_of", 3))
        
        # Muestra 1: Jugador A saca, Jugador B resta
        # Solo agregar si tenemos el target real para A
        target_a = row.get("jugador_a_serve_points_won_pct_match")
        if pd.notna(target_a):
            rows.append({
                "target": float(target_a),
                "server_serve_pct_last20": row.get("jugador_a_serve_points_won_pct_last20"),
                "returner_return_pct_last20": row.get("jugador_b_return_points_won_pct_last20"),
                "server_dias": row.get("jugador_a_dias_desde_ultimo_partido"),
                "returner_dias": row.get("jugador_b_dias_desde_ultimo_partido"),
                "server_injury": row.get("jugador_a_returning_from_injury"),
                "returner_injury": row.get("jugador_b_returning_from_injury"),
                "server_minutos": row.get("jugador_a_minutos_acumulados_torneo"),
                "returner_minutos": row.get("jugador_b_minutos_acumulados_torneo"),
                "is_hard": 1 if surface == "Hard" else 0,
                "is_clay": 1 if surface == "Clay" else 0,
                "is_grass": 1 if surface == "Grass" else 0,
                "is_carpet": 1 if surface == "Carpet" else 0,
                "best_of": best_of
            })
            
        # Muestra 2: Jugador B saca, Jugador A resta
        target_b = row.get("jugador_b_serve_points_won_pct_match")
        if pd.notna(target_b):
            rows.append({
                "target": float(target_b),
                "server_serve_pct_last20": row.get("jugador_b_serve_points_won_pct_last20"),
                "returner_return_pct_last20": row.get("jugador_a_return_points_won_pct_last20"),
                "server_dias": row.get("jugador_b_dias_desde_ultimo_partido"),
                "returner_dias": row.get("jugador_a_dias_desde_ultimo_partido"),
                "server_injury": row.get("jugador_b_returning_from_injury"),
                "returner_injury": row.get("jugador_a_returning_from_injury"),
                "server_minutos": row.get("jugador_b_minutos_acumulados_torneo"),
                "returner_minutos": row.get("jugador_a_minutos_acumulados_torneo"),
                "is_hard": 1 if surface == "Hard" else 0,
                "is_clay": 1 if surface == "Clay" else 0,
                "is_grass": 1 if surface == "Grass" else 0,
                "is_carpet": 1 if surface == "Carpet" else 0,
                "best_of": best_of
            })
            
    df = pd.DataFrame(rows)
    # Limpiar nulos en features claves
    df = df.dropna(subset=["server_serve_pct_last20", "returner_return_pct_last20"])
    
    # Imputar el resto de features secundarias
    df["server_dias"] = df["server_dias"].fillna(30.0)
    df["returner_dias"] = df["returner_dias"].fillna(30.0)
    df["server_injury"] = df["server_injury"].fillna(0.0)
    df["returner_injury"] = df["returner_injury"].fillna(0.0)
    df["server_minutos"] = df["server_minutos"].fillna(0.0)
    df["returner_minutos"] = df["returner_minutos"].fillna(0.0)
    
    return df


def main() -> None:
    print("=== Iniciando entrenamiento del regresor de saque ===")
    
    # 1. Cargar y preparar datos históricos
    print("Cargando partidos de Sackmann...")
    matches = load_raw_matches(RAW_DIR)
    
    print("Preparando filas de jugadores...")
    player_rows = build_player_match_rows(matches)
    
    print("Calculando features sin leakage...")
    featured = add_no_leakage_player_features(player_rows)
    
    print("Cruzando características del partido...")
    merged = merge_player_features_for_training(matches, featured)
    
    print("Construyendo dataset final para entrenamiento...")
    dataset = prepare_training_dataset(merged)
    
    print(f"Total registros para el modelo: {len(dataset)}")
    
    # Split temporal para validación (ej. entrenar con <2025 y validar con >=2025)
    # En este script simple dividiremos aleatoriamente o por train_test_split.
    # Dado que los datos ya están libres de leakage por el transform shift(1), un split aleatorio es correcto.
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(dataset))
    split_point = int(len(dataset) * 0.8)
    
    train_indices = shuffled_indices[:split_point]
    test_indices = shuffled_indices[split_point:]
    
    features = [
        "server_serve_pct_last20",
        "returner_return_pct_last20",
        "server_dias",
        "returner_dias",
        "server_injury",
        "returner_injury",
        "server_minutos",
        "returner_minutos",
        "is_hard",
        "is_clay",
        "is_grass",
        "is_carpet",
        "best_of"
    ]
    
    X_train = dataset.iloc[train_indices][features].values
    y_train = dataset.iloc[train_indices]["target"].values
    
    X_test = dataset.iloc[test_indices][features].values
    y_test = dataset.iloc[test_indices]["target"].values
    
    # 2. Entrenar el modelo RandomForest (con regularización para evitar sobreajuste)
    print("Entrenando RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 3. Evaluar el modelo
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\n=== Métricas del Modelo ===")
    print(f"R² Train: {r2_train:.4f} | R² Test (Out-of-sample): {r2_test:.4f}")
    print(f"MSE Train: {mse_train:.6f} | MSE Test: {mse_test:.6f}")
    print(f"RMSE Test: {np.sqrt(mse_test):.4%}")
    
    # Importancia de las variables
    print("\nImportancia de las variables:")
    importances = model.feature_importances_
    for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"  - {name:<30}: {imp:.4%}")
        
    # 4. Exportar el modelo
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "features": features
        }, f)
        
    print(f"\nModelo exportado con éxito a {MODEL_PATH}")


if __name__ == "__main__":
    main()
