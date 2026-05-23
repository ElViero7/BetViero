import pandas as pd
import numpy as np
from pathlib import Path

# Configurar rutas
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "tennis" / "data" / "processed" / "backtest_bets_history.csv"

def print_dimension_table(df, dimension_name, col_name, categories):
    """
    Imprime una tabla comparativa para una dimensión dada (2022-2025 vs 2026).
    """
    print(f"\n=== DIMENSIÓN: {dimension_name} ===")
    print(f"{'Categoría':<25} | {'2022-2025':^30} | {'2026':^30} | {'Diferencia ROI':^15}")
    print(f"{'':<25} | {'Vol. / WR% / ROI%':^30} | {'Vol. / WR% / ROI%':^30} | {'':^15}")
    print("-" * 110)
    
    # Crear la columna de periodo si no existe
    if 'periodo' not in df.columns:
        df['periodo'] = df['año'].apply(lambda y: '2022-2025' if y in [2022, 2023, 2024, 2025] else ('2026' if y == 2026 else 'Otro'))
        
    for cat in categories:
        # Filtrar por categoría
        df_cat = df[df[col_name] == cat]
            
        # Stats 2022-2025
        df_p1 = df_cat[df_cat['periodo'] == '2022-2025']
        n_p1 = len(df_p1)
        wr_p1 = (df_p1['win'].sum() / n_p1) * 100 if n_p1 > 0 else 0.0
        roi_p1 = (df_p1['pnl'].sum() / n_p1) * 100 if n_p1 > 0 else 0.0
        
        # Stats 2026
        df_p2 = df_cat[df_cat['periodo'] == '2026']
        n_p2 = len(df_p2)
        wr_p2 = (df_p2['win'].sum() / n_p2) * 100 if n_p2 > 0 else 0.0
        roi_p2 = (df_p2['pnl'].sum() / n_p2) * 100 if n_p2 > 0 else 0.0
        
        diff_roi = roi_p2 - roi_p1
        
        p1_str = f"{n_p1:5d} / {wr_p1:5.1f}% / {roi_p1:+6.2f}%" if n_p1 > 0 else "      -      "
        p2_str = f"{n_p2:5d} / {wr_p2:5.1f}% / {roi_p2:+6.2f}%" if n_p2 > 0 else "      -      "
        diff_str = f"{diff_roi:+6.2f}%" if (n_p1 > 0 and n_p2 > 0) else "   -   "
        
        print(f"{cat:<25} | {p1_str:<30} | {p2_str:<30} | {diff_str:^15}")
    print("-" * 110)

def get_cuota_bracket(odds):
    if odds < 1.80:
        return "Cuotas < 1.80"
    elif 1.80 <= odds <= 2.20:
        return "Cuotas [1.80, 2.20]"
    else:
        return "Cuotas > 2.20"

def get_ev_bracket(ev):
    if ev <= 0.10:
        return "EV Moderado (2% - 10%)"
    else:
        return "EV Alto (> 10%)"

def main():
    if not CSV_PATH.exists():
        print(f"Error: No se encontró el archivo de historial de apuestas en: {CSV_PATH}")
        print("Por favor, asegúrate de ejecutar primero el script de backtest para generar los datos.")
        return
        
    df = pd.read_csv(CSV_PATH)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['año'] = df['fecha'].dt.year
    df['win'] = df['win'].astype(bool)
    
    print("\n" + "="*80)
    print(" REPORTE DE ANÁLISIS DE CAÍDA DE ROI (2022-2025 vs 2026)")
    print("="*80)
    print(f"Total apuestas cargadas: {len(df):,}")
    
    # 1. Por Tipo de Mercado
    print_dimension_table(
        df=df,
        dimension_name="TIPO DE MERCADO",
        col_name="mercado",
        categories=["H2H", "Totals", "Spreads"]
    )
    
    # 2. Por Superficie
    # Normalizar superficie a Title case
    df['superficie_norm'] = df['superficie'].astype(str).str.strip().str.title()
    print_dimension_table(
        df=df,
        dimension_name="SUPERFICIE",
        col_name="superficie_norm",
        categories=["Hard", "Clay", "Grass"]
    )
    
    # 3. Por Tramo de Cuota
    df['cuota_bracket'] = df['cuota'].apply(get_cuota_bracket)
    print_dimension_table(
        df=df,
        dimension_name="TRAMO DE CUOTA",
        col_name="cuota_bracket",
        categories=["Cuotas < 1.80", "Cuotas [1.80, 2.20]", "Cuotas > 2.20"]
    )
    
    # 4. Por Rango de EV
    df['ev_bracket'] = df['ev'].apply(get_ev_bracket)
    print_dimension_table(
        df=df,
        dimension_name="RANGO DE EV",
        col_name="ev_bracket",
        categories=["EV Moderado (2% - 10%)", "EV Alto (> 10%)"]
    )

if __name__ == "__main__":
    main()
