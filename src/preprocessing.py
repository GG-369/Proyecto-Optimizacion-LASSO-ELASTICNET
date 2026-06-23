import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_housing_data(filepath):
    """
    Carga el dataset de Lima, separa las variables, hace el split de 
    entrenamiento/prueba y estandariza los predictores de forma segura 
    (evitando data leakage).
    
    Parámetros:
    -----------
    filepath : str
        Ruta al archivo de datos (.csv o .xlsx)
        
    Retorna:
    --------
    X_train_scaled, X_test_scaled : matrices de diseño estandarizadas.
    y_train, y_test : vectores objetivo (log del precio).
    feature_cols : lista con los nombres de las variables predictoras.
    scaler : el objeto StandardScaler ajustado (útil para predecir nuevos datos).
    """
    
    # 1. Carga de datos (soporta tanto su Excel original como el CSV que tienes)
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
        
    # 2. Definir Target (log del precio) y Features
    # Exactamente como lo definió el equipo de Elastic Net
    y = df['precio_usd_log'].values
    feature_cols = [c for c in df.columns if c not in ['Precio', 'precio_usd_log']]
    X_raw = df[feature_cols].values
    
    # 3. Train/Test split (80/20)
    # CRÍTICO: random_state=42 asegura que ambos equipos tengan las mismas casas en train y test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )
    
    # 4. Estandarización
    # Se ajusta (fit) solo con los datos de entrenamiento para evitar data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler