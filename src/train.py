# src/train.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def train_model(data_dir="data/processed", output_dir="models"):
    """
    Entraîne un modèle LinearRegression sur les données préparées.
    Sauvegarde le modèle dans output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Lecture des données préparées
    data_path = os.path.join(data_dir, "train_processed.csv")
    df = pd.read_csv(data_path)
    
    # Définition des features et cible
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    
    # Binning pour stratification
    y_binned = pd.cut(y, bins=10, labels=False)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y_binned
    )
    
    # Entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("Évaluation du modèle LinearRegression :")
    print(f"RMSE Train : {rmse_train:.2f}")
    print(f"RMSE Test  : {rmse_test:.2f}")
    print(f"R² Train   : {r2_train:.4f}")
    print(f"R² Test    : {r2_test:.4f}")
    
    # Sauvegarde du modèle
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé dans : {model_path}")

if __name__ == "__main__":
    train_model()
