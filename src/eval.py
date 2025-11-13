# src/eval.py

import os
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):
    """
    Évalue le modèle sauvegardé sur le dataset préparé.
    """
    # Lecture des données
    data_path = os.path.join(data_dir, "train_processed.csv")
    df = pd.read_csv(data_path)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    
    # Chargement du modèle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Prédictions
    y_pred = model.predict(X)
    
    # Évaluation
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print("Évaluation finale du modèle :")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

if __name__ == "__main__":
    evaluate_model()
