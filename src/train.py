import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os
import yaml

def train_model(data_dir="data/processed", output_dir="models"):
    # Charger la config
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    train_params = params.get("train", {})
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv")
    
    # Si y_train est un DataFrame avec une colonne, extraire la série
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    
    model = LinearRegression(fit_intercept=train_params.get("fit_intercept", True))
    model.fit(X_train, y_train)
    
    with open(f"{output_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Modèle entraîné et sauvegardé.")

if __name__ == "__main__":
    train_model()
