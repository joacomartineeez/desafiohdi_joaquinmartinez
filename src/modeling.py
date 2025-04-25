import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score, confusion_matrix
import numpy as np
import joblib

def train_and_calibrate(path_processed: str):
    df = pd.read_csv(path_processed)
    features = [
        'MARCA_VEHICULO_te','lag_dias','PRIMA_MENSUAL_UF',
        'PRODUCTO_te','ANIO_VEHICULO','CANTIDAD_HIJOS'
    ]
    X = df[features]
    y = df['fraude_bin'].values

    model = CatBoostClassifier(
        iterations=200, auto_class_weights='Balanced', depth=10,
        learning_rate=0.001, l2_leaf_reg=10, bagging_temperature=1.0,
        border_count=32, random_seed=42, verbose=False
    )
    model.fit(X, y)
    calib = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calib.fit(X, y)
    return calib, X, y


def main():
    calib, X, y = train_and_calibrate('data/processed/fraud_prepared_with_id.csv')
    # Serializar
    joblib.dump(calib, 'outputs/models/catboost_calib.pkl')
    print("Modelo entrenado y serializado en outputs/models/catboost_calib.pkl")

if __name__ == '__main__':
    main()