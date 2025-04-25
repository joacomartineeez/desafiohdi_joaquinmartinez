# src/scoring.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, confusion_matrix
from joblib import load

def compute_recall_at_40(y, proba):
    """
    Calcula recall, matriz de confusión, umbral y predicciones
    para el 40% de alertas.
    """
    thr = np.percentile(proba, 60)
    pred = (proba >= thr).astype(int)
    return recall_score(y, pred), confusion_matrix(y, pred), thr, pred

def compute_monto_and_dfs(proba, y, montos, claim_ids):
    """
    Construye DataFrame con scores y montos, devuelve:
      - monto capturado entre los TP del top 40%
      - DataFrame top40
      - DataFrame completo
    """
    df = pd.DataFrame({
        'claim_id':     claim_ids,
        'score':        proba,
        'fraud':        y,
        'monto_fraude': montos
    })
    n40     = int(0.4 * len(df))
    top40   = df.nlargest(n40, 'score')
    monto   = top40.loc[top40['fraud'] == 1, 'monto_fraude'].sum()
    return monto, top40, df

def main():
    # 1) Carga datos procesados
    df       = pd.read_csv('data/processed/fraud_prepared_with_id.csv')
    features = [
        'MARCA_VEHICULO_te','lag_dias','PRIMA_MENSUAL_UF',
        'PRODUCTO_te','ANIO_VEHICULO','CANTIDAD_HIJOS'
    ]
    X         = df[features]
    y         = df['fraude_bin'].values
    montos    = df['monto_fraude'].values
    claim_ids = df['claim_id'].values

    # 2) Carga modelo serializado
    calib    = load('outputs/models/catboost_calib.pkl')
    proba_cb = calib.predict_proba(X)[:, 1]

    # 3) Calcula métricas @40%
    recall40, cm, thr40, pred40 = compute_recall_at_40(y, proba_cb)
    monto_cap, top40_df, all_df = compute_monto_and_dfs(proba_cb, y, montos, claim_ids)

    # 4) Matriz de Confusión (paleta verde pastel)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap='Greens', interpolation='nearest', alpha=0.6)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_xticklabels(['No Fraud','Fraud'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['No Fraud','Fraud'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Matriz de Confusión – CatBoost Afinado')
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha='center', va='center', fontweight='bold')
    plt.tight_layout()

    # 5) Métricas como texto en gráfico
    fig2, ax2 = plt.subplots(figsize=(10,1.5))
    ax2.axis('off')
    text = (
        f"Tasa de Detección de Fraudes (Recall@40%): {recall40:.3f}\n"
        f"Monto capturado: {monto_cap:,.0f} [unidades monetarias]\n"
        "Dataset de fraudes predichos guardado en data/processed/predicted_frauds.csv"
    )
    # fondo pastel
    rect = plt.Rectangle((0,0),1,1, transform=ax2.transAxes,
                         facecolor='lightgreen', alpha=0.3)
    ax2.add_patch(rect)
    ax2.text(0.02, 0.5, text, fontsize=12, va='center')
    plt.tight_layout()
    plt.show()

    # 6) Top-10 en Markdown
    top10 = top40_df[['claim_id','score','monto_fraude']].head(10).reset_index(drop=True)
    print("\nTop 10 Siniestros por Probabilidad de Fraude (Markdown):\n")
    print(top10.to_markdown(index=False))

     # 7) Exporta todos los fraudes detectados
    frauds = (
       all_df[all_df['score'] >= thr40]
       [['claim_id','score','monto_fraude']]
       .sort_values('score', ascending=False)
       .reset_index(drop=True)
    )
    frauds.to_csv('data/processed/predicted_frauds.csv', index=False)

if __name__ == '__main__':
    main()