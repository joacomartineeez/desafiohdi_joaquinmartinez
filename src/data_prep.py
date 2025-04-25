import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import category_encoders as ce

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='|')
    return df

def filter_and_map_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['FRAUD'].isin(['Fraude','Descartado'])].reset_index(drop=True)
    df['fraude_bin'] = (df['FRAUD']=='Fraude').astype(int)
    return df

def prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    df['FEC_DENUNCIO']  = pd.to_datetime(df['FEC_DENUNCIO'], errors='coerce')
    df['FEC_SINIESTRO'] = pd.to_datetime(df['FEC_SINIESTRO'], errors='coerce')
    df['FEC_SINIESTRO'].fillna(df['FEC_DENUNCIO'], inplace=True)
    df['lag_dias'] = (df['FEC_DENUNCIO'] - df['FEC_SINIESTRO']).dt.days
    return df

def handle_missing_and_encode(df: pd.DataFrame) -> pd.DataFrame:
     # Eliminar CANTIDAD_AUTOS
    df.drop(columns=['CANTIDAD_AUTOS'], inplace=True)
    # CANTIDAD_HIJOS
    df['CANTIDAD_HIJOS'] = df['CANTIDAD_HIJOS'].replace(999, np.nan)
    df['CANTIDAD_HIJOS'].fillna(df['CANTIDAD_HIJOS'].mode()[0], inplace=True)
    df['CANTIDAD_HIJOS'] = df['CANTIDAD_HIJOS'].astype(int)
    # ANIO_VEHICULO
    df['ANIO_VEHICULO'] = df['ANIO_VEHICULO'].fillna(df['ANIO_VEHICULO'].median()).astype(int)
    # PRODUCTO → agrupar rarezas + target encode
    df['PRODUCTO'] = df['PRODUCTO'].astype(str).fillna('Desconocido')
    freqs = df['PRODUCTO'].value_counts(normalize=True)
    rares = freqs[freqs < 0.01].index
    df['PRODUCTO_grp'] = df['PRODUCTO'].replace(rares, 'Otros')
    te_prod = ce.TargetEncoder(cols=['PRODUCTO_grp'], smoothing=0.3)
    df['PRODUCTO_te'] = te_prod.fit_transform(df[['PRODUCTO_grp']], df['fraude_bin'])
    # ESTADO_CIVIL → one-hot
    mapping_ec = {
        'Casada/o':'Casado/a','Casada':'Casado/a',
        'Divorciado':'Divorciado/a','Divorciado/a':'Divorciado/a'
    }
    df['ESTADO_CIVIL'] = df['ESTADO_CIVIL'].replace(mapping_ec).fillna('Desconocido')
    ohe_ec = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ec_arr = ohe_ec.fit_transform(df[['ESTADO_CIVIL']])
    df[ohe_ec.get_feature_names_out(['ESTADO_CIVIL'])] = ec_arr
    df.drop(columns=['ESTADO_CIVIL'], inplace=True)
    # MARCA_VEHICULO → target encode
    df['MARCA_VEHICULO'] = df['MARCA_VEHICULO'].fillna('Desconocido')
    te_marca = ce.TargetEncoder(cols=['MARCA_VEHICULO'], smoothing=0.3)
    df['MARCA_VEHICULO_te'] = te_marca.fit_transform(df[['MARCA_VEHICULO']], df['fraude_bin'])
    # PRIMA_MENSUAL_UF
    df['PRIMA_MENSUAL_UF'].fillna(df['PRIMA_MENSUAL_UF'].median(), inplace=True)
    # ROBO category
    df['ROBO'] = df['ROBO'].astype('category')
    # DEDUCIBLE quartiles + one-hot
    df['ded_q'] = pd.qcut(df['DEDUCIBLE'], 4, labels=[f'Q{i}' for i in range(1,5)])
    ohe_ded = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ded_arr = ohe_ded.fit_transform(df[['ded_q']])
    df[ohe_ded.get_feature_names_out(['ded_q'])] = ded_arr
    df['DEDUCIBLE_scaled'] = StandardScaler().fit_transform(df[['DEDUCIBLE']])
    df.drop(columns=['ded_q','DEDUCIBLE'], inplace=True)
    # CANAL_CONTRATACION one-hot
    ohe_can = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    can_arr = ohe_can.fit_transform(df[['CANAL_CONTRATACION']])
    df[ohe_can.get_feature_names_out(['CANAL_CONTRATACION'])] = can_arr
    df.drop(columns=['CANAL_CONTRATACION'], inplace=True)
    return df

def finalize_and_save(df_enc: pd.DataFrame, df_raw: pd.DataFrame, out_path: str):
    df_enc = df_enc.reset_index(drop=True)
    df_raw = df_raw.reset_index(drop=True)
    df_enc['claim_id']     = df_raw['CLAIM_ID'].values
    df_enc['monto_fraude'] = df_raw['MONTO_FRAUDE'].values
    selected = [
        'claim_id','monto_fraude',
        'MARCA_VEHICULO_te','lag_dias','PRIMA_MENSUAL_UF',
        'PRODUCTO_te','ANIO_VEHICULO','CANTIDAD_HIJOS','fraude_bin'
    ]
    df_enc[selected].to_csv(out_path, index=False)

def main():
    raw = load_raw('data/raw/dataset.csv')
    filt = filter_and_map_target(raw)
    df = prepare_dates(filt)
    df = handle_missing_and_encode(df)
    finalize_and_save(df, filt, 'data/processed/fraud_prepared_with_id.csv')
    print("Data preparada y guardada.")

if __name__ == '__main__':
    main()