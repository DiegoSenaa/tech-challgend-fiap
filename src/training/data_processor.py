import os
import joblib
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_preprocess_data():
    """ Carrega os bancos de dados, higieniza as features, realiza Split e aplica Feature Engineering """
    print("Iniciando pré-processamento para a Rede Neural PyTorch...")
    df = pd.read_excel("data/Telco_customer_churn.xlsx")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors='coerce')
    df = df.dropna(subset=['total_charges'])
    df = df.drop(["customerid", "churn_value", "churn_score", "cltv", "churn_reason", "country", "state", "city", "lat_long", "zip_code", "count", "latitude", "longitude"], axis=1, errors="ignore")

    # Mapeando a variável alvo se for object
    target = "churn_label"
    if df[target].dtype == 'O':
        df[target] = df[target].map({'Yes': 1, 'No': 0})

    X = df.drop(target, axis=1)
    y = df[target].values

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False), cat_cols)
        ]
    )

    # Divisão: Train (64%), Val (16%), Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Salvando preprocessor para uso na inferência / API
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    # Pegando a assinatura (nomes das colunas com one-hot)
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + cat_feature_names.tolist()
    joblib.dump(all_feature_names, "models/feature_names.pkl")

    X_train_t = torch.FloatTensor(X_train_trans)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)

    X_val_t = torch.FloatTensor(X_val_trans)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)

    X_test_t = torch.FloatTensor(X_test_trans)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1)

    return (X_train_t, y_train_t), (X_val_t, y_val_t), (X_test_t, y_test_t, y_test), X_train_trans.shape[1]
