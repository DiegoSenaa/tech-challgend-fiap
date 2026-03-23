import os
import random

# Adicionamos a importação usando caminho relativo ou adicionando sys.path
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.mlp import ChurnMLP


# 1. Carregamento e Pre-processamento
def load_and_preprocess_data():
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


# Função auxiliar para garantir reprodutibilidade
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 2. Treinamento com Early Stopping e Baseline
def train_mlp():
    set_seeds(42)  # Garantindo reprodutibilidade na inicialização
    (X_train, y_train), (X_val, y_val), (X_test, y_test_t, y_test_true), input_dim = load_and_preprocess_data()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ChurnMLP(input_dim=input_dim, hidden_layers=[64, 32], dropout_rate=0.3)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Telco_Churn_PyTorch")

    print("\n--- Treinando Baseline (Logistic Regression) ---")
    with mlflow.start_run(run_name="Baseline_LogisticRegression"):
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train.numpy(), y_train.squeeze().numpy())
        
        lr_preds = lr_model.predict(X_test.numpy())
        lr_probs = lr_model.predict_proba(X_test.numpy())[:, 1]
        
        lr_metrics = {
            "test_accuracy": accuracy_score(y_test_true, lr_preds),
            "test_precision": precision_score(y_test_true, lr_preds, zero_division=0),
            "test_recall": recall_score(y_test_true, lr_preds),
            "test_f1_score": f1_score(y_test_true, lr_preds),
            "test_roc_auc": roc_auc_score(y_test_true, lr_probs)
        }
        mlflow.log_metrics({f"baseline_{k}": v for k, v in lr_metrics.items()})
        mlflow.sklearn.log_model(lr_model, "baseline_model")
        for k, v in lr_metrics.items():
            print(f" - [Baseline] {k}: {v:.4f}")

    epochs = 150
    patience = 20
    best_val_loss = float('inf')
    counter = 0

    with mlflow.start_run(run_name="MLP_Early_Stopping"):
        mlflow.log_params({"batch_size": batch_size, "hidden_layers": [64, 32], "dropout": 0.3, "lr": 0.001, "pos_weight": float(pos_weight.item()), "patience": patience})

        print("Treinando Rede Neural com Early Stopping...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            train_loss /= len(train_loader.dataset)

            # Validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_loader.dataset)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), "models/best_mlp.pth")
            else:
                counter += 1
                if counter >= patience:
                    print(f"-> Early stopping acionado na época {epoch+1}. Melhor loss: {best_val_loss:.4f}")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)

        # 3. Avaliação no Conjunto de Teste
        print("Avaliando melhor modelo no Dataset de Testes (dados nunca vistos)...")
        model.load_state_dict(torch.load("models/best_mlp.pth", map_location='cpu'))
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            probs = torch.sigmoid(logits).squeeze().numpy()
            preds = (probs >= 0.5).astype(int)

        metrics = {
            "test_accuracy": accuracy_score(y_test_true, preds),
            "test_precision": precision_score(y_test_true, preds, zero_division=0),
            "test_recall": recall_score(y_test_true, preds),
            "test_f1_score": f1_score(y_test_true, preds),
            "test_roc_auc": roc_auc_score(y_test_true, probs)
        }
        mlflow.log_metrics(metrics)
        for k, v in metrics.items():
            print(f" - {k}: {v:.4f}")

        # Análise Custo/Matriz de Confusão
        cm = confusion_matrix(y_test_true, preds)
        tn, fp, fn, tp = cm.ravel()
        print("\nMatriz de Confusão (Trade-off de Negócio):")
        print(f" TN (Ficaram/Acertamos): {tn}")
        print(f" FP (Ficaram/Oferecemos desconto atoa): {fp} -> Custo Moderado")
        print(f" FN (Cancelaram/Não identificamos): {fn} -> Custo Alto (Pior cenário)")
        print(f" TP (Cancelariam/Ação de Retenção): {tp}")

        # Salvando a arquitetura completa do Pytorch no MLflow
        mlflow.pytorch.log_model(model, "model")
        print("Modelo registrado com sucesso no MLflow.")

if __name__ == "__main__":
    train_mlp()
