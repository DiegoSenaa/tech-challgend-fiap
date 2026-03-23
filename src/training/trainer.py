import os
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import ChurnMLP
from src.training.data_processor import load_and_preprocess_data


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_training_pipeline():
    """ Orquestra a injeção dos dados, treinamento Baseline e da Rede Neural """
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
