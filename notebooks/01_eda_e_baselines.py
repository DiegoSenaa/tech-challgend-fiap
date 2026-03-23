# %%
# Importação de Bibliotecas
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %%
# Carregamento dos dados
print("Carregando dataset...")
df = pd.read_excel("data/Telco_customer_churn.xlsx")

# Limpeza e Pre-processamento Básico
print("Limpando dados...")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df["total_charges"] = pd.to_numeric(df["total_charges"], errors='coerce')

# Remover NA apenas na coluna total_charges
df = df.dropna(subset=['total_charges'])

# Remover ID do cliente
df = df.drop("customerid", axis=1)

# Converter target para 0 e 1
df["churn_label"] = df["churn_value"] # O dataset já tem churn_value como 0 e 1
df = df.drop(["churn_value", "churn_score", "cltv", "churn_reason", "country", "state", "city", "lat_long", "zip_code"], axis=1, errors="ignore")

# Extrair features numéricas e categóricas
target = "churn_label" # No dataset baixado pode ter variações. Vou ajustar logo abaixo de forma robusta.

# Ajustando target se for objeto 'Yes'/'No'
if "churn" in df.columns and df["churn"].dtype == 'object':
    df["churn"] = df["churn"].map({'Yes': 1, 'No': 0})
    target = "churn"
elif "churn_value" in df.columns:
    target = "churn_value"

X = df.drop(target, axis=1)
y = df[target]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# EDA Rápida (Exibição de stats)
print("Distribuição do Target (Churn):")
print(y.value_counts(normalize=True))

# %%
# Preparação do Pipeline Scikit-Learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), cat_cols)
    ]
)

# Configuração do MLFlow
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Telco_Churn_Baselines")

def train_and_log_model(model_name, model):
    with mlflow.start_run(run_name=model_name):
        # Pipeline completo
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])

        # Treinamento
        pipeline.fit(X_train, y_train)

        # Predições
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

        # Métricas
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        # Log no MLFlow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"--- Modelo: {model_name} ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("\n")

# %%
# Treinamento dos Baselines
print("Iniciando treinamento dos Baselines...")
train_and_log_model("Dummy_Classifier_Baseline", DummyClassifier(strategy="stratified", random_state=42))
train_and_log_model("Logistic_Regression_Baseline", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))

print("Treinamento de baselines finalizado e salvo no MLflow.")
