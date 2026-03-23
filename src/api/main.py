import os
import sys
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, Request

from .schemas import CustomerData, PredictionResponse

# Ajusta path para importar o model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mlp import ChurnMLP

# Variáveis globais para armazenar os artefatos carregados
preprocessor = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, model
    # Carregar preprocessor do Scikit-Learn
    preprocessor_path = "models/preprocessor.pkl"
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)

    # Carregar modelo PyTorch
    model_path = "models/best_mlp.pth"
    feature_names_path = "models/feature_names.pkl"
    if os.path.exists(model_path) and os.path.exists(feature_names_path):
        input_dim = len(joblib.load(feature_names_path))
        model = ChurnMLP(input_dim=input_dim, hidden_layers=[64, 32], dropout_rate=0) # Dropout 0 para inferência
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    yield
    # Limpeza se necessário

app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para prever o cancelamento de clientes (Churn) utilizando PyTorch",
    version="1.0.0",
    lifespan=lifespan
)



@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    """ Middleware para logging de latência """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time-Sec"] = f"{process_time:.4f}"
    return response

@app.get("/health")
def health_check():
    if model is None or preprocessor is None:
        return {"status": "error", "message": "Model artifacts not fully loaded"}
    return {"status": "ok", "message": "API and Model are up and running!"}

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prevê risco de Churn",
    description="Recebe as características demográficas e de faturamento de um cliente e retorna a probabilidade de cancelamento.",
    tags=["Machine Learning"]
)
def predict_churn(customer: CustomerData):
    # Converte o request Pydantic para DataFrame do Pandas (para o preprocessor)
    data_dict = customer.model_dump()
    
    # UX Simplificado: Se não enviarem total_charges, aproximamos (Mensalidade x Meses)
    if data_dict["total_charges"] == -1.0:
        data_dict["total_charges"] = data_dict["monthly_charges"] * data_dict["tenure_months"]

    df = pd.DataFrame([data_dict])

    # Pre-processamento
    tensor_input = preprocessor.transform(df)
    tensor_input = torch.FloatTensor(tensor_input)

    # Inferência PyTorch
    with torch.no_grad():
        logits = model(tensor_input)
        prob = torch.sigmoid(logits).item()

    prediction = 1 if prob >= 0.5 else 0
    msg = "Alto risco de Churn!" if prediction == 1 else "Cliente estável."

    return PredictionResponse(
        churn_probability=prob,
        churn_prediction=prediction,
        message=msg
    )
