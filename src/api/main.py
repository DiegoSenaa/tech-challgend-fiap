import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.api.controllers.prediction_controller import router as predict_router
from src.api.services.inference_service import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carregamento pesado é delegado ao Service Layer durante o startup
    inference_service.load_artifacts()
    yield

app = FastAPI(
    title="Telco Churn Prediction API",
    description="API (Clean Code) para prever cancelamento de clientes usando Clean Architecture e PyTorch",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    """ Middleware global para métricas de tempo """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time-Sec"] = f"{process_time:.4f}"
    return response

# Acoplamento de controladoras
app.include_router(predict_router)
