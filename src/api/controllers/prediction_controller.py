from fastapi import APIRouter, HTTPException

from src.api.schemas import CustomerData, PredictionResponse
from src.api.services.inference_service import inference_service

router = APIRouter()

@router.get("/health", tags=["System"])
def health_check():
    """ Verifica integridade da API e se o modelo AI foi carregado localmente """
    if not inference_service.is_ready():
        raise HTTPException(status_code=503, detail="Model artifacts not fully loaded")
    return {"status": "ok", "message": "API and Model are up and running!"}

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prevê risco de Churn",
    description="Recebe características do cliente e delega ao serviço de inferência IA.",
    tags=["Machine Learning"]
)
def predict_churn(customer: CustomerData):
    """ Controller Limpo: Apenas delega o payload pro Service receber a resposta de domínio """
    try:
        data_dict = customer.model_dump()
        result = inference_service.predict(data_dict)
        return PredictionResponse(**result)
    except RuntimeError as ex:
        raise HTTPException(status_code=503, detail=str(ex))
