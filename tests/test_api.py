from fastapi.testclient import TestClient

from src.api.main import app


def test_health_check_smoke_test():
    """
    Smoke Test: Verifica se a API subiu e se carregou os artefatos corretamente.
    """
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_predict_schema_validation():
    """
    Schema / Pydantic Test: Valida se a API barra payloads incorretos ou faltantes.
    """
    payload = {
        "gender": "Male",
        "senior_citizen": "No"
        # Faltam todos os outros campos, a API deve recusar
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422 # Unprocessable Entity (Erro de validação)

def test_predict_success():
    """
    Unit/Integration Test: Envia um cliente válido e verifica a inferência.
    """
    payload = {
        "gender": "Female",
        "senior_citizen": "0",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 2,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 50.0,
        "total_charges": 100.0
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        json_resp = response.json()
        assert "churn_probability" in json_resp
        assert "churn_prediction" in json_resp
        assert json_resp["churn_prediction"] in [0, 1]
        assert "message" in json_resp
