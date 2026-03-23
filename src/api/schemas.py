from pydantic import BaseModel, ConfigDict, Field


class CustomerData(BaseModel):
    # Campos Essenciais (Obrigatórios na requisição)
    tenure_months: int = Field(..., description="Meses com a empresa")
    monthly_charges: float = Field(..., description="Cobrança mensal em dólares")
    contract: str = Field(..., description="Month-to-month, One year, Two year")
    internet_service: str = Field(..., description="DSL, Fiber optic, No")
    
    # Campos Secundários (Optativos com valores padrão para simplificação)
    gender: str = Field(default="Female", description="Male ou Female")
    senior_citizen: str = Field(default="0", description="0 (No) ou 1 (Yes)")
    partner: str = Field(default="No", description="Yes ou No")
    dependents: str = Field(default="No", description="Yes ou No")
    phone_service: str = Field(default="Yes", description="Yes ou No")
    multiple_lines: str = Field(default="No", description="Yes, No ou No phone service")
    online_security: str = Field(default="No", description="Yes, No ou No internet service")
    online_backup: str = Field(default="No", description="Yes, No ou No internet service")
    device_protection: str = Field(default="No", description="Yes, No ou No internet service")
    tech_support: str = Field(default="No", description="Yes, No ou No internet service")
    streaming_tv: str = Field(default="No", description="Yes, No ou No internet service")
    streaming_movies: str = Field(default="No", description="Yes, No ou No internet service")
    paperless_billing: str = Field(default="Yes", description="Yes ou No")
    payment_method: str = Field(default="Electronic check", description="Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)")
    total_charges: float = Field(default=-1.0, description="Calculado via tenure se não enviar (-1.0)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure_months": 5,
                "monthly_charges": 80.5,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic"
            }
        }
    )

class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probabilidade [0, 1] de cancelamento")
    churn_prediction: int = Field(..., description="0 para retenção esperada, 1 para alerta de churn")
    message: str = Field(..., description="Mensagem legível sobre a predição")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "churn_probability": 0.85,
                "churn_prediction": 1,
                "message": "Alto risco de Churn!"
            }
        }
    )
