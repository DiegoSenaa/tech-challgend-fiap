import pytest
import pandera as pa
import pandas as pd
from pandera import Column, DataFrameSchema, Check

# Definindo o Schema Esperado para Telecom Churn (Simplificado)
churn_schema = DataFrameSchema({
    "tenure": Column(int, Check.ge(0), nullable=False),
    "monthlycharges": Column(float, Check.ge(0.0), nullable=False),
    "totalcharges": Column(float, Check.ge(0.0), nullable=True),
    "churn_label": Column(str, Check.isin(["Yes", "No"]), nullable=False)
})

def test_raw_data_schema():
    """
    Testa se o dataset de entrada obedece o contrato das features principais
    antes do pré-processamento.
    """
    # Cria pequeno dummy para representar a tabela que vem pro sistema
    data = {
        "tenure": [1, 34, 2, 45],
        "monthlycharges": [29.85, 56.95, 53.85, 42.30],
        "totalcharges": [29.85, 1889.5, 108.15, 1840.75],
        "churn_label": ["No", "No", "Yes", "No"]
    }
    
    df = pd.DataFrame(data)

    try:
        churn_schema.validate(df)
        assert True
    except pa.errors.SchemaError as exc:
        pytest.fail(f"Falha na validação de schema: {exc}")

def test_invalid_schema_data_types():
    """ Testa se pandera pega tipos de dados inválidos ou violações de regras (negativos) """
    invalid_data = {
        "tenure": [-5, 34], # tenure não deve ser negativo
        "monthlycharges": [29.85, "invalid"], # string em float
        "totalcharges": [29.85, 1889.5],
        "churn_label": ["Maybe", "No"] # Maybe não está na lista de aceitos
    }
    
    df = pd.DataFrame(invalid_data)
    
    with pytest.raises(pa.errors.SchemaError):
        churn_schema.validate(df)
