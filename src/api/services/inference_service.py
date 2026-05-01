import sys
from pathlib import Path

import joblib
import pandas as pd
import torch

# Ajuste de path para alcançar a raiz do projeto (para os imports de src.* funcionarem corretamente)
root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.models.mlp import ChurnMLP


class InferenceService:
    """ Servico de negócio responsável por orquestrar a IA e o pre-processamento """
    def __init__(self):
        self.preprocessor = None
        self.model = None

    def load_artifacts(self):
        """ Carrega o MinMaxScaler/OHE e a Rede Neural na inicialização da aplicação """
        models_dir = root_dir / "models"
        preprocessor_path = models_dir / "preprocessor.pkl"
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)

        model_path = models_dir / "best_mlp.pth"
        feature_names_path = models_dir / "feature_names.pkl"
        if model_path.exists() and feature_names_path.exists():
            input_dim = len(joblib.load(feature_names_path))
            self.model = ChurnMLP(input_dim=input_dim, hidden_layers=[64, 32], dropout_rate=0) # inferência no dropout
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()

    def is_ready(self) -> bool:
        return self.preprocessor is not None and self.model is not None

    def predict(self, customer_data_dict: dict) -> dict:
        if not self.is_ready():
            raise RuntimeError("Model artifacts missing. Treine o modelo primeiro!")

        # 1. Regra de Negócio Padrão: Imputação Simplificada UX
        if customer_data_dict["total_charges"] == -1.0:
            customer_data_dict["total_charges"] = customer_data_dict["monthly_charges"] * customer_data_dict["tenure_months"]

        # 2. Pré-processamento
        df = pd.DataFrame([customer_data_dict])
        tensor_input = self.preprocessor.transform(df)
        tensor_input = torch.FloatTensor(tensor_input)

        # 3. Inferência PyTorch do Score
        with torch.no_grad():
            logits = self.model(tensor_input)
            prob = torch.sigmoid(logits).item()

        # 4. Resultado pós-modelo
        prediction = 1 if prob >= 0.5 else 0
        msg = "Alto risco de Churn!" if prediction == 1 else "Cliente estável."

        return {
            "churn_probability": prob,
            "churn_prediction": prediction,
            "message": msg
        }

# Singleton Global para o app
inference_service = InferenceService()
