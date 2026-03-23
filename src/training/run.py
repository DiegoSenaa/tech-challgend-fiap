"""
Ponto de entrada único para o Pipeline de Treinamento e Avaliação.
Refatorado visando Clean Code e separação de responsabilidades (Orquestração vs Regras).
"""
from src.training.trainer import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline()
