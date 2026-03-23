# Tech Challenge Fase 01: Previsão de Churn de Telecom 📱

Este é o repositório principal do Tech Challenge da Fase 01, focado em construir uma pipeline de ML de ponta a ponta para previsão do cancelamento de clientes (Churn), seguindo as melhores práticas de MLOps.

## 📌 Arquitetura do Projeto
A arquitetura foi escolhida para ser servida em **tempo real** através de uma REST API hospedável em instâncias escaláveis (GCP Run ou AWS Fargate).
Consulte a documentação completa:
- [Arquitetura de Deploy e Monitoramento](docs/deploy_architecture.md)
- [Model Card (Performance e Viéses)](docs/model_card.md)
- [ML Canvas](docs/ml_canvas.md)

## 📁 Estrutura do Repositório
- `data/`: Dataset bruto (ignorado pelo git).
- `models/`: Artefatos locais (preprocessor e pesos da rede neural pytorch).
- `notebooks/`: Análise exploratória (EDA) e treinamento de baselines (Scikit-Learn).
- `src/`: Core do projeto escalável e modular contendo os pacotes `api`, `training` e `models`.
- `tests/`: Bateria de testes usando Pytest (Smoke test, schema validation c/ Pydantic).
- `docs/`: Arquivos complementares de MLOps.

## 🚀 Como Levantar o Projeto Localmente

### 1. Setup do Ambiente
Este projeto usa `pyproject.toml` como Single Source of Truth para as dependências. Requer Python 3.9+.

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependências pelo Makefile
make setup
```

### 2. Baixar o Banco de Dados
O script a seguir baixará via `kagglehub` e copiará para a pasta local automática (Você já deve possuir as credenciais locais no ambiente e rodado o script original da modelagem):
> (No nosso laboratório, o dataset Telco_customer_churn já está em `data/`)

### 3. Execução do Treinamento
Para treinar o modelo do zero, visualizar o early-stopping e coletar novas métricas no MLflow:
```bash
python -m src.training.run
```

### 4. Rodando A API (Inferência em Produção)
Para iniciar o servidor FastAPI:
```bash
make run
# Ou acesse via uvicorn src.api.main:app --reload
```
Acesse o portal do Swagger / Docs em: `http://localhost:8000/docs`

### 5. Checagem de Qualidade (Testes e Lint)
Garantindo que o código não perca formato:
```bash
make test
make lint
```

## Tracking no MLflow
Para visualizar as experimentações anteriores e o tracking dos modelos Baseline (Dummy vs Regressão Logística) contra nossa poderosa MLP PyTorch, rode:
```bash
make mlflow
```
