# Software Design Document (SDD) - Telco Churn Prediction

## 1. Visão Geral do Sistema
O **Telco Churn Prediction** é um projeto de Machine Learning projetado para identificar a probabilidade de um cliente cancelar seu plano de telecomunicações (Churn). Ele é construído sobre os princípios fundamentais de **MLOps**, englobando desde rastreamento sistemático de experimentos até servir a aplicação através de uma API resiliente orientada a Clean Architecture.

Este documento foi especificamente redigido para que as IAs generativas de código ou "AI Harness Tools" possam entender instantaneamente a infraestrutura do projeto, seu pipeline de aprendizado e interfaces de serviço para conseguir evoluir o código mantendo o estilo de arquitetura definido.

## 2. Arquitetura de Software e Componentes da Aplicação

### 2.1 Diretórios Principais do Core
- **`src/api/`**: Servidor voltado para inferência em produção. Possui arquitetura em camadas (`controllers` recebem requests da FastAPI, e `services` cuidam da regra de negócio - injeção de dependências e previsões). O entrypoint principal é `src/api/main.py`.
- **`src/models/`**: Definições puras de modelos. Por exemplo, a arquitetura da rede neural profunda criada com `torch.nn.Module`, como visto no arquivo `mlp.py`. O objetivo é que os modelos sejam independentes de fluxo de treino/dados.
- **`src/training/`**: Script modular de treinamento offline. É composto por ingestão de dados (`data_processor.py`), pipeline de preprocessamento padrão (Scikit-Learn ColumnTransformer) e loop de repetição/early_stopping (`trainer.py`).
- **`docs/`**: Central com outras informações de contexto de governança e deployment de infraestrutura.
- **`tests/`**: Suite de validação garantindo aderência da API e regras de inputs com `Pandera` e `Pytest`.

### 2.2 Tecnologias Principais (Stack)
- **Modelagem**: *PyTorch* (Perceptrons/MLP) e *Scikit-Learn* (Pipeline Data Prep e Baselines).
- **Tracking & Model Registry**: *MLflow* parametrizado para arquivar binários do modelo e curvas de validação/early stopping no diretório local `.mlruns/` por padrão DB SQLite (`mlruns.db`).
- **Serviço Online**: *FastAPI* usando concorrência assíncrona com `uvicorn`.
- **Validação de Contratos de Dados**: *Pydantic* no lado API e *Pandera* no nível dos DataFrames tabulares durante treino/testes.

## 3. Data Pipeline & Treinamento

A ingestão e processamento de dados (detalhada em `data_processor.py`) envolve a separação minuciosa de tipos (`categorical_features` vs `num_cols`).
1. **Dados Injetados**: Variáveis da Telco (ex: `tenure`, `MonthlyCharges`, `TotalCharges`, entre dezenas de `object columns` definindo os pacotes da IBM Telco).
2. **Separação Estratificada**: A separação para treino, validação, e teste (que é sagrado) garante distribuições realistas (evitando desbalanceamentos aleatórios).
3. **MLflow Tracking**: Conforme especificado em `trainer.py`, os modelos baselines são comparados à MLP (treinada com PyTorch e uso de Early Stopping). Para replicar testes, é necessário assegurar que "mlflow experiments" continuem recebendo novas *Runs* a cada mudança.

## 4. API Specification & Inference (Deployment)
As rotas de modelagem foram concebidas no ambiente REST com middlewares para latência (`X-Process-Time-Sec`).
- **`GET /health`**: Validar carga do modelo na memória. O container deve checar esse endpoint.
- **`POST /predict`**: O controlador recebe a assinatura exata exigida pelo Pydantic Schema de inferência (baseado nos features). O objeto é convertido internamente no service, passado pelo Scaler (MinMaxScaler/OneHot salvos como pkl/joblib) para gerar matrizes e então é submetido ao `.forward()` do modelo pré-treinado PyTorch carregado no sistema.

## 5. Diretrizes para Replicação/Edição via I.A. 

Se você (LLM) estiver recebendo requests para evoluir este sistema, aplique os seguintes *constraints*:

1. **Clean Code Strict**: Todo IO do modelo no ambiente de produção ocorre no *Service Layer*, `controllers` devem ser o mais burros/simples focados em lidar com HTTPException e resposta Pydantic.
2. **Loggers Structure**: Não adicione nenhum `print()` no console. Importe e configure `import logging` utilizando `logger = logging.getLogger(__name__)`.
3. **Tracking Requirement**: Qualquer algoritmo ou hyper-parâmetro adicionado deve obrigatória ser encapsulado em `with mlflow.start_run` para assegurar rastreabilidade.
4. **Reproducibility Limit**: Treinamento invoca pipelines. Não force substituições manuais de imputers por lógicas de dicionários difíceis de rastrear. Prefira `ColumnTransformer` exportáveis via Joblib.
