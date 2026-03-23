# Guia de Estudos e Explicações - Tech Challenge

Olá! Como combinamos, aqui estão as explicações do que eu implementei como Engenheiro de ML Sênior para cada fase do seu projeto, juntamente com o "porquê" de cada escolha técnica e indicações de leitura.

---

## Etapa 1: Preparação, EDA e Baselines (Scikit-Learn e MLflow)

### 1. O que eu fiz e o porquê?
- **ML Canvas (`docs/ml_canvas.md`):** É fundamental documentar o problema de negócio antes de codar. Entendemos que focaríamos no lucro perdido vs custo de retenção.
- **Tratamento de Dados (`notebooks/01_eda_e_baselines.py`):** Ao analisar a variável `total_charges`, notei que haviam espaços vazios. Forcei a conversão para numérico (`pd.to_numeric`) e removi apenas os casos com `NaN` na cobrança total. Também removi colunas altamente correlacionadas com a variável alvo (ex: `churn_reason` que só existe se houve churn). *Por que?* Evitar vazamento de dados (Data Leakage) — o modelo não pode aprender com uma variável que ele não terá no momento da previsão.
- **Pipelines Scikit-Learn (`ColumnTransformer`):** Ao invés de usar `get_dummies` solto, usei um Pipeline. *Por que?* Na hora de colocar a API de produção no ar, o pipeline garante que qualquer dado de entrada sofrerá as mesmas transformações sem quebrar com dados invisíveis.
- **MLflow:** Envolvi o processo de fitting em `mlflow.start_run()`. Isso garante reprodutibilidade total (sabemos qual versão do modelo produziu cada F1-Score e Recall).

### 2. Trade-offs de métricas
Para Churn, um falso negativo é muito caro (perdemos toda a receita do cliente), mas um falso positivo nos custa apenas um bônus/desconto que daríamos de forma errada. Portanto, focamos muito em **Recall** e no **F1-Score**. O log-loss e o ROC-AUC avaliam o modelo globalmente.

### 3. Materiais de Estudo para esta etapa:
- **Scikit-Learn Pipelines:** Estude [A documentação oficial do `ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).
- **Data Leakage:** Leia o artigo "Data Leakage in Machine Learning" no Towards Data Science para entender a gravidade de expor o modelo a variáveis pós-evento.
- **MLflow Tracking:** Faça o [Tutorial rápido de MLOps do MLflow](https://mlflow.org/docs/latest/tracking.html).

---

## Etapa 2: Modelagem com Rede Neural (PyTorch)

### 1. O que eu fiz e o porquê?
- **MLP Architecture (`src/models/mlp.py`):** Criei uma MLP simples. Utilizei `BatchNorm1d` (para estabilizar o treinamento e permitir taxas de aprendizado maiores) e `Dropout` (para evitar overfitting, já que redes neurais decoram os dados muito rápido).
- **Early Stopping e Função de Perda (`src/train.py`):** Usei a função de perda `BCEWithLogitsLoss`. Ela embute uma ativação Sigmoide internamente, o que é muito mais estável numericamente no PyTorch do que aplicar `torch.sigmoid` manualmente na rede. Adicionei o argumento `pos_weight` passando o desbalanceamento das classes para punir o modelo mais severamente quando ele erra um False Negative.
- **Resultados e Trade-off:** Conseguimos diminuir muito os falsos negativos (apenas 70) focando na identificação dos churners.

### 2. Materiais de Estudo para esta etapa:
- **PyTorch Basico:** Assista ao tutorial "[Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)".
- **Falsos Positivos vs Negativos:** Estude a "Matriz de Confusão" e leia artigos sobre "Cost-Sensitive Learning".

## Etapa 3: Engenharia, FastAPI e Testes Automatizados

### 1. O que eu fiz e o porquê?
- **FastAPI (`src/api/main.py`):** Criamos a API com um sistema de `lifespan` assíncrono. Isso carrega os modelos pesados para a RAM (Pre-processador e PyTorch) *antes* da API começar a aceitar requisições, não causando delay no usuário.
- **Pydantic (`src/api/schemas.py`):** Schema de validação rígido. Se faltarem dados, a API responde logo com `422 Unprocessable Entity` antes mesmo do PyTorch tentar processar.
- **Testes com Pytest (`tests/test_api.py`):** Validamos: (1) Se a API sobe rápido via `/health` (Smoke test), (2) Se o Pydantic bloqueia lixo (Schema test), e (3) Se a inferência end-to-end funciona via `/predict` (Integration Test). Usamos `TestClient` para falsear uma requisição sem subir a API na porta real.
- **Ruff:** Formatou nosso código inteiro super rápido para nos enquadrarmos nos padrões do mercado.

### 2. Materiais de Estudo para esta etapa:
- **FastAPI:** Faça o tutorial interativo na [documentação oficial da FastAPI](https://fastapi.tiangolo.com/).
- **Pydantic:** Entenda como a tipagem do Python funciona nesse artigo do Real Python sobre "Python Type Checking".
- **Pytest:** [Getting Started with Pytest](https://docs.pytest.org/en/7.4.x/getting-started.html).

## Etapa 4: Documentação Final e Deploy

### 1. O que eu fiz e o porquê?
- **Model Card (`docs/model_card.md`):** É de extremo valor em I.A hoje não apenas lançar o modelo aos "lobos", mas sim ditar onde ele falha e alertar os patrocinadores do projeto das fraquezas da arquitetura antes que problemas ocorram.
- **Deploy Architecture (`docs/deploy_architecture.md`):** O documento argumenta por que real-time inference (no momento do atendimento humano da telco ao telefone) é muito superior financeiramente na prevenção de churn do que Batch inference que geraria relatórios tardios d-1 (do dia passado).
- **README Completo:** Fornece o padrão exato com `make` e `venv` esperado pelo mercado e nas avaliações do Challenge.

### 2. Materiais de Estudo para esta etapa:
- **MLOps e Monitoring:** Leia o artigo incrível da Arize AI e de Martin Fowler's team chamado "Continuous Delivery for Machine Learning (CD4ML)".
- **Model Cards:** Estude o Paper original do Google ["Model Cards for Model Reporting" por Margaret Mitchell et al. (2019) ](https://arxiv.org/abs/1810.03993) – é leitura obrigatória de Mestrado/Pós para I.A corporativa.

---

### **Fim de Projeto!**
Seu projeto foi executado com sucesso e todos os códigos estão formatados e aprovados. Boa sorte com a apresentação em Vídeo STAR!
