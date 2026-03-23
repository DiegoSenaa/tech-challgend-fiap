# Model Card - Telco Customer Churn MLP

## 1. Detalhes do Modelo
- **Model Name:** Telco Churn Predictor (MLP API)
- **Versão:** 1.0.0
- **Data:** Março de 2026
- **Desenvolvedores:** Grupo do Tech Challenge Fase 1 / Diego
- **Tipo de Modelo:** Rede Neural Perceptron Multicamadas (PyTorch). Classificador binário.
- **Licença:** MIT
- **Informações do Treinamento:** Batch Size de 64, Otimizador Adam com Learning Rate de 0.001. A função de perda foi BCEWithLogitsLoss modificada com um peso (`pos_weight`) para penalizar agressivamente o erro em casos da classe minoritária. O treinamento utilizou parada antecipada (Early Stopping) de 20 épocas no set de validação.

## 2. Uso Intencional (Intended Use)
- **Primary Use Case:** Previsão da probabilidade de um cliente cancelar sua conta na Telecom no próximo ciclo de faturamento.
- **Out of Scope:** O modelo NÃO deve ser executado para mercados em países com regulações demográficas diferentes dos EUA, nem para outras verticais que não sejam Telco.

## 3. Fatores, Vieses e Limitações
- **Vieses Demográficos Potenciais:** O modelo considera `gender` e `senior_citizen`. Como o grupo histórico reflete padrões sociais antigos (pode haver desequilíbrio), o modelo pode atribuir taxas de risco diferentes baseadas puramente na idade ou gênero de forma discriminatória. A sugestão para V2 é testar algoritmos de métrica de fairness.
- **Limitações:** Falsos positivos geram custo leve, mas priorizamos o Recall para diminuir ao máximo os Falsos Negativos.

## 4. Dados
- **Tipo de Dataset:** Tabular estruturado (Features demográficas, do usuário e de faturamento).
- **Proporções de Treinamento, Validação e Teste:** Split estratificado de dados (64% Treino, 16% Validação e 20% Testes invisíveis ao modelo).

## 5. Avaliação (Evaluation Metrics) no holdout set (20%)
- **Accuracy:** 71%
- **F1-Score:** 60.2%
- **Recall (Métrica principal de negócio):** 81.3%
- **Precisão:** 47.8%
- **ROC-AUC:** ~84%

## 6. Cenários de Falha (Failure Scenarios)
- Clientes muito novos (menor de 1 mês de casa) têm pouco histórico de faturamento (`total_charges` zerado) e o modelo não extrapola perfeitamente sem este input. O modelo não tratará bem dados omissos para clientes recém contratados.
