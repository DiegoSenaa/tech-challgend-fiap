# ML Canvas - Previsão de Churn (Telco)

## 1. Problema de Negócio (Decisão)
A operadora de telecomunicações está perdendo clientes. O objetivo é prever quais clientes têm alto risco de cancelamento (Churn) para que a equipe de retenção possa agir proativamente (ex: oferecer descontos, melhorias no plano).

## 2. Proposição de Valor
Redução na taxa de Churn, aumentando o Life Time Value (LTV) dos clientes e reduzindo a perda de receita mensal (MRR).

## 3. Coleta de Dados
O dataset de clientes da Telco da IBM com diversas features demográficas (gênero, idosos, parceiro, dependentes), serviços contratados (telefone, linhas múltiplas, internet, segurança, backup) e métricas de conta (contrato, cobrança sem papel, método de pagamento, cobrança mensal, cobranças totais).

## 4. Métricas de Modelagem
- **Métrica Técnica Primária:** F1-Score (pois a classe de churn é desbalanceada) e ROC-AUC para avaliar a capacidade de distinguir as duas classes.
- **Outras Métricas:** Recall (para garantir que a maioria dos churners sejam detectados) e Precision.

## 5. Métrica de Negócio (ROI / Custo)
Custo do falso positivo (desconto dado a quem não ia cancelar) vs Custo do falso negativo (perda total da receita de quem cancelou).

## 6. Predições
Probabilidade de um cliente cancelar o serviço no próximo ciclo (score de 0 a 1). Se o score for maior que o threshold otimizado, classificado como Churn = Sim.
