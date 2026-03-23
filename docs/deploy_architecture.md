# Documentação de Arquitetura de Deploy

## Decisão de Deploy: Batch vs Real-Time
A decisão técnica por trás deste projeto optou pela adoção de inferência em **Real-Time** (tempo real via endpoints REST RESTful, viabilizados com FastAPI e Uvicorn).

### Justificativas
1. **Necessidade do Negócio:**
   - A retenção de um cliente ocorre frequentemente enquanto ele interage com o aplicativo da Telco ou ligando para a central de atendimento telefônico. 
   - Um score estático "diário" (Batch) não é rápido o suficiente. No atendimento humano (call-center), os atendentes podem inserir dados das novas solicitações em tela e pedir uma avaliação de risco de churn imediata para habilitar um "botão de desconto" instantâneo.
   
2. **Infraestrutura Cloud Indicada (AWS/GCP/Azure)**:
   - A API será conteinerizada (`Docker`) e alocada num orquestrador de contêineres gerenciado sem a necessidade de instâncias pesadas de processamento em massa (como Amazon ECS com AWS Fargate ou Google Cloud Run). 
   - Estas tecnologias permitem que a arquitetura escale a 0 pods de madrugada para salvar custos e a picos absurdos nos horários comerciais de atendimento, lidando apenas com payloads Pydantic leves recebidos como JSON.

---

## Estratégia e Plano de Monitoramento Pós-Deploy

Um modelo estático apodrece ao longo do tempo se não vigiado. (Conceito de Model Drift e Data Drift). Faremos o plano usando Prometheus e MLflow Tracking:

### Métricas que serão coletadas
- **Software Metrics (DevOps):**
   - Latência p95 no endpoint de predição.
   - Taxa de chamados retornando o código HTTP 500 ou 422.
- **Model Metrics (MLOps):**
   - *Distribution Drift:* Monitorar se a distribuição do score da rede neural começou a se distanciar historicamente dos dados de treinamento originais.
   - *Recall Real:* Fazer cross-checking depois de 30 dias na base SQL para verificar quantos clientes com risco previsto BAIXO de fato foram embora (Falso Negativo real apurado no backend).

### Alertas e Playbook
- **Alerta Moderado (Slack p/ Cientistas de Dados):**
   - A proporção de classificados como classe "1" (Churn detectado) passou de 30% em uma janela de duas semanas para 50%. Causa provável: Drift nas variáveis de cobrança. (Ação: Checar os dados).
- **Alerta Grave (PagerDuty p/ DevOps):**
   - Tempo de latência médio saltando os SLA's (Ex: Demorando 3s por inferência PyTorch ao invés de < 100ms). Causa Provável: Gargalo de rede. (Ação: Escalar horizontalmente pods no Kubernetes ou GCP Run).
