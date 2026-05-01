FROM python:3.10-slim

# Evita que o Python crie arquivos .pyc e não bufferize o stdout (melhor para logs em containers)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependências básicas do sistema e faz o upgrade do pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Copia apenas o pyproject.toml primeiro para aproveitar o cache de dependências do Docker
COPY pyproject.toml .

# Instala as dependências de produção descritas no pyproject.toml
RUN pip install --no-cache-dir .

# Copia as pastas essenciais do projeto (código fonte e os modelos treinados)
COPY src/ ./src/
COPY models/ ./models/

# Expõe a porta que o Uvicorn vai usar
EXPOSE 8000

# Executa o Uvicorn apontando para o app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
