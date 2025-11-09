# -----------------------------------------------------------------------------
# Базовый образ
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

# -----------------------------------------------------------------------------
# Метаданные
# -----------------------------------------------------------------------------
LABEL maintainer="predictlab-e"
LABEL project="EcoPredict"
LABEL description="Analytics dashboard for prediction markets with Streamlit and advanced models"
LABEL version="1.0.0"

# -----------------------------------------------------------------------------
# Переменные окружения
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PORT=8501 \
    PATH="/home/eco/.local/bin:$PATH"

WORKDIR $APP_HOME

# -----------------------------------------------------------------------------
# Системные зависимости
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    wget \
    git \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Копируем requirements
# -----------------------------------------------------------------------------
COPY requirements.txt .

# -----------------------------------------------------------------------------
# Установка зависимостей
# -----------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install streamlit plotly pandas numpy \
    && pip install gunicorn uvicorn fastapi \
    && pip install scikit-learn xgboost lightgbm catboost \
    && pip install pymc arviz statsmodels seaborn matplotlib \
    && pip install loguru structlog diskcache cachetools \
    && pip install networkx sympy tqdm

# -----------------------------------------------------------------------------
# Копируем исходники
# -----------------------------------------------------------------------------
COPY Single.py app.py model.py utils.py ./

# -----------------------------------------------------------------------------
# Оптимизация: создаём непривилегированного пользователя
# -----------------------------------------------------------------------------
RUN useradd -m eco && chown -R eco $APP_HOME
USER eco

# -----------------------------------------------------------------------------
# Кэширование и тома
# -----------------------------------------------------------------------------
VOLUME ["/app/cache"]

# -----------------------------------------------------------------------------
# Экспонируем порт
# -----------------------------------------------------------------------------
EXPOSE $PORT

# -----------------------------------------------------------------------------
# Healthcheck
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:$PORT/_stcore/health || exit 1

# -----------------------------------------------------------------------------
# Логирование
# -----------------------------------------------------------------------------
ENV STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false \
    STREAMLIT_LOG_LEVEL=info

# -----------------------------------------------------------------------------
# Команда запуска
# -----------------------------------------------------------------------------
# Используем Streamlit напрямую, но можно переключить на gunicorn/uvicorn
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# -----------------------------------------------------------------------------
# Дополнительные стадии (опционально)
# -----------------------------------------------------------------------------
FROM base AS dev
RUN pip install black flake8 mypy pytest pytest-cov
CMD ["streamlit", "run", "app.py"]

FROM base AS prod
ENV STREAMLIT_SERVER_HEADLESS=true
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
