FROM python:3.12-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[dev]"

# 소스 복사
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/
COPY data/ data/

# API 서버 실행
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
