"""FastAPI 애플리케이션."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.dependencies import model_service
from src.api.routes import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델 로드."""
    print("모델 로드 중...")
    try:
        model_service.load()
        print(f"모델 로드 완료: {model_service.model_type}")
    except FileNotFoundError as e:
        print(f"경고: {e} - 모델 없이 시작합니다.")
    yield


app = FastAPI(
    title="GameAI Analytics API",
    description="게임 유저 이탈 예측 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    from src.config import API_HOST, API_PORT

    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)
