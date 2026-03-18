"""FastAPI 애플리케이션."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.auth_jwt import router as auth_router
from src.api.cache import close_redis
from src.api.dependencies import model_service
from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import health, model_info, predict, segment
from src.config import RATE_LIMIT
from src.monitoring.logging_config import setup_logging
from src.monitoring.metrics import MODEL_LOADED

# 구조화된 로깅 설정
setup_logging(
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
    json_format=os.environ.get("LOG_FORMAT", "json") == "json",
)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델 로드, 종료 시 리소스 정리."""
    from src.monitoring.logging_config import get_logger

    logger = get_logger("gameai.api")

    logger.info("model_loading")
    try:
        model_service.load()
        MODEL_LOADED.set(1)
        logger.info("model_loaded", model_type=model_service.model_type)
    except FileNotFoundError as e:
        MODEL_LOADED.set(0)
        logger.warning("model_not_found", error=str(e))

    yield

    # 종료 시 Redis 연결 정리
    await close_redis()
    logger.info("shutdown_complete")


app = FastAPI(
    title="GameAI Analytics API",
    description="게임 유저 이탈 예측 API",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate Limiter 등록
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 로깅 미들웨어
app.add_middleware(RequestLoggingMiddleware)

# Prometheus 메트릭 자동 수집
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app, endpoint="/metrics")

# 라우트 등록
app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")
app.include_router(segment.router, prefix="/api/v1")
app.include_router(model_info.router, prefix="/api/v1")

# JWT 인증 라우트
app.include_router(auth_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 핸들러."""
    from src.monitoring.logging_config import get_logger

    logger = get_logger("gameai.api")
    logger.error(
        "unhandled_exception",
        path=str(request.url.path),
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    from src.config import API_HOST, API_PORT

    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)
