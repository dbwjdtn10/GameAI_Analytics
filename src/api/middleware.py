"""API 미들웨어: 구조화된 로깅, request_id 추적."""

import time

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.logging_config import generate_request_id
from src.monitoring.metrics import PREDICTION_ERROR_COUNT

logger = structlog.get_logger("gameai.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅 미들웨어 (structlog + request_id)."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or generate_request_id()
        start = time.time()

        # structlog 컨텍스트에 request_id 바인딩
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        try:
            response = await call_next(request)
        except Exception:
            PREDICTION_ERROR_COUNT.labels(
                endpoint=request.url.path, error_type="unhandled"
            ).inc()
            raise

        duration_ms = (time.time() - start) * 1000

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 1),
            client_ip=request.client.host if request.client else "unknown",
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        return response
