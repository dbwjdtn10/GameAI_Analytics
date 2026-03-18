"""헬스 체크 라우트 (확장)."""

import time

from fastapi import APIRouter

from src.api.dependencies import model_service
from src.api.schemas import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인 (모델 + Redis 상태 포함)."""
    components = {}

    # 모델 상태
    components["model"] = ComponentHealth(
        status="healthy" if model_service.is_loaded else "unavailable",
    )

    # Redis 상태
    try:
        from src.api.cache import get_redis

        start = time.time()
        client = await get_redis()
        if client is not None:
            await client.ping()
            latency = (time.time() - start) * 1000
            components["redis"] = ComponentHealth(status="healthy", latency_ms=round(latency, 1))
        else:
            components["redis"] = ComponentHealth(status="unavailable")
    except Exception:
        components["redis"] = ComponentHealth(status="unhealthy")

    overall = "ok" if model_service.is_loaded else "degraded"

    return HealthResponse(
        status=overall,
        model_loaded=model_service.is_loaded,
        model_type=model_service.model_type,
        components=components,
    )
