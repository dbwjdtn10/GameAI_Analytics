"""헬스 체크 라우트."""

from fastapi import APIRouter

from src.api.dependencies import model_service
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인."""
    return HealthResponse(
        status="ok" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        model_type=model_service.model_type,
    )
