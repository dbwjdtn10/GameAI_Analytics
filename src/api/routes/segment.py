"""유저 세그먼트 API 라우트."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import verify_api_key
from src.api.schemas import PlayerFeatures
from src.features.engineer import engineer_gaming_behavior_features
from src.models.segmenter import predict_segment

router = APIRouter(prefix="/segment", tags=["segment"])


class SegmentResponse(BaseModel):
    """세그먼트 분류 응답."""

    segment: str = Field(..., description="세그먼트 라벨")
    description: str = Field(..., description="세그먼트 설명")
    strategy: list[str] = Field(..., description="추천 리텐션 전략")


@router.post("/classify", response_model=SegmentResponse)
async def classify_segment(
    player: PlayerFeatures,
    _api_key: str = Depends(verify_api_key),
):
    """유저 세그먼트 분류 + 리텐션 전략 제안."""
    import pandas as pd

    raw = pd.DataFrame([player.model_dump()])
    raw = engineer_gaming_behavior_features(raw)

    try:
        results = predict_segment(raw)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Segmenter model not loaded")

    return SegmentResponse(**results[0])
