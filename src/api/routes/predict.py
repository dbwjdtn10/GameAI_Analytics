"""이탈 예측 API 라우트."""

from fastapi import APIRouter, Depends

from src.api.dependencies import ModelService, get_model_service, verify_api_key
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PlayerFeatures,
    PredictionResponse,
    RiskFactor,
)

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/single", response_model=PredictionResponse)
async def predict_single(
    player: PlayerFeatures,
    _api_key: str = Depends(verify_api_key),
    model: ModelService = Depends(get_model_service),
):
    """단일 유저 이탈 예측."""
    result = model.predict(player)
    return PredictionResponse(
        churn_probability=round(result["proba"], 4),
        churn_prediction=result["prediction"],
        risk_level=result["risk_level"],
        top_risk_factors=[RiskFactor(**f) for f in result["risk_factors"]],
        recommended_actions=result["actions"],
    )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    _api_key: str = Depends(verify_api_key),
    model: ModelService = Depends(get_model_service),
):
    """배치 유저 이탈 예측 (최대 1000명)."""
    results = model.predict_batch(request.players)

    predictions = []
    high_risk = 0
    for i, r in enumerate(results):
        predictions.append(PredictionResponse(
            player_id=str(i),
            churn_probability=round(r["proba"], 4),
            churn_prediction=r["prediction"],
            risk_level=r["risk_level"],
            top_risk_factors=[RiskFactor(**f) for f in r["risk_factors"]],
            recommended_actions=r["actions"],
        ))
        if r["risk_level"] in ("high", "critical"):
            high_risk += 1

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high_risk,
    )
