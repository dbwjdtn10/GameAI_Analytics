"""이탈 예측 API 라우트."""

import time

from fastapi import APIRouter, Depends

from src.api.cache import get_cached_prediction, set_cached_prediction
from src.api.dependencies import ModelService, get_model_service, verify_api_key
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PlayerFeatures,
    PredictionResponse,
    RiskFactor,
)
from src.monitoring.metrics import (
    BATCH_SIZE,
    CACHE_HIT_COUNT,
    CACHE_MISS_COUNT,
    HIGH_RISK_USERS_DETECTED,
    MODEL_INFERENCE_COUNT,
    PREDICTION_LATENCY,
    PREDICTION_REQUEST_COUNT,
)

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/single", response_model=PredictionResponse)
async def predict_single(
    player: PlayerFeatures,
    _api_key: str = Depends(verify_api_key),
    model: ModelService = Depends(get_model_service),
):
    """단일 유저 이탈 예측."""
    # 캐시 조회
    player_dict = player.model_dump()
    cached = await get_cached_prediction(player_dict)
    if cached is not None:
        CACHE_HIT_COUNT.inc()
        PREDICTION_REQUEST_COUNT.labels(
            endpoint="single", risk_level=cached["risk_level"]
        ).inc()
        return PredictionResponse(**cached)

    CACHE_MISS_COUNT.inc()

    # 모델 추론
    start = time.time()
    result = model.predict(player)
    latency = time.time() - start

    # 메트릭 기록
    PREDICTION_LATENCY.labels(endpoint="single").observe(latency)
    PREDICTION_REQUEST_COUNT.labels(
        endpoint="single", risk_level=result["risk_level"]
    ).inc()
    MODEL_INFERENCE_COUNT.labels(model_type=model.model_type or "unknown").inc()

    if result["risk_level"] in ("high", "critical"):
        HIGH_RISK_USERS_DETECTED.labels(risk_level=result["risk_level"]).inc()

    response = PredictionResponse(
        churn_probability=round(result["proba"], 4),
        churn_prediction=result["prediction"],
        risk_level=result["risk_level"],
        top_risk_factors=[RiskFactor(**f) for f in result["risk_factors"]],
        recommended_actions=result["actions"],
    )

    # 캐시 저장
    await set_cached_prediction(player_dict, response.model_dump())

    return response


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    _api_key: str = Depends(verify_api_key),
    model: ModelService = Depends(get_model_service),
):
    """배치 유저 이탈 예측 (최대 1000명)."""
    BATCH_SIZE.observe(len(request.players))

    start = time.time()
    results = model.predict_batch(request.players)
    latency = time.time() - start

    PREDICTION_LATENCY.labels(endpoint="batch").observe(latency)

    predictions = []
    high_risk = 0
    for i, r in enumerate(results):
        PREDICTION_REQUEST_COUNT.labels(
            endpoint="batch", risk_level=r["risk_level"]
        ).inc()
        MODEL_INFERENCE_COUNT.labels(model_type=model.model_type or "unknown").inc()

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
            HIGH_RISK_USERS_DETECTED.labels(risk_level=r["risk_level"]).inc()

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high_risk,
    )
