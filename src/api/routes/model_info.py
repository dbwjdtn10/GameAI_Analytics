"""모델 정보 & 피처 중요도 API 라우트."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import ModelService, get_model_service, verify_api_key
from src.config import MODEL_DIR

router = APIRouter(prefix="/model", tags=["model"])


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    features: list[FeatureImportance]
    model_type: str


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_count: int
    feature_names: list[str]


@router.get("/info", response_model=ModelInfoResponse)
async def model_info(
    _api_key: str = Depends(verify_api_key),
    model: ModelService = Depends(get_model_service),
):
    """현재 배포된 모델 정보."""
    feature_path = MODEL_DIR / "feature_names.txt"
    features = []
    if feature_path.exists():
        features = feature_path.read_text().strip().split("\n")

    return ModelInfoResponse(
        model_type=model.model_type or "unknown",
        feature_count=len(features),
        feature_names=features,
    )


@router.get("/features/importance", response_model=FeatureImportanceResponse)
async def feature_importance(
    _api_key: str = Depends(verify_api_key),
    model: ModelService = Depends(get_model_service),
):
    """현재 모델의 피처 중요도."""
    inner = model._model
    if not hasattr(inner, "feature_importances_"):
        raise HTTPException(status_code=400, detail="Model does not support feature importance")

    feature_path = MODEL_DIR / "feature_names.txt"
    if not feature_path.exists():
        raise HTTPException(status_code=404, detail="Feature names not found")

    names = feature_path.read_text().strip().split("\n")
    importances = inner.feature_importances_

    features = sorted(
        [FeatureImportance(feature=n, importance=round(float(v), 6))
         for n, v in zip(names, importances)],
        key=lambda x: x.importance,
        reverse=True,
    )

    return FeatureImportanceResponse(
        features=features,
        model_type=model.model_type or "unknown",
    )
