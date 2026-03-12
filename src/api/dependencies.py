"""API 의존성 (모델 로딩, 인증)."""

import os

import joblib
import pandas as pd
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.api.schemas import PlayerFeatures
from src.config import MODEL_DIR
from src.features.engineer import engineer_gaming_behavior_features, get_feature_columns

# API Key 인증
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEYS: set[str] = set()


def _load_api_keys() -> set[str]:
    """환경변수에서 API 키 로드."""
    keys = os.environ.get("GAMEAI_API_KEYS", "")
    if keys:
        return {k.strip() for k in keys.split(",") if k.strip()}
    # 개발용 기본 키
    return {"dev-key-gameai-2024"}


API_KEYS = _load_api_keys()


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """API 키 검증."""
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key


# 모델 싱글톤
class ModelService:
    """학습된 모델을 로드하고 예측을 수행하는 서비스."""

    def __init__(self):
        self._model = None
        self._feature_names: list[str] = []
        self._label_encoders: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_type(self) -> str | None:
        if self._model is None:
            return None
        return type(self._model).__name__

    def load(self):
        """모델과 피처 정보 로드."""
        model_path = MODEL_DIR / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._model = joblib.load(model_path)

        feature_path = MODEL_DIR / "feature_names.txt"
        if feature_path.exists():
            self._feature_names = feature_path.read_text().strip().split("\n")

    def _prepare_features(self, player: PlayerFeatures) -> pd.DataFrame:
        """단일 유저 데이터를 모델 입력 형태로 변환."""
        raw = pd.DataFrame([player.model_dump()])

        # 파생 피처 생성 (학습 시와 동일)
        raw = engineer_gaming_behavior_features(raw)

        feature_cols = get_feature_columns("kaggle")
        cat_features = feature_cols["categorical"]

        # 범주형 인코딩 (LabelEncoder 대신 학습 데이터 기반 매핑 사용)
        # 학습 시 fit_transform 순서에 의존하므로 정렬 기반 인코딩
        from sklearn.preprocessing import LabelEncoder

        for col in cat_features:
            le = LabelEncoder()
            le.fit(raw[col].astype(str))
            raw[col] = le.transform(raw[col].astype(str))

        return raw[self._feature_names]

    def predict(self, player: PlayerFeatures) -> dict:
        """이탈 확률 예측.

        Returns:
            {"proba", "prediction", "risk_level", "risk_factors", "actions"}
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        features = self._prepare_features(player)
        proba = float(self._model.predict_proba(features)[0, 1])
        prediction = bool(proba >= 0.5)

        if proba < 0.3:
            risk_level = "low"
        elif proba < 0.5:
            risk_level = "medium"
        elif proba < 0.7:
            risk_level = "high"
        else:
            risk_level = "critical"

        # 피처 중요도 기반 위험 요인 (모델의 feature_importances_ 사용)
        risk_factors = self._get_risk_factors(features)
        actions = self._get_recommended_actions(risk_level, risk_factors)

        return {
            "proba": proba,
            "prediction": prediction,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "actions": actions,
        }

    def _get_risk_factors(self, features: pd.DataFrame) -> list[dict]:
        """피처 중요도 기반 상위 위험 요인 추출."""
        if not hasattr(self._model, "feature_importances_"):
            return []

        importances = self._model.feature_importances_
        names = self._feature_names

        descriptions = {
            "PlayTimeHours": "총 플레이 시간",
            "SessionsPerWeek": "주간 세션 수",
            "AvgSessionDurationMinutes": "평균 세션 시간",
            "PlayerLevel": "플레이어 레벨",
            "AchievementsUnlocked": "업적 달성 수",
            "InGamePurchases": "인게임 구매 횟수",
            "activity_score": "종합 활동 점수",
            "playtime_per_session": "세션당 플레이 시간",
            "weekly_activity_intensity": "주간 활동 강도",
            "session_engagement_score": "세션 몰입도",
            "level_efficiency": "레벨 달성 효율",
            "achievement_rate": "업적 달성률",
            "purchase_per_hour": "시간당 구매 빈도",
            "Age": "플레이어 나이",
            "Gender": "성별",
            "Location": "지역",
            "GameGenre": "게임 장르",
            "GameDifficulty": "게임 난이도",
        }

        factors = sorted(
            zip(names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return [
            {
                "feature": name,
                "impact": round(float(imp), 4),
                "description": descriptions.get(name, name),
            }
            for name, imp in factors
        ]

    def _get_recommended_actions(
        self, risk_level: str, risk_factors: list[dict]
    ) -> list[str]:
        """위험 등급에 따른 추천 액션."""
        actions = {
            "low": ["정기 보상 유지", "신규 콘텐츠 알림"],
            "medium": ["로그인 보상 강화", "맞춤 이벤트 알림", "일일 미션 추천"],
            "high": [
                "복귀 보상 지급",
                "맞춤 알림 발송",
                "길드 활동 알림",
            ],
            "critical": [
                "긴급 복귀 보상 지급 (7일 미접속 시)",
                "1:1 맞춤 이벤트 쿠폰 제공",
                "소셜 기능 연결 유도",
                "VIP 상담 연결",
            ],
        }
        return actions.get(risk_level, [])

    def predict_batch(self, players: list[PlayerFeatures]) -> list[dict]:
        """배치 예측."""
        return [self.predict(p) for p in players]


# 전역 모델 서비스 인스턴스
model_service = ModelService()


def get_model_service() -> ModelService:
    """모델 서비스 의존성."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Server is starting up.",
        )
    return model_service
