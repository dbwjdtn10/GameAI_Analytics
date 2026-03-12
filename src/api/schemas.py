"""API 요청/응답 스키마."""

from pydantic import BaseModel, Field


class PlayerFeatures(BaseModel):
    """단일 유저 예측 요청."""

    Age: int = Field(..., ge=10, le=100, description="플레이어 나이")
    Gender: str = Field(..., description="성별 (Male/Female)")
    Location: str = Field(..., description="지역")
    GameGenre: str = Field(..., description="게임 장르")
    GameDifficulty: str = Field(..., description="게임 난이도 (Easy/Medium/Hard)")
    PlayTimeHours: float = Field(..., ge=0, description="총 플레이 시간(시간)")
    SessionsPerWeek: int = Field(..., ge=0, description="주간 세션 수")
    AvgSessionDurationMinutes: float = Field(..., ge=0, description="평균 세션 길이(분)")
    PlayerLevel: int = Field(..., ge=1, description="플레이어 레벨")
    AchievementsUnlocked: int = Field(..., ge=0, description="달성한 업적 수")
    InGamePurchases: int = Field(..., ge=0, description="인게임 구매 횟수")

    model_config = {"json_schema_extra": {
        "examples": [{
            "Age": 25,
            "Gender": "Male",
            "Location": "USA",
            "GameGenre": "RPG",
            "GameDifficulty": "Medium",
            "PlayTimeHours": 120.5,
            "SessionsPerWeek": 5,
            "AvgSessionDurationMinutes": 45.0,
            "PlayerLevel": 32,
            "AchievementsUnlocked": 15,
            "InGamePurchases": 8,
        }]
    }}


class PredictionResponse(BaseModel):
    """이탈 예측 응답."""

    player_id: str | None = Field(None, description="플레이어 ID (요청 시 제공된 경우)")
    churn_probability: float = Field(..., description="이탈 확률 (0~1)")
    churn_prediction: bool = Field(..., description="이탈 예측 (True=이탈 예상)")
    risk_level: str = Field(..., description="위험 등급 (low/medium/high/critical)")


class BatchPredictionRequest(BaseModel):
    """배치 예측 요청."""

    players: list[PlayerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """배치 예측 응답."""

    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int = Field(..., description="고위험(high+critical) 유저 수")


class HealthResponse(BaseModel):
    """헬스 체크 응답."""

    status: str
    model_loaded: bool
    model_type: str | None = None


class ErrorResponse(BaseModel):
    """에러 응답."""

    detail: str
