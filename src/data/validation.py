"""Pandera 기반 데이터 검증 스키마.

학습 데이터와 추론 데이터 양쪽에 적용하여 데이터 품질을 보장한다.
"""

from pandera import Check, Column, DataFrameSchema

# 원본 Gaming Behavior 데이터 스키마
RawGamingBehaviorSchema = DataFrameSchema(
    columns={
        "Age": Column(
            int,
            checks=[Check.in_range(10, 100)],
            description="플레이어 나이",
        ),
        "Gender": Column(
            str,
            checks=[Check.isin(["Male", "Female"])],
            description="성별",
        ),
        "Location": Column(
            str,
            checks=[Check.isin([
                "USA", "Europe", "Asia", "South America",
                "Africa", "Australia", "Other",
            ])],
            description="지역",
        ),
        "GameGenre": Column(
            str,
            checks=[Check.isin([
                "RPG", "FPS", "MOBA", "Sports", "Strategy",
                "Simulation", "Action", "Adventure",
            ])],
            description="게임 장르",
        ),
        "GameDifficulty": Column(
            str,
            checks=[Check.isin(["Easy", "Medium", "Hard"])],
            description="게임 난이도",
        ),
        "PlayTimeHours": Column(
            float,
            checks=[Check.ge(0), Check.le(50000)],
            coerce=True,
            description="총 플레이 시간(시간)",
        ),
        "SessionsPerWeek": Column(
            int,
            checks=[Check.ge(0), Check.le(100)],
            coerce=True,
            description="주간 세션 수",
        ),
        "AvgSessionDurationMinutes": Column(
            float,
            checks=[Check.ge(0), Check.le(1440)],
            coerce=True,
            description="평균 세션 시간(분)",
        ),
        "PlayerLevel": Column(
            int,
            checks=[Check.ge(1), Check.le(1000)],
            coerce=True,
            description="플레이어 레벨",
        ),
        "AchievementsUnlocked": Column(
            int,
            checks=[Check.ge(0)],
            coerce=True,
            description="달성한 업적 수",
        ),
        "InGamePurchases": Column(
            int,
            checks=[Check.ge(0)],
            coerce=True,
            description="인게임 구매 횟수",
        ),
        "EngagementLevel": Column(
            str,
            checks=[Check.isin(["Low", "Medium", "High"])],
            description="참여도 레벨",
        ),
    },
    checks=[
        Check(lambda df: len(df) > 0, error="DataFrame must not be empty"),
        Check(lambda df: df.duplicated().sum() / len(df) < 0.05,
              error="Duplicate ratio exceeds 5%"),
    ],
    coerce=True,
    strict=False,  # 추가 컬럼 허용 (PlayerID 등)
    name="RawGamingBehavior",
    description="Kaggle Gaming Behavior 원본 데이터 검증 스키마",
)

# 피처 엔지니어링 후 데이터 스키마
EngineeredFeaturesSchema = DataFrameSchema(
    columns={
        "playtime_per_session": Column(
            float, checks=[Check.ge(0)], coerce=True,
            description="세션당 플레이 시간",
        ),
        "weekly_activity_intensity": Column(
            float, checks=[Check.ge(0)], coerce=True,
            description="주간 활동 강도",
        ),
        "session_engagement_score": Column(
            float, checks=[Check.ge(0)], coerce=True,
            description="세션 몰입도",
        ),
        "level_efficiency": Column(
            float, checks=[Check.ge(0)], coerce=True,
            description="레벨 달성 효율",
        ),
        "achievement_rate": Column(
            float, checks=[Check.ge(0)], coerce=True,
            description="업적 달성률",
        ),
        "purchase_per_hour": Column(
            float, checks=[Check.ge(0)], coerce=True,
            description="시간당 구매 빈도",
        ),
        "activity_score": Column(
            float, coerce=True,
            description="종합 활동 점수",
        ),
        "is_churned": Column(
            int, checks=[Check.isin([0, 1])], coerce=True,
            description="이탈 여부 타겟",
        ),
    },
    strict=False,
    name="EngineeredFeatures",
    description="피처 엔지니어링 후 데이터 검증 스키마",
)

# API 추론 입력 스키마
InferenceInputSchema = DataFrameSchema(
    columns={
        "Age": Column(int, checks=[Check.in_range(10, 100)], coerce=True),
        "Gender": Column(str, checks=[Check.isin(["Male", "Female"])]),
        "Location": Column(str),
        "GameGenre": Column(str),
        "GameDifficulty": Column(str, checks=[Check.isin(["Easy", "Medium", "Hard"])]),
        "PlayTimeHours": Column(float, checks=[Check.ge(0)], coerce=True),
        "SessionsPerWeek": Column(int, checks=[Check.ge(0)], coerce=True),
        "AvgSessionDurationMinutes": Column(float, checks=[Check.ge(0)], coerce=True),
        "PlayerLevel": Column(int, checks=[Check.ge(1)], coerce=True),
        "AchievementsUnlocked": Column(int, checks=[Check.ge(0)], coerce=True),
        "InGamePurchases": Column(int, checks=[Check.ge(0)], coerce=True),
    },
    strict=False,
    name="InferenceInput",
    description="API 추론 입력 데이터 검증 스키마",
)


def validate_raw_data(df):
    """원본 데이터 검증."""
    return RawGamingBehaviorSchema.validate(df)


def validate_engineered_data(df):
    """피처 엔지니어링 후 데이터 검증."""
    return EngineeredFeaturesSchema.validate(df)


def validate_inference_input(df):
    """추론 입력 데이터 검증."""
    return InferenceInputSchema.validate(df)
