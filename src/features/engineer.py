"""피처 엔지니어링 모듈.

Kaggle 데이터와 합성 데이터 각각에 맞는 피처 생성 로직을 제공한다.
"""

import numpy as np
import pandas as pd


def engineer_gaming_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gaming Behavior 데이터셋에서 파생 피처 생성.

    원본 컬럼: PlayTimeHours, SessionsPerWeek, AvgSessionDurationMinutes,
              PlayerLevel, AchievementsUnlocked, InGamePurchases,
              Age, Gender, Location, GameGenre, GameDifficulty
    """
    df = df.copy()

    # 세션당 플레이 시간 (시간)
    df["playtime_per_session"] = np.where(
        df["SessionsPerWeek"] > 0,
        df["PlayTimeHours"] / df["SessionsPerWeek"],
        0,
    )

    # 주간 활동 강도 (세션 수 × 평균 세션 길이)
    df["weekly_activity_intensity"] = (
        df["SessionsPerWeek"] * df["AvgSessionDurationMinutes"]
    )

    # 세션 몰입도 (평균 세션 길이 / 전체 평균)
    global_avg_session = df["AvgSessionDurationMinutes"].mean()
    df["session_engagement_score"] = df["AvgSessionDurationMinutes"] / global_avg_session

    # 플레이 시간 대비 레벨 효율
    df["level_efficiency"] = np.where(
        df["PlayTimeHours"] > 0,
        df["PlayerLevel"] / df["PlayTimeHours"],
        0,
    )

    # 레벨 대비 업적 달성률
    df["achievement_rate"] = np.where(
        df["PlayerLevel"] > 0,
        df["AchievementsUnlocked"] / df["PlayerLevel"],
        0,
    )

    # 시간당 구매 빈도
    df["purchase_per_hour"] = np.where(
        df["PlayTimeHours"] > 0,
        df["InGamePurchases"] / df["PlayTimeHours"],
        0,
    )

    # 나이 그룹
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 25, 35, 50, 100],
        labels=["teen", "young_adult", "adult", "middle", "senior"],
    )

    # 활동 점수 (종합)
    df["activity_score"] = (
        df["PlayTimeHours"] * 0.3
        + df["SessionsPerWeek"] * 0.3
        + df["PlayerLevel"] * 0.2
        + df["AchievementsUnlocked"] * 0.2
    )

    return df


def engineer_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """합성 데이터에서 추가 파생 피처 생성."""
    df = df.copy()

    # 구매 금액 대비 플레이 시간 비율 (가치 유저 판별)
    df["spend_per_hour"] = np.where(
        df["total_playtime"] > 0,
        df["total_purchase_amount"] / (df["total_playtime"] / 60),
        0,
    )

    # 소셜 활성도 종합 점수
    df["social_score"] = (
        df["friend_count"] * 0.4
        + df["guild_activity_rate"] * 100 * 0.3
        + df["chat_messages_per_session"] * 0.3
    )

    # 활동 감소 지표 (7일 트렌드가 음수이면 위험)
    df["activity_declining"] = (df["playtime_trend_7d"] < 0).astype(int)

    # 비활성 위험도 (마지막 로그인 경과일 × 로그인 스트릭 역수)
    df["inactivity_risk"] = df["days_since_last_login"] / (df["login_streak"] + 1)

    # 종합 이탈 위험 점수 (규칙 기반)
    df["rule_based_risk"] = (
        (df["days_since_last_login"] > 7).astype(float) * 0.3
        + (df["playtime_trend_7d"] < -5).astype(float) * 0.3
        + (df["purchase_count"] == 0).astype(float) * 0.2
        + (df["friend_count"] < 3).astype(float) * 0.2
    )

    return df


def get_feature_columns(dataset_type: str = "kaggle") -> dict[str, list[str]]:
    """데이터셋 타입별 피처 컬럼 목록 반환."""
    if dataset_type == "kaggle":
        return {
            "numeric": [
                "Age", "PlayTimeHours", "SessionsPerWeek",
                "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked",
                "InGamePurchases", "playtime_per_session", "weekly_activity_intensity",
                "session_engagement_score", "level_efficiency", "achievement_rate",
                "purchase_per_hour", "activity_score",
            ],
            "categorical": ["Gender", "Location", "GameGenre", "GameDifficulty"],
        }
    elif dataset_type == "synthetic":
        return {
            "numeric": [
                "avg_daily_playtime", "playtime_trend_7d", "login_streak",
                "days_since_last_login", "total_playtime", "avg_sessions_per_day",
                "session_count_7d", "avg_session_length", "peak_hour_ratio",
                "purchase_count", "total_purchase_amount", "avg_purchase_amount",
                "days_since_last_purchase", "friend_count", "guild_activity_rate",
                "chat_messages_per_session", "player_level", "achievement_rate",
                "content_completion_rate", "spend_per_hour", "social_score",
                "inactivity_risk", "rule_based_risk",
            ],
            "categorical": ["user_type"],
        }
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
