"""합성 게임 로그 데이터 생성기.

실제 게임 로그 패턴을 모방한 synthetic data를 생성한다.
유저 타입별 행동 프로파일을 기반으로 시계열 데이터를 생성하며,
파라미터로 데이터 볼륨, 이탈률, 시즌 이벤트 효과 등을 조절할 수 있다.
"""

import numpy as np
import pandas as pd

# 유저 타입별 행동 프로파일
USER_PROFILES = {
    "hardcore": {
        "weight": 0.15,
        "daily_playtime_mean": 180,  # 분
        "daily_playtime_std": 40,
        "sessions_per_day_mean": 4,
        "purchase_prob": 0.3,
        "avg_purchase_amount": 15000,
        "friend_count_range": (10, 50),
        "guild_activity_rate": 0.8,
        "churn_base_prob": 0.05,
        "level_gain_per_day": 1.5,
    },
    "casual": {
        "weight": 0.45,
        "daily_playtime_mean": 45,
        "daily_playtime_std": 20,
        "sessions_per_day_mean": 1.5,
        "purchase_prob": 0.05,
        "avg_purchase_amount": 5000,
        "friend_count_range": (2, 15),
        "guild_activity_rate": 0.3,
        "churn_base_prob": 0.15,
        "level_gain_per_day": 0.4,
    },
    "at_risk": {
        "weight": 0.25,
        "daily_playtime_mean": 30,
        "daily_playtime_std": 25,
        "sessions_per_day_mean": 1.0,
        "purchase_prob": 0.02,
        "avg_purchase_amount": 3000,
        "friend_count_range": (0, 8),
        "guild_activity_rate": 0.1,
        "churn_base_prob": 0.60,
        "level_gain_per_day": 0.2,
    },
    "returning": {
        "weight": 0.15,
        "daily_playtime_mean": 60,
        "daily_playtime_std": 30,
        "sessions_per_day_mean": 2.0,
        "purchase_prob": 0.10,
        "avg_purchase_amount": 8000,
        "friend_count_range": (3, 20),
        "guild_activity_rate": 0.4,
        "churn_base_prob": 0.30,
        "level_gain_per_day": 0.6,
    },
}


def _assign_user_types(num_users: int, rng: np.random.Generator) -> np.ndarray:
    """유저 타입 할당."""
    types = list(USER_PROFILES.keys())
    weights = [USER_PROFILES[t]["weight"] for t in types]
    return rng.choice(types, size=num_users, p=weights)


def _generate_playtime_series(
    profile: dict,
    days: int,
    is_churned: bool,
    churn_day: int,
    rng: np.random.Generator,
    event_days: set[int],
    event_effect: float,
) -> np.ndarray:
    """일별 플레이 시간 시계열 생성."""
    playtime = np.zeros(days)
    base_mean = profile["daily_playtime_mean"]

    for d in range(days):
        if is_churned and d >= churn_day:
            # 이탈 후 급감 → 0에 수렴
            decay = np.exp(-0.3 * (d - churn_day))
            mean = base_mean * decay
        else:
            mean = base_mean

        # 이벤트 효과
        if d in event_days:
            mean *= 1 + event_effect

        # 주말 효과 (토/일 플레이 증가)
        if d % 7 in (5, 6):
            mean *= 1.3

        playtime[d] = max(0, rng.normal(mean, profile["daily_playtime_std"]))

    return playtime


def generate_synthetic_data(
    num_users: int = 10000,
    days: int = 90,
    churn_rate: float = 0.25,
    event_effect: float = 0.15,
    event_frequency: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    """합성 게임 유저 데이터 생성.

    Args:
        num_users: 생성할 유저 수
        days: 관찰 기간 (일)
        churn_rate: 전체 이탈률
        event_effect: 이벤트 시 플레이 시간 증가율
        event_frequency: 이벤트 주기 (일)
        seed: 랜덤 시드

    Returns:
        유저별 집계 피처 DataFrame
    """
    rng = np.random.default_rng(seed)

    # 이벤트 일자
    event_days = set(range(0, days, event_frequency))

    # 유저 타입 할당
    user_types = _assign_user_types(num_users, rng)

    records = []
    for i in range(num_users):
        user_type = user_types[i]
        profile = USER_PROFILES[user_type]

        # 이탈 여부 결정
        is_churned = rng.random() < profile["churn_base_prob"]
        churn_day = rng.integers(days // 3, days) if is_churned else days

        # 플레이 시간 시계열
        playtime = _generate_playtime_series(
            profile, days, is_churned, churn_day, rng, event_days, event_effect
        )

        # 세션 수 시계열
        sessions = np.maximum(0, rng.poisson(profile["sessions_per_day_mean"], size=days))
        if is_churned:
            for d in range(churn_day, days):
                sessions[d] = max(0, int(sessions[d] * np.exp(-0.3 * (d - churn_day))))

        # 로그인 여부
        login_days = playtime > 0

        # 집계 피처 계산
        last_7d = slice(-7, None)
        record = {
            "user_id": f"user_{i:06d}",
            "user_type": user_type,
            "is_churned": int(is_churned),
            # 활동 패턴
            "avg_daily_playtime": playtime.mean(),
            "playtime_trend_7d": np.polyfit(range(7), playtime[last_7d], 1)[0]
            if playtime[last_7d].sum() > 0
            else 0,
            "login_streak": _calc_login_streak(login_days),
            "days_since_last_login": _calc_days_since_last(login_days),
            "total_playtime": playtime.sum(),
            # 세션
            "avg_sessions_per_day": sessions.mean(),
            "session_count_7d": sessions[last_7d].sum(),
            "avg_session_length": playtime.sum() / max(sessions.sum(), 1),
            "peak_hour_ratio": rng.uniform(0.3, 0.8),
            # 소비 패턴
            "purchase_count": int(rng.binomial(days, profile["purchase_prob"])),
            "total_purchase_amount": 0,  # 아래서 계산
            "days_since_last_purchase": rng.integers(1, days),
            # 소셜
            "friend_count": rng.integers(*profile["friend_count_range"]),
            "guild_activity_rate": max(0, rng.normal(profile["guild_activity_rate"], 0.15)),
            "chat_messages_per_session": max(0, rng.poisson(5 if user_type == "hardcore" else 2)),
            # 진행도
            "player_level": max(1, int(profile["level_gain_per_day"] * days + rng.normal(0, 5))),
            "achievement_rate": min(1.0, max(0, rng.beta(3, 5))),
            "content_completion_rate": min(1.0, max(0, rng.beta(2, 4))),
        }

        # 구매 금액 계산
        if record["purchase_count"] > 0:
            amounts = rng.exponential(profile["avg_purchase_amount"], size=record["purchase_count"])
            record["total_purchase_amount"] = int(amounts.sum())
            record["avg_purchase_amount"] = int(amounts.mean())
        else:
            record["avg_purchase_amount"] = 0
            record["days_since_last_purchase"] = days  # 구매 이력 없음

        records.append(record)

    return pd.DataFrame(records)


def _calc_login_streak(login_days: np.ndarray) -> int:
    """마지막 날부터 연속 로그인 일수 계산."""
    streak = 0
    for logged_in in reversed(login_days):
        if logged_in:
            streak += 1
        else:
            break
    return streak


def _calc_days_since_last(login_days: np.ndarray) -> int:
    """마지막 로그인 이후 경과일 계산."""
    for i, logged_in in enumerate(reversed(login_days)):
        if logged_in:
            return i
    return len(login_days)
