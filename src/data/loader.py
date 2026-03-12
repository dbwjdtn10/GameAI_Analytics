import pandas as pd

from src.config import COOKIE_CATS_PATH, GAMING_BEHAVIOR_PATH


def load_gaming_behavior() -> pd.DataFrame:
    """Online Gaming Behavior 데이터셋 로드 + 이탈 타겟 생성."""
    df = pd.read_csv(GAMING_BEHAVIOR_PATH)

    # EngagementLevel=Low → 이탈(1), Medium/High → 유지(0)
    df["is_churned"] = (df["EngagementLevel"] == "Low").astype(int)

    return df


def load_cookie_cats() -> pd.DataFrame:
    """Cookie Cats A/B 테스트 데이터셋 로드."""
    df = pd.read_csv(COOKIE_CATS_PATH)
    return df


def get_gaming_behavior_summary(df: pd.DataFrame) -> dict:
    """Gaming Behavior 데이터셋 요약 통계."""
    return {
        "total_users": len(df),
        "churn_rate": df["is_churned"].mean(),
        "churn_count": df["is_churned"].sum(),
        "retained_count": (df["is_churned"] == 0).sum(),
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(include="object").columns),
    }
