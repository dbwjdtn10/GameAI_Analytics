"""피처 선택 모듈."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def calculate_correlation_matrix(df: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    """높은 상관관계를 가진 피처 쌍에서 하나를 제거할 후보 반환."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    for col in upper.columns:
        if any(upper[col] > threshold):
            to_drop.append(col)

    return to_drop


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """VIF(Variance Inflation Factor) 계산."""
    from sklearn.linear_model import LinearRegression

    vif_data = []
    for i, col in enumerate(df.columns):
        X = df.drop(columns=[col]).values
        y = df[col].values

        if np.std(y) == 0:
            vif_data.append({"feature": col, "VIF": float("inf")})
            continue

        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)
        vif = 1 / (1 - r_squared) if r_squared < 1 else float("inf")
        vif_data.append({"feature": col, "VIF": vif})

    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def select_by_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 15,
) -> list[str]:
    """Mutual Information 기반 상위 K개 피처 선택."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False)

    return mi_df.head(top_k)["feature"].tolist()
