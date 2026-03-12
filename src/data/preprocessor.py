import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


class TargetEncoder(BaseEstimator, TransformerMixin):
    """범주형 변수를 타겟 평균으로 인코딩."""

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.encoding_map_: dict[str, dict] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        self.global_mean_ = y.mean()
        for col in X.columns:
            stats = y.groupby(X[col]).agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (
                stats["count"] + self.smoothing
            )
            self.encoding_map_[col] = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()
        for col in X.columns:
            X_encoded[col] = X[col].map(self.encoding_map_.get(col, {})).fillna(self.global_mean_)
        return X_encoded


class OutlierClipper(BaseEstimator, TransformerMixin):
    """IQR 기반 이상치 클리핑."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierClipper":
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clipped = X.copy()
        for col in X.columns:
            lower, upper = self.bounds_[col]
            X_clipped[col] = X_clipped[col].clip(lower, upper)
        return X_clipped


def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """전처리 파이프라인 구성."""
    numeric_pipeline = Pipeline([
        ("outlier_clip", OutlierClipper()),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", "passthrough", categorical_features),
        ],
        remainder="drop",
    )


def preprocess_gaming_behavior(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Gaming Behavior 데이터 전처리.

    Returns:
        (features_df, target_series)
    """
    df = df.copy()

    # 불필요한 컬럼 제거
    drop_cols = ["PlayerID", "EngagementLevel"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 타겟 분리
    target = df.pop("is_churned")

    # 범주형 라벨 인코딩
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, target
