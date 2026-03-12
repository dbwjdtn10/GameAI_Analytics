import pytest

from src.data.loader import load_gaming_behavior
from src.data.synthetic import generate_synthetic_data
from src.features.engineer import (
    engineer_gaming_behavior_features,
    engineer_synthetic_features,
    get_feature_columns,
)


@pytest.fixture
def gaming_df():
    return load_gaming_behavior()


@pytest.fixture
def synthetic_df():
    return generate_synthetic_data(num_users=500, days=30, seed=42)


def test_gaming_behavior_features_created(gaming_df):
    df = engineer_gaming_behavior_features(gaming_df)
    expected = [
        "playtime_per_session", "weekly_activity_intensity",
        "session_engagement_score", "level_efficiency",
        "achievement_rate", "purchase_per_hour", "activity_score",
    ]
    for col in expected:
        assert col in df.columns, f"Missing feature: {col}"


def test_gaming_behavior_features_no_nan(gaming_df):
    df = engineer_gaming_behavior_features(gaming_df)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        nan_count = df[col].isna().sum()
        assert nan_count == 0, f"{col} has {nan_count} NaN values"


def test_synthetic_features_created(synthetic_df):
    df = engineer_synthetic_features(synthetic_df)
    expected = ["spend_per_hour", "social_score", "activity_declining", "inactivity_risk"]
    for col in expected:
        assert col in df.columns, f"Missing feature: {col}"


def test_get_feature_columns_kaggle():
    cols = get_feature_columns("kaggle")
    assert "numeric" in cols
    assert "categorical" in cols
    assert len(cols["numeric"]) > 0


def test_get_feature_columns_synthetic():
    cols = get_feature_columns("synthetic")
    assert "numeric" in cols
    assert len(cols["numeric"]) > 10
