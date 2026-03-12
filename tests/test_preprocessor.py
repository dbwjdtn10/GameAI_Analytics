import pandas as pd
import pytest

from src.config import GAMING_BEHAVIOR_PATH
from src.data.loader import load_gaming_behavior
from src.data.preprocessor import OutlierClipper, preprocess_gaming_behavior


@pytest.fixture
def gaming_df():
    if not GAMING_BEHAVIOR_PATH.exists():
        pytest.skip("Data file not found")
    return load_gaming_behavior()


def test_load_gaming_behavior_has_target(gaming_df):
    assert "is_churned" in gaming_df.columns
    assert set(gaming_df["is_churned"].unique()) == {0, 1}


def test_preprocess_returns_correct_shape(gaming_df):
    X, y = preprocess_gaming_behavior(gaming_df)
    assert len(X) == len(y)
    assert "is_churned" not in X.columns
    assert "PlayerID" not in X.columns
    assert "EngagementLevel" not in X.columns


def test_preprocess_no_object_columns(gaming_df):
    X, y = preprocess_gaming_behavior(gaming_df)
    object_cols = X.select_dtypes(include="object").columns.tolist()
    assert len(object_cols) == 0, f"Object columns remain: {object_cols}"


def test_outlier_clipper():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100], "b": [10, 20, 30, 40, 500]})
    clipper = OutlierClipper(factor=1.5)
    clipped = clipper.fit_transform(df)
    assert clipped["a"].max() < 100
    assert clipped["b"].max() < 500
