"""모델 성능 regression 테스트.

배포된 모델이 최소 성능 기준을 충족하는지 검증한다.
CI에서 모델 파일이 없으면 자동 스킵.
"""

import joblib
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import GAMING_BEHAVIOR_PATH, MODEL_DIR, RANDOM_STATE, TEST_SIZE
from src.data.loader import load_gaming_behavior
from src.features.engineer import engineer_gaming_behavior_features, get_feature_columns

MODEL_PATH = MODEL_DIR / "best_model.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not GAMING_BEHAVIOR_PATH.exists(),
    reason="Model or data file not found",
)

# 최소 성능 기준
MIN_AUC_ROC = 0.90
MIN_ACCURACY = 0.85


@pytest.fixture(scope="module")
def test_predictions():
    """테스트 셋 예측 결과 생성."""
    model = joblib.load(MODEL_PATH)

    df = load_gaming_behavior()
    df = engineer_gaming_behavior_features(df)

    feature_cols = get_feature_columns("kaggle")
    cat_features = feature_cols["categorical"]
    numeric_features = feature_cols["numeric"]

    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    all_features = numeric_features + cat_features
    X = df[all_features]
    y = df["is_churned"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return y_test, y_pred, y_proba


def test_auc_roc_above_threshold(test_predictions):
    y_test, _, y_proba = test_predictions
    auc = roc_auc_score(y_test, y_proba)
    assert auc >= MIN_AUC_ROC, f"AUC-ROC {auc:.4f} < {MIN_AUC_ROC}"


def test_accuracy_above_threshold(test_predictions):
    y_test, y_pred, _ = test_predictions
    accuracy = np.mean(y_test == y_pred)
    assert accuracy >= MIN_ACCURACY, f"Accuracy {accuracy:.4f} < {MIN_ACCURACY}"


def test_predictions_valid_range(test_predictions):
    _, _, y_proba = test_predictions
    assert np.all((y_proba >= 0) & (y_proba <= 1)), "Probabilities out of [0, 1] range"


def test_predictions_not_constant(test_predictions):
    _, y_pred, _ = test_predictions
    unique = np.unique(y_pred)
    assert len(unique) > 1, "Model predicts only one class"
