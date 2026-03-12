"""모델 학습/평가 테스트."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.evaluator import evaluate_model
from src.models.trainer import (
    get_baseline_model,
    get_ensemble_voting,
    get_lgbm_model,
    get_xgboost_model,
)


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    return X[:400], X[400:], y[:400], y[400:]


def test_baseline_model(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = get_baseline_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)


def test_xgboost_model(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = get_xgboost_model(n_estimators=10)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_lgbm_model(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = get_lgbm_model(n_estimators=10)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)


def test_ensemble_voting(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = get_ensemble_voting()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)


def test_evaluate_model():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.4, 0.3, 0.8])

    metrics = evaluate_model(y_true, y_pred, y_proba)
    assert "accuracy" in metrics
    assert "auc_roc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["auc_roc"] <= 1
