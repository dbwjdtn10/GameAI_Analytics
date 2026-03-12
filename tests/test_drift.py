"""드리프트 탐지 테스트."""

import numpy as np
import pandas as pd

from src.monitoring.drift import calculate_psi, detect_data_drift, detect_prediction_drift


def test_no_drift_same_distribution():
    rng = np.random.RandomState(42)
    ref = pd.DataFrame({"feat1": rng.normal(0, 1, 1000), "feat2": rng.uniform(0, 1, 1000)})
    cur = pd.DataFrame({"feat1": rng.normal(0, 1, 1000), "feat2": rng.uniform(0, 1, 1000)})

    result = detect_data_drift(ref, cur, ["feat1", "feat2"])
    assert len(result["drifted_features"]) == 0


def test_drift_detected():
    rng = np.random.RandomState(42)
    ref = pd.DataFrame({"feat1": rng.normal(0, 1, 1000)})
    cur = pd.DataFrame({"feat1": rng.normal(5, 1, 1000)})  # 평균 크게 이동

    result = detect_data_drift(ref, cur, ["feat1"])
    assert "feat1" in result["drifted_features"]
    assert result["details"]["feat1"]["drifted"] is True


def test_prediction_drift():
    rng = np.random.RandomState(42)
    ref_pred = rng.beta(2, 5, 1000)
    cur_pred = rng.beta(5, 2, 1000)  # 분포 역전

    result = detect_prediction_drift(ref_pred, cur_pred)
    assert result["drifted"] is True


def test_psi_no_change():
    rng = np.random.RandomState(42)
    data = rng.normal(0, 1, 5000)
    psi = calculate_psi(data, data)
    assert psi < 0.1


def test_psi_significant_change():
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)
    psi = calculate_psi(ref, cur)
    assert psi >= 0.2
