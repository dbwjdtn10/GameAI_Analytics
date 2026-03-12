"""모델/데이터 드리프트 탐지 모듈."""

import numpy as np
import pandas as pd
from scipy import stats


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    threshold: float = 0.05,
) -> dict:
    """KS 검정으로 수치형 피처 데이터 드리프트 탐지.

    Args:
        reference: 학습 데이터 (기준)
        current: 새로운 데이터
        numeric_features: 수치형 피처 목록
        threshold: p-value 임계치

    Returns:
        {"drifted_features": [...], "details": {feat: {statistic, p_value, drifted}}}
    """
    details = {}
    drifted = []

    for feat in numeric_features:
        if feat not in reference.columns or feat not in current.columns:
            continue

        ref_vals = reference[feat].dropna()
        cur_vals = current[feat].dropna()

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)

        is_drifted = bool(p_value < threshold)
        details[feat] = {
            "statistic": float(ks_stat),
            "p_value": float(p_value),
            "drifted": is_drifted,
        }
        if is_drifted:
            drifted.append(feat)

    return {
        "drifted_features": drifted,
        "total_features": len(details),
        "drift_ratio": len(drifted) / max(len(details), 1),
        "details": details,
    }


def detect_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """예측 분포 드리프트 탐지 (KS 검정).

    Args:
        reference_predictions: 기존 예측 확률
        current_predictions: 새 예측 확률

    Returns:
        {"drifted": bool, "statistic": float, "p_value": float,
         "ref_mean": float, "cur_mean": float}
    """
    ks_stat, p_value = stats.ks_2samp(reference_predictions, current_predictions)

    return {
        "drifted": bool(p_value < threshold),
        "statistic": float(ks_stat),
        "p_value": float(p_value),
        "ref_mean": float(np.mean(reference_predictions)),
        "cur_mean": float(np.mean(current_predictions)),
        "ref_std": float(np.std(reference_predictions)),
        "cur_std": float(np.std(current_predictions)),
    }


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index (PSI) 계산.

    PSI < 0.1: 변화 없음
    0.1 <= PSI < 0.2: 약간 변화
    PSI >= 0.2: 유의미한 변화

    Args:
        reference: 기준 분포
        current: 현재 분포
        n_bins: 히스토그램 빈 수

    Returns:
        PSI 값
    """
    eps = 1e-4

    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    # 0 방지
    ref_counts = np.maximum(ref_counts, eps)
    cur_counts = np.maximum(cur_counts, eps)

    psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
    return float(psi)
