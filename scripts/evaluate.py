"""SHAP 분석 + 상세 평가 스크립트.

학습된 best_model을 로드하여 SHAP 분석을 수행한다.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import RANDOM_STATE, TEST_SIZE, MODEL_DIR
from src.data.loader import load_gaming_behavior
from src.features.engineer import engineer_gaming_behavior_features, get_feature_columns


def prepare_test_data():
    """학습 시와 동일한 전처리로 테스트 데이터 준비."""
    df = load_gaming_behavior()
    df = engineer_gaming_behavior_features(df)

    feature_cols = get_feature_columns("kaggle")
    numeric_features = feature_cols["numeric"]
    cat_features = feature_cols["categorical"]

    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    all_features = numeric_features + cat_features
    X = df[all_features]
    y = df["is_churned"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_test, y_test, all_features


def main():
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        print(f"모델 파일이 없습니다: {model_path}")
        print("먼저 python scripts/train.py 를 실행하세요.")
        return

    print("모델 로드 중...")
    model = joblib.load(model_path)
    print(f"  모델 타입: {type(model).__name__}")

    X_test, y_test, feature_names = prepare_test_data()
    print(f"  테스트 데이터: {len(X_test):,}행")

    # SHAP 분석
    print("\nSHAP 분석 중...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        print("  TreeExplainer 실패, KernelExplainer로 대체...")
        sample = X_test.sample(n=min(200, len(X_test)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict_proba, sample)
        shap_values = explainer.shap_values(sample)
        X_test = sample

    # SHAP Summary Plot
    print("  SHAP Summary Plot 생성...")
    fig, ax = plt.subplots(figsize=(12, 8))
    if isinstance(shap_values, list):
        # 이진 분류의 경우 클래스 1(이탈)의 SHAP values 사용
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

    save_path = MODEL_DIR / "shap_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  저장: {save_path}")

    # SHAP Bar Plot (평균 절대 SHAP 값)
    print("  SHAP Bar Plot 생성...")
    fig, ax = plt.subplots(figsize=(10, 8))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names,
                          plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          plot_type="bar", show=False)

    save_path = MODEL_DIR / "shap_bar.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  저장: {save_path}")

    # 개별 유저 SHAP Waterfall (고위험 유저 예시)
    print("  고위험 유저 SHAP Waterfall 생성...")
    y_proba = model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.argmax(y_proba)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    explanation = shap.Explanation(
        values=sv[high_risk_idx],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                    else explainer.expected_value,
        data=X_test.iloc[high_risk_idx].values,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    save_path = MODEL_DIR / "shap_waterfall_high_risk.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  저장: {save_path}")
    print(f"  해당 유저 이탈 확률: {y_proba[high_risk_idx]:.2%}")

    print("\nSHAP 분석 완료!")


if __name__ == "__main__":
    main()
