"""모델 학습 실행 스크립트.

전체 파이프라인: 데이터 로드 → 피처 엔지니어링 → 학습 → 평가 → MLflow 기록
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # 비대화형 백엔드 (GUI 없이 파일 저장만)

import argparse

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import MODEL_DIR, RANDOM_STATE, TEST_SIZE
from src.data.loader import load_gaming_behavior
from src.features.engineer import engineer_gaming_behavior_features, get_feature_columns
from src.models.evaluator import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
    print_evaluation_report,
)
from src.models.registry import log_experiment
from src.models.trainer import (
    get_baseline_model,
    get_ensemble_stacking,
    get_ensemble_voting,
    get_lgbm_model,
    get_xgboost_model,
    tune_lgbm,
    tune_xgboost,
)


def prepare_data():
    """데이터 로드 + 피처 엔지니어링 + train/val/test 분할."""
    print("데이터 로드 중...")
    df = load_gaming_behavior()

    print("피처 엔지니어링 중...")
    df = engineer_gaming_behavior_features(df)

    # 피처/타겟 분리
    feature_cols = get_feature_columns("kaggle")
    numeric_features = feature_cols["numeric"]
    cat_features = feature_cols["categorical"]

    # 범주형 인코딩
    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    all_features = numeric_features + cat_features
    X = df[all_features]
    y = df["is_churned"]

    # Train / Val / Test 분할 (60/20/20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    tr, va, te = y_train.mean(), y_val.mean(), y_test.mean()
    print(f"  이탈률 - Train: {tr:.2%}, Val: {va:.2%}, Test: {te:.2%}")
    print(f"  피처 수: {len(all_features)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, all_features, label_encoders


def train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val,
                       X_test, y_test, feature_names, params=None):
    """단일 모델 학습 + 평가 + MLflow 기록."""
    print(f"\n{'='*60}")
    print(f"  {model_name} 학습 중...")
    print(f"{'='*60}")

    model.fit(X_train, y_train)

    # 검증 세트 평가
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    val_metrics = print_evaluation_report(y_val, y_val_pred, y_val_proba, f"{model_name} (Val)")

    # 테스트 세트 평가
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    test_metrics = print_evaluation_report(
        y_test, y_test_pred, y_test_proba, f"{model_name} (Test)"
    )

    # MLflow 기록
    all_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
    all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    model_params = params or {}
    if hasattr(model, "get_params"):
        model_params = {k: str(v) for k, v in model.get_params().items()
                        if not callable(v) and k != "estimators"}

    try:
        run_id = log_experiment(
            model=model,
            model_name=model_name,
            params=model_params,
            metrics=all_metrics,
            feature_names=feature_names,
            tags={"stage": "training"},
        )
        print(f"  MLflow run_id: {run_id}")
    except Exception as e:
        print(f"  MLflow 기록 실패 (무시): {e}")
        run_id = None

    return {
        "model": model,
        "model_name": model_name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_test_true": y_test,
        "y_test_pred": y_test_pred,
        "y_test_proba": y_test_proba,
        "run_id": run_id,
    }


def main():
    parser = argparse.ArgumentParser(description="모델 학습")
    parser.add_argument("--tune", action="store_true", help="Optuna 하이퍼파라미터 튜닝")
    parser.add_argument("--tune-trials", type=int, default=30, help="Optuna 시행 횟수")
    parser.add_argument("--no-ensemble", action="store_true", help="앙상블 모델 스킵")
    args = parser.parse_args()

    # 데이터 준비
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, _ = prepare_data()

    results = {}

    # 1. Baseline: Logistic Regression
    lr = get_baseline_model()
    results["LogisticRegression"] = train_and_evaluate(
        lr, "LogisticRegression", X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )

    # 2. XGBoost
    xgb_params = {}
    if args.tune:
        print("\n XGBoost 하이퍼파라미터 튜닝 중...")
        xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=args.tune_trials)
        print(f"  Best params: {xgb_params}")

    xgb = get_xgboost_model(**xgb_params)
    results["XGBoost"] = train_and_evaluate(
        xgb, "XGBoost", X_train, y_train, X_val, y_val, X_test, y_test, feature_names, xgb_params
    )

    # 3. LightGBM
    lgbm_params = {}
    if args.tune:
        print("\n LightGBM 하이퍼파라미터 튜닝 중...")
        lgbm_params = tune_lgbm(X_train, y_train, X_val, y_val, n_trials=args.tune_trials)
        print(f"  Best params: {lgbm_params}")

    lgbm = get_lgbm_model(**lgbm_params)
    results["LightGBM"] = train_and_evaluate(
        lgbm, "LightGBM", X_train, y_train, X_val, y_val, X_test, y_test, feature_names, lgbm_params
    )

    # 4. Ensemble (Voting)
    if not args.no_ensemble:
        voting = get_ensemble_voting(xgb_params, lgbm_params)
        results["Voting"] = train_and_evaluate(
            voting, "VotingEnsemble", X_train, y_train, X_val, y_val, X_test, y_test, feature_names
        )

        stacking = get_ensemble_stacking(xgb_params, lgbm_params)
        results["Stacking"] = train_and_evaluate(
            stacking, "StackingEnsemble", X_train, y_train,
            X_val, y_val, X_test, y_test, feature_names,
        )

    # 결과 비교
    print("\n" + "=" * 70)
    print("  모델 성능 비교 (Test Set)")
    print("=" * 70)
    comparison = []
    for name, r in results.items():
        m = r["test_metrics"]
        comparison.append({"Model": name, **m})
    comparison_df = pd.DataFrame(comparison).set_index("Model")
    print(comparison_df.round(4).to_string())

    # 최고 모델 저장
    best_model_name = comparison_df["auc_roc"].idxmax()
    best_result = results[best_model_name]
    print(f"\n최고 모델: {best_model_name} (AUC-ROC: {best_result['test_metrics']['auc_roc']:.4f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_result["model"], model_path)
    print(f"모델 저장: {model_path}")

    # 피처 이름 저장
    feature_path = MODEL_DIR / "feature_names.txt"
    feature_path.write_text("\n".join(feature_names))

    # ROC 커브 비교
    roc_data = {}
    for name, r in results.items():
        if r["y_test_proba"] is not None:
            roc_data[name] = {"y_true": r["y_test_true"], "y_pred_proba": r["y_test_proba"]}

    if roc_data:
        plot_roc_curves(roc_data, save_path=str(MODEL_DIR / "roc_comparison.png"))

    # 최고 모델 피처 중요도
    best_model = best_result["model"]
    plot_feature_importance(best_model, feature_names, model_name=best_model_name,
                            save_path=str(MODEL_DIR / "feature_importance.png"))

    # Confusion Matrix
    plot_confusion_matrix(best_result["y_test_true"], best_result["y_test_pred"],
                          model_name=best_model_name,
                          save_path=str(MODEL_DIR / "confusion_matrix.png"))

    print("\n학습 완료!")


if __name__ == "__main__":
    main()
