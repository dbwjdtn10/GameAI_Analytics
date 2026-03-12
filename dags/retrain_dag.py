"""Airflow DAG: 모델 재학습 파이프라인.

주기적으로 모델을 재학습하고 드리프트를 체크하여
성능이 저하되면 자동으로 새 모델을 배포한다.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

default_args = {
    "owner": "gameai",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_data_drift(**context):
    """데이터 드리프트 탐지."""
    from sklearn.model_selection import train_test_split

    from src.config import RANDOM_STATE, TEST_SIZE
    from src.data.loader import load_gaming_behavior
    from src.features.engineer import engineer_gaming_behavior_features, get_feature_columns
    from src.monitoring.drift import detect_data_drift

    df = load_gaming_behavior()
    df = engineer_gaming_behavior_features(df)
    feature_cols = get_feature_columns("kaggle")

    # 학습 데이터 vs 최근 데이터 비교 시뮬레이션
    # 실서비스에서는 새로 수집된 데이터를 사용
    _, recent_data = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE)
    train_data, _ = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    result = detect_data_drift(
        reference=train_data,
        current=recent_data,
        numeric_features=feature_cols["numeric"],
    )

    drift_ratio = result["drift_ratio"]
    context["ti"].xcom_push(key="drift_ratio", value=drift_ratio)
    context["ti"].xcom_push(key="drifted_features", value=result["drifted_features"])

    print(f"드리프트 비율: {drift_ratio:.2%}")
    print(f"드리프트 피처: {result['drifted_features']}")

    return drift_ratio


def decide_retrain(**context):
    """드리프트 결과에 따라 재학습 여부 결정."""
    drift_ratio = context["ti"].xcom_pull(key="drift_ratio", task_ids="check_drift")

    if drift_ratio > 0.3:
        print(f"드리프트 비율 {drift_ratio:.2%} > 30% → 재학습 실행")
        return "retrain_model"
    else:
        print(f"드리프트 비율 {drift_ratio:.2%} <= 30% → 재학습 스킵")
        return "skip_retrain"


def retrain_model(**context):
    """모델 재학습."""
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from src.config import MODEL_DIR, RANDOM_STATE, TEST_SIZE
    from src.data.loader import load_gaming_behavior
    from src.features.engineer import engineer_gaming_behavior_features, get_feature_columns
    from src.models.evaluator import evaluate_model
    from src.models.registry import log_experiment
    from src.models.trainer import get_xgboost_model, tune_xgboost

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # 튜닝 + 학습
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train,
    )
    best_params = tune_xgboost(X_tr, y_tr, X_val, y_val, n_trials=20)
    model = get_xgboost_model(**best_params)
    model.fit(X_train, y_train)

    # 평가
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_proba)

    print(f"재학습 결과: AUC-ROC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")

    # 기존 모델 대비 성능 비교
    old_model_path = MODEL_DIR / "best_model.joblib"
    deploy = True
    if old_model_path.exists():
        old_model = joblib.load(old_model_path)
        old_proba = old_model.predict_proba(X_test)[:, 1]
        old_pred = old_model.predict(X_test)
        old_metrics = evaluate_model(y_test, old_pred, old_proba)

        if metrics["auc_roc"] < old_metrics["auc_roc"] - 0.01:
            new_auc = metrics["auc_roc"]
            old_auc = old_metrics["auc_roc"]
            print(f"새 모델({new_auc:.4f}) < 기존({old_auc:.4f}) → 배포 안함")
            deploy = False

    if deploy:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "best_model.joblib")
        print("새 모델 배포 완료")

    # MLflow 기록
    try:
        log_experiment(
            model=model,
            model_name="XGBoost-retrain",
            params={k: str(v) for k, v in best_params.items()},
            metrics={f"test_{k}": v for k, v in metrics.items()},
            feature_names=all_features,
            tags={"stage": "retrain", "deployed": str(deploy)},
        )
    except Exception as e:
        print(f"MLflow 기록 실패: {e}")

    context["ti"].xcom_push(key="auc_roc", value=metrics["auc_roc"])
    context["ti"].xcom_push(key="deployed", value=deploy)


def notify_result(**context):
    """재학습 결과 알림."""
    auc = context["ti"].xcom_pull(key="auc_roc", task_ids="retrain_model")
    deployed = context["ti"].xcom_pull(key="deployed", task_ids="retrain_model")
    drift = context["ti"].xcom_pull(key="drift_ratio", task_ids="check_drift")

    print("=== 재학습 파이프라인 완료 ===")
    print(f"드리프트 비율: {drift:.2%}")
    print(f"새 모델 AUC-ROC: {auc:.4f}")
    print(f"배포 여부: {'YES' if deployed else 'NO'}")


with DAG(
    dag_id="gameai_retrain_pipeline",
    default_args=default_args,
    description="게임 이탈 예측 모델 재학습 파이프라인",
    schedule="0 2 * * 1",  # 매주 월요일 02:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gameai", "ml", "retrain"],
) as dag:

    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=check_data_drift,
    )

    decide = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=decide_retrain,
    )

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    skip = EmptyOperator(task_id="skip_retrain")

    notify = PythonOperator(
        task_id="notify_result",
        python_callable=notify_result,
        trigger_rule="none_failed_min_one_success",
    )

    check_drift >> decide >> [retrain, skip]
    retrain >> notify
    skip >> notify
