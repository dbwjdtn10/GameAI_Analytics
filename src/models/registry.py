"""MLflow 모델 레지스트리 모듈."""

from pathlib import Path

import mlflow.sklearn

import mlflow
from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def init_mlflow():
    """MLflow 초기화."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_experiment(
    model,
    model_name: str,
    params: dict,
    metrics: dict,
    feature_names: list[str] = None,
    artifacts: dict[str, str] = None,
    tags: dict[str, str] = None,
) -> str:
    """MLflow에 실험 결과 로깅.

    Returns:
        run_id
    """
    init_mlflow()

    with mlflow.start_run(run_name=model_name) as run:
        # 파라미터
        mlflow.log_params(params)

        # 메트릭
        mlflow.log_metrics(metrics)

        # 태그
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("model_type", model_name)

        # 모델 저장
        mlflow.sklearn.log_model(model, "model")

        # 피처 이름 로깅
        if feature_names:
            mlflow.log_text("\n".join(feature_names), "feature_names.txt")

        # 추가 아티팩트 (차트 이미지 등)
        if artifacts:
            for name, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, name)

        return run.info.run_id


def get_best_run(metric_name: str = "auc_roc") -> dict:
    """최고 성능 run 조회."""
    init_mlflow()

    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1,
    )

    if runs.empty:
        return None

    best = runs.iloc[0]
    return {
        "run_id": best["run_id"],
        "model_type": best.get("tags.model_type", "unknown"),
        metric_name: best.get(f"metrics.{metric_name}"),
    }


def load_model(run_id: str):
    """MLflow에서 모델 로드."""
    init_mlflow()
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
