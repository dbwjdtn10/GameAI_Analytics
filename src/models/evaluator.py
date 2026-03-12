"""모델 평가 모듈."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(y_true, y_pred, y_pred_proba=None) -> dict:
    """모델 성능 메트릭 계산."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_pred_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["avg_precision"] = average_precision_score(y_true, y_pred_proba)

    return metrics


def print_evaluation_report(y_true, y_pred, y_pred_proba=None, model_name: str = "Model"):
    """평가 결과 출력."""
    metrics = evaluate_model(y_true, y_pred, y_pred_proba)

    print(f"\n{'=' * 50}")
    print(f"  {model_name} 평가 결과")
    print(f"{'=' * 50}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1-Score:          {metrics['f1']:.4f}")
    if "auc_roc" in metrics:
        print(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
        print(f"  Avg Precision:     {metrics['avg_precision']:.4f}")
    print(f"{'=' * 50}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['유지', '이탈'])}")

    return metrics


def plot_roc_curves(results: dict[str, dict], save_path: str = None):
    """여러 모델의 ROC 커브 비교 시각화.

    Args:
        results: {model_name: {"y_true": ..., "y_pred_proba": ...}}
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_pred_proba"])
        auc = roc_auc_score(data["y_true"], data["y_pred_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_precision_recall_curves(results: dict[str, dict], save_path: str = None):
    """Precision-Recall 커브 비교."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, data in results.items():
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_pred_proba"])
        ap = average_precision_score(data["y_true"], data["y_pred_proba"])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", save_path: str = None):
    """Confusion Matrix 시각화."""
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["유지", "이탈"], yticklabels=["유지", "이탈"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_feature_importance(model, feature_names: list[str], top_k: int = 20,
                            model_name: str = "Model", save_path: str = None):
    """피처 중요도 시각화."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print(f"  {model_name}: feature_importances_ 속성 없음")
        return None

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=True).tail(top_k)

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))
    ax.barh(feat_imp["feature"], feat_imp["importance"], color="#3498db")
    ax.set_title(f"Feature Importance - {model_name} (Top {top_k})")
    ax.set_xlabel("Importance")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
