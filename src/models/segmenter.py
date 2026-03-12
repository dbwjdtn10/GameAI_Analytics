"""K-Means 유저 세그먼트 분류 모듈."""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR

# 세그먼트 라벨 매핑
SEGMENT_LABELS = {
    "hardcore": {
        "description": "높은 플레이타임, 높은 레벨, 활발한 구매",
        "strategy": ["VIP 보상 프로그램", "경쟁 콘텐츠 제공", "한정판 아이템 우선 접근"],
    },
    "casual": {
        "description": "낮은 세션, 간헐적 접속",
        "strategy": ["가벼운 일일 미션", "로그인 보상 강화", "짧은 세션 콘텐츠"],
    },
    "at_risk": {
        "description": "활동 감소 추세, 낮은 활동",
        "strategy": ["복귀 보상 지급", "맞춤 알림 발송", "이벤트 쿠폰 제공"],
    },
    "new_user": {
        "description": "낮은 레벨, 짧은 플레이 기간",
        "strategy": ["튜토리얼 보상", "초보자 가이드", "멘토 매칭"],
    },
}

SEGMENT_FEATURES = [
    "PlayTimeHours",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "PlayerLevel",
    "AchievementsUnlocked",
    "InGamePurchases",
    "activity_score",
]


def train_segmenter(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """K-Means 세그먼트 모델 학습.

    Returns:
        {"model": KMeans, "scaler": StandardScaler, "cluster_map": dict}
    """
    X = df[SEGMENT_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_

    # 클러스터별 특성 분석 → 도메인 라벨 부여
    cluster_map = _assign_segment_labels(df, labels, n_clusters)

    # 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": kmeans,
        "scaler": scaler,
        "cluster_map": cluster_map,
    }, MODEL_DIR / "segmenter.joblib")

    return {"model": kmeans, "scaler": scaler, "cluster_map": cluster_map}


def _assign_segment_labels(df: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> dict:
    """클러스터 특성에 따라 도메인 라벨 자동 부여."""
    df_temp = df[SEGMENT_FEATURES].copy()
    df_temp["cluster"] = labels

    cluster_means = df_temp.groupby("cluster")[SEGMENT_FEATURES].mean()

    # 각 클러스터의 activity_score 기준으로 정렬
    sorted_clusters = cluster_means["activity_score"].sort_values(ascending=False).index.tolist()

    label_names = ["hardcore", "casual", "at_risk", "new_user"]
    # 클러스터 수에 맞게 라벨 조정
    if n_clusters > len(label_names):
        label_names.extend([f"segment_{i}" for i in range(len(label_names), n_clusters)])

    cluster_map = {}
    for i, cluster_id in enumerate(sorted_clusters):
        if i < len(label_names):
            cluster_map[int(cluster_id)] = label_names[i]
        else:
            cluster_map[int(cluster_id)] = f"segment_{i}"

    return cluster_map


def predict_segment(player_data: pd.DataFrame) -> list[dict]:
    """유저 세그먼트 예측.

    Args:
        player_data: SEGMENT_FEATURES 컬럼을 포함한 DataFrame

    Returns:
        [{"segment": str, "description": str, "strategy": list}]
    """
    seg_path = MODEL_DIR / "segmenter.joblib"
    if not seg_path.exists():
        raise FileNotFoundError("Segmenter model not found. Run train first.")

    seg_data = joblib.load(seg_path)
    kmeans = seg_data["model"]
    scaler = seg_data["scaler"]
    cluster_map = seg_data["cluster_map"]

    X = player_data[SEGMENT_FEATURES]
    X_scaled = scaler.transform(X)
    cluster_ids = kmeans.predict(X_scaled)

    results = []
    for cid in cluster_ids:
        seg_name = cluster_map.get(int(cid), "unknown")
        seg_info = SEGMENT_LABELS.get(seg_name, {
            "description": "분류 불가",
            "strategy": [],
        })
        results.append({
            "segment": seg_name,
            "description": seg_info["description"],
            "strategy": seg_info["strategy"],
        })

    return results


def get_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """전체 유저의 세그먼트 분포 요약."""
    seg_path = MODEL_DIR / "segmenter.joblib"
    if not seg_path.exists():
        raise FileNotFoundError("Segmenter model not found.")

    seg_data = joblib.load(seg_path)
    kmeans = seg_data["model"]
    scaler = seg_data["scaler"]
    cluster_map = seg_data["cluster_map"]

    X = df[SEGMENT_FEATURES]
    X_scaled = scaler.transform(X)
    df = df.copy()
    df["segment"] = [cluster_map.get(int(c), "unknown") for c in kmeans.predict(X_scaled)]

    summary = df.groupby("segment").agg(
        count=("segment", "size"),
        churn_rate=("is_churned", "mean"),
        avg_playtime=("PlayTimeHours", "mean"),
        avg_level=("PlayerLevel", "mean"),
        avg_purchases=("InGamePurchases", "mean"),
    ).round(4)

    summary["ratio"] = (summary["count"] / summary["count"].sum()).round(4)
    return summary
