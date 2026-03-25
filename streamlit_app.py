"""GameAI Analytics - Streamlit Cloud 배포용 대시보드.

Streamlit Cloud에서 독립 실행 가능하도록 데이터 자동 다운로드 + 모델 자동 학습 기능 포함.
원본: src/dashboard/app.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ===== Configuration =====
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "raw" / "gaming_behavior" / "online_gaming_behavior_dataset.csv"

st.set_page_config(page_title="GameAI Analytics", page_icon="🎮", layout="wide")


# ===== Data & Model Loading =====
def _ensure_data():
    """데이터가 없으면 Kaggle에서 다운로드 시도, 실패 시 합성 데이터 생성."""
    if DATA_PATH.exists():
        return
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/dbwjdtn10/GameAI_Analytics/main/data/raw/gaming_behavior/online_gaming_behavior_dataset.csv"
        urllib.request.urlretrieve(url, DATA_PATH)
    except Exception:
        _generate_synthetic_kaggle_format()


def _generate_synthetic_kaggle_format():
    """Kaggle 포맷과 동일한 합성 데이터 생성 (fallback)."""
    np.random.seed(RANDOM_STATE)
    n = 5000
    genders = np.random.choice(["Male", "Female"], n)
    locations = np.random.choice(["USA", "Europe", "Asia", "Other"], n)
    genres = np.random.choice(["Action", "RPG", "Strategy", "Sports", "Simulation"], n)
    diffs = np.random.choice(["Easy", "Medium", "Hard"], n)
    ages = np.random.randint(15, 55, n)
    playtime = np.round(np.random.exponential(10, n), 1)
    sessions = np.random.randint(1, 15, n)
    avg_dur = np.round(np.random.normal(60, 25, n).clip(10, 180), 1)
    levels = np.random.randint(1, 100, n)
    achievements = np.random.randint(0, 50, n)
    purchases = np.random.randint(0, 30, n)
    engagement = np.random.choice(["Low", "Medium", "High"], n, p=[0.2, 0.45, 0.35])
    df = pd.DataFrame({
        "PlayerID": range(1, n + 1), "Age": ages, "Gender": genders,
        "Location": locations, "GameGenre": genres, "PlayTimeHours": playtime,
        "InGamePurchases": purchases, "GameDifficulty": diffs,
        "SessionsPerWeek": sessions, "AvgSessionDurationMinutes": avg_dur,
        "PlayerLevel": levels, "AchievementsUnlocked": achievements,
        "EngagementLevel": engagement,
    })
    df.to_csv(DATA_PATH, index=False)


def _engineer_features(df):
    """피처 엔지니어링."""
    df = df.copy()
    df["playtime_per_session"] = np.where(df["SessionsPerWeek"] > 0, df["PlayTimeHours"] / df["SessionsPerWeek"], 0)
    df["weekly_activity_intensity"] = df["SessionsPerWeek"] * df["AvgSessionDurationMinutes"]
    global_avg = df["AvgSessionDurationMinutes"].mean()
    df["session_engagement_score"] = df["AvgSessionDurationMinutes"] / global_avg
    df["level_efficiency"] = np.where(df["PlayTimeHours"] > 0, df["PlayerLevel"] / df["PlayTimeHours"], 0)
    df["achievement_rate"] = np.where(df["PlayerLevel"] > 0, df["AchievementsUnlocked"] / df["PlayerLevel"], 0)
    df["purchase_per_hour"] = np.where(df["PlayTimeHours"] > 0, df["InGamePurchases"] / df["PlayTimeHours"], 0)
    df["age_group"] = pd.cut(df["Age"], bins=[0, 18, 25, 35, 50, 100], labels=["teen", "young_adult", "adult", "middle", "senior"])
    df["activity_score"] = df["PlayTimeHours"] * 0.3 + df["SessionsPerWeek"] * 0.3 + df["PlayerLevel"] * 0.2 + df["AchievementsUnlocked"] * 0.2
    return df


NUMERIC_FEATURES = [
    "Age", "PlayTimeHours", "SessionsPerWeek", "AvgSessionDurationMinutes",
    "PlayerLevel", "AchievementsUnlocked", "InGamePurchases",
    "playtime_per_session", "weekly_activity_intensity", "session_engagement_score",
    "level_efficiency", "achievement_rate", "purchase_per_hour", "activity_score",
]
CAT_FEATURES = ["Gender", "Location", "GameGenre", "GameDifficulty"]


@st.cache_data
def load_data():
    _ensure_data()
    df = pd.read_csv(DATA_PATH)
    df["is_churned"] = (df["EngagementLevel"] == "Low").astype(int)
    df = _engineer_features(df)
    return df


@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "best_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    # 모델이 없으면 XGBoost를 빠르게 학습
    return _train_quick_model()


def _train_quick_model():
    """Streamlit Cloud에서 모델이 없을 때 빠르게 학습."""
    from xgboost import XGBClassifier
    df = load_data()
    df_enc = df.copy()
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    X = df_enc[NUMERIC_FEATURES + CAT_FEATURES]
    y = df_enc["is_churned"]
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE)
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "best_model.joblib")
    feature_path = MODEL_DIR / "feature_names.txt"
    feature_path.write_text("\n".join(NUMERIC_FEATURES + CAT_FEATURES))
    return model


@st.cache_data
def get_test_predictions():
    df = load_data()
    df_enc = df.copy()
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    X = df_enc[NUMERIC_FEATURES + CAT_FEATURES]
    y = df_enc["is_churned"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    model = load_model()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return y_test, y_pred, y_proba


# ===== Pages =====
def page_overview():
    st.title("🎮 GameAI Analytics Dashboard")
    st.markdown("**게임 유저 행동 분석 & 이탈 예측 시스템** — XGBoost AUC-ROC 0.9409")
    st.caption("이 대시보드는 게임 유저의 플레이 데이터를 분석하여 이탈을 예측하고, 세그먼트별 리텐션 전략을 제안합니다.")

    df = load_data()
    total = len(df)
    churned = df["is_churned"].sum()
    churn_rate = churned / total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 유저 수", f"{total:,}")
    col2.metric("이탈 유저", f"{churned:,}", delta=f"-{churn_rate:.1%}", delta_color="inverse")
    col3.metric("이탈률", f"{churn_rate:.1%}")
    col4.metric("유지 유저", f"{total - churned:,}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("이탈 분포")
        churn_counts = df["is_churned"].value_counts().reset_index()
        churn_counts.columns = ["is_churned", "count"]
        churn_counts["label"] = churn_counts["is_churned"].map({0: "유지", 1: "이탈"})
        fig = px.pie(churn_counts, values="count", names="label", color_discrete_sequence=["#3498db", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("장르별 이탈률")
        genre_churn = df.groupby("GameGenre")["is_churned"].mean().sort_values(ascending=False)
        fig = px.bar(x=genre_churn.index, y=genre_churn.values, labels={"x": "장르", "y": "이탈률"}, color=genre_churn.values, color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("주요 피처 분포 (이탈 vs 유지)")
    feat = st.selectbox("피처 선택", ["PlayTimeHours", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked", "InGamePurchases", "activity_score", "playtime_per_session"])
    fig = px.histogram(df, x=feat, color="is_churned", barmode="overlay", color_discrete_map={0: "#3498db", 1: "#e74c3c"}, labels={"is_churned": "이탈 여부"}, opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)


def page_model_performance():
    st.title("📊 모델 성능")
    model = load_model()
    st.info(f"모델: **{type(model).__name__}**")
    y_test, y_pred, y_proba = get_test_predictions()

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    metrics = {
        "AUC-ROC": roc_auc_score(y_test, y_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }
    cols = st.columns(5)
    for i, (name, val) in enumerate(metrics.items()):
        cols[i].metric(name, f"{val:.4f}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={metrics['AUC-ROC']:.4f}", line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="예측", y="실제"), x=["유지", "이탈"], y=["유지", "이탈"])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Precision-Recall Curve")
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, line=dict(color="#e74c3c", width=2)))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=350)
    st.plotly_chart(fig, use_container_width=True)


def page_feature_importance():
    st.title("🔍 Feature Importance")
    model = load_model()

    if hasattr(model, "feature_importances_"):
        feature_names = NUMERIC_FEATURES + CAT_FEATURES
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=True).tail(15)
        fig = px.bar(fi_df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
        fig.update_layout(height=500, yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("SHAP 분석")
    st.info("SHAP(SHapley Additive exPlanations) 값은 각 피처가 개별 예측에 미치는 영향을 보여줍니다. 양의 SHAP 값은 이탈 확률을 높이고, 음의 SHAP 값은 낮춥니다.")

    try:
        import shap
        df = load_data()
        df_enc = df.copy()
        for col in CAT_FEATURES:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        X = df_enc[NUMERIC_FEATURES + CAT_FEATURES]
        sample = X.sample(200, random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        import matplotlib.pyplot as plt
        fig_shap, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        st.pyplot(fig_shap, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP 분석을 로드할 수 없습니다: {e}")


def page_prediction():
    st.title("🎯 유저 이탈 예측")
    st.markdown("플레이어 정보를 입력하면 이탈 확률을 예측합니다.")

    model = load_model()

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("나이", 15, 60, 25)
        gender = st.selectbox("성별", ["Male", "Female"])
        location = st.selectbox("지역", ["USA", "Europe", "Asia", "Other"])
        genre = st.selectbox("장르", ["Action", "RPG", "Strategy", "Sports", "Simulation"])
    with c2:
        playtime = st.slider("주간 플레이 시간(h)", 0.0, 50.0, 10.0, 0.5)
        sessions = st.slider("주간 세션 수", 1, 20, 5)
        avg_dur = st.slider("평균 세션 시간(분)", 10, 180, 60)
        difficulty = st.selectbox("난이도", ["Easy", "Medium", "Hard"])
    with c3:
        level = st.slider("레벨", 1, 100, 30)
        achievements = st.slider("업적 수", 0, 50, 10)
        purchases = st.slider("인앱 구매 수", 0, 30, 3)

    if st.button("예측하기", type="primary", use_container_width=True):
        # 피처 계산
        pts = playtime / sessions if sessions > 0 else 0
        wai = sessions * avg_dur
        seg = avg_dur / 60.0
        le_ = level / playtime if playtime > 0 else 0
        ar = achievements / level if level > 0 else 0
        pph = purchases / playtime if playtime > 0 else 0
        ascore = playtime * 0.3 + sessions * 0.3 + level * 0.2 + achievements * 0.2

        input_df = pd.DataFrame([{
            "Age": age, "PlayTimeHours": playtime, "SessionsPerWeek": sessions,
            "AvgSessionDurationMinutes": avg_dur, "PlayerLevel": level,
            "AchievementsUnlocked": achievements, "InGamePurchases": purchases,
            "playtime_per_session": pts, "weekly_activity_intensity": wai,
            "session_engagement_score": seg, "level_efficiency": le_,
            "achievement_rate": ar, "purchase_per_hour": pph, "activity_score": ascore,
            "Gender": gender, "Location": location, "GameGenre": genre, "GameDifficulty": difficulty,
        }])

        for col in CAT_FEATURES:
            le = LabelEncoder()
            df = load_data()
            le.fit(df[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))

        prob = model.predict_proba(input_df[NUMERIC_FEATURES + CAT_FEATURES])[:, 1][0]

        # 결과 표시
        if prob >= 0.8:
            risk, color, emoji = "CRITICAL", "#e74c3c", "🚨"
        elif prob >= 0.6:
            risk, color, emoji = "HIGH", "#e67e22", "⚠️"
        elif prob >= 0.3:
            risk, color, emoji = "MEDIUM", "#f1c40f", "📊"
        else:
            risk, color, emoji = "LOW", "#2ecc71", "✅"

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("이탈 확률", f"{prob:.1%}")
        col2.markdown(f"### {emoji} 위험도: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)

        if risk in ("CRITICAL", "HIGH"):
            col3.warning("즉시 리텐션 조치가 필요합니다: 복귀 보상, 개인화 이벤트, VIP 지원")
        elif risk == "MEDIUM":
            col3.info("모니터링 필요: 활동량 추이 관찰, 리텐션 오퍼 준비")
        else:
            col3.success("안정 유저: 현재 참여도 유지 전략 계속")


def page_segmentation():
    st.title("👥 유저 세그먼테이션")
    st.markdown("K-Means 클러스터링으로 유저를 4개 세그먼트로 분류합니다.")

    df = load_data()
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    seg_features = ["PlayTimeHours", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked", "InGamePurchases", "activity_score"]
    X_seg = df[seg_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    df["segment_id"] = kmeans.fit_predict(X_scaled)

    # 세그먼트별 activity_score 기준 정렬 → 라벨 할당
    cluster_scores = df.groupby("segment_id")["activity_score"].mean().sort_values(ascending=False)
    label_map = {}
    labels = ["🔥 Hardcore", "🎮 Casual", "⚠️ At Risk", "🆕 New User"]
    for i, cluster_id in enumerate(cluster_scores.index):
        label_map[cluster_id] = labels[i]
    df["segment"] = df["segment_id"].map(label_map)

    # 세그먼트 분포
    c1, c2 = st.columns(2)
    with c1:
        seg_counts = df["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.pie(seg_counts, values="count", names="segment", color_discrete_sequence=["#e74c3c", "#3498db", "#f39c12", "#2ecc71"])
        fig.update_layout(title="세그먼트 분포")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg_churn = df.groupby("segment")["is_churned"].mean().sort_values(ascending=False).reset_index()
        seg_churn.columns = ["segment", "churn_rate"]
        fig = px.bar(seg_churn, x="segment", y="churn_rate", color="churn_rate", color_continuous_scale="RdYlGn_r", labels={"churn_rate": "이탈률"})
        fig.update_layout(title="세그먼트별 이탈률")
        st.plotly_chart(fig, use_container_width=True)

    # 세그먼트별 상세
    st.subheader("세그먼트별 평균 지표")
    seg_summary = df.groupby("segment")[seg_features + ["is_churned"]].mean().round(2)
    seg_summary["유저 수"] = df.groupby("segment").size()
    st.dataframe(seg_summary, use_container_width=True)

    # 리텐션 전략
    st.subheader("세그먼트별 리텐션 전략")
    strategies = {
        "🔥 Hardcore": "VIP 보상 프로그램, 엔드게임 경쟁 콘텐츠, 독점 아이템, 길드 리더 기능",
        "🎮 Casual": "가벼운 일일 미션, 로그인 보상, 짧은 세션 콘텐츠, 소셜 기능 강화",
        "⚠️ At Risk": "복귀 보상 패키지, 개인화 알림, 이벤트 쿠폰, 진행도 리마인더",
        "🆕 New User": "튜토리얼 보상 강화, 초보자 가이드, 멘토 매칭, 첫 구매 할인",
    }
    for seg, strategy in strategies.items():
        st.markdown(f"**{seg}**: {strategy}")


def page_whatif():
    st.title("🔬 What-If 분석")
    st.markdown("피처 값을 변경하면서 이탈 확률이 어떻게 달라지는지 탐색합니다.")

    model = load_model()
    df = load_data()

    feat = st.selectbox("분석할 피처", ["PlayTimeHours", "SessionsPerWeek", "PlayerLevel", "InGamePurchases", "activity_score"])

    df_enc = df.copy()
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc[NUMERIC_FEATURES + CAT_FEATURES]
    base = X.median().to_frame().T
    base = pd.concat([base] * 50, ignore_index=True)

    feat_range = np.linspace(df[feat].quantile(0.05), df[feat].quantile(0.95), 50)
    base[feat] = feat_range
    probs = model.predict_proba(base)[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feat_range, y=probs, mode="lines+markers", line=dict(color="#8b5cf6", width=2), marker=dict(size=4)))
    fig.update_layout(xaxis_title=feat, yaxis_title="이탈 확률", height=400, title=f"{feat} 변화에 따른 이탈 확률 (Partial Dependence)")
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="임계값 0.5")
    st.plotly_chart(fig, use_container_width=True)


# ===== Main =====
PAGES = {
    "📋 Overview": page_overview,
    "📊 모델 성능": page_model_performance,
    "🔍 Feature Importance": page_feature_importance,
    "🎯 이탈 예측": page_prediction,
    "👥 세그먼테이션": page_segmentation,
    "🔬 What-If 분석": page_whatif,
}

with st.sidebar:
    st.title("🎮 GameAI Analytics")
    st.caption("게임 유저 이탈 예측 시스템")
    st.divider()
    page = st.radio("페이지", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.markdown("**Tech Stack**")
    st.caption("XGBoost · Optuna · SHAP · K-Means · FastAPI · MLflow · DVC · Airflow · Prometheus")
    st.divider()
    st.markdown("[GitHub](https://github.com/dbwjdtn10/GameAI_Analytics) · [Portfolio](https://dbwjdtn10.github.io)")

PAGES[page]()
