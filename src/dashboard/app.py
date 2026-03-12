"""Streamlit 대시보드 - 게임 유저 이탈 분석."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import MODEL_DIR, RANDOM_STATE, TEST_SIZE
from src.data.loader import load_gaming_behavior
from src.features.engineer import engineer_gaming_behavior_features, get_feature_columns

st.set_page_config(
    page_title="GameAI Analytics",
    page_icon="🎮",
    layout="wide",
)


@st.cache_data
def load_data():
    """데이터 로드 + 피처 엔지니어링."""
    df = load_gaming_behavior()
    df = engineer_gaming_behavior_features(df)
    return df


@st.cache_resource
def load_model():
    """학습된 모델 로드."""
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def get_test_predictions():
    """테스트 셋 예측 결과 생성."""
    df = load_data()
    feature_cols = get_feature_columns("kaggle")
    cat_features = feature_cols["categorical"]
    numeric_features = feature_cols["numeric"]

    df_enc = df.copy()
    for col in cat_features:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    all_features = numeric_features + cat_features
    X = df_enc[all_features]
    y = df_enc["is_churned"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = load_model()
    if model is None:
        return None, None, None

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return y_test, y_pred, y_proba


def page_overview():
    """개요 페이지."""
    st.title("🎮 GameAI Analytics Dashboard")
    st.markdown("게임 유저 행동 분석 & 이탈 예측 시스템")

    df = load_data()
    total = len(df)
    churned = df["is_churned"].sum()
    churn_rate = churned / total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 유저 수", f"{total:,}")
    col2.metric("이탈 유저", f"{churned:,}")
    col3.metric("이탈률", f"{churn_rate:.1%}")
    col4.metric("유지 유저", f"{total - churned:,}")

    st.divider()

    # 이탈 분포
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("이탈 분포")
        churn_counts = df["is_churned"].value_counts().reset_index()
        churn_counts.columns = ["is_churned", "count"]
        churn_counts["label"] = churn_counts["is_churned"].map({0: "유지", 1: "이탈"})
        fig = px.pie(churn_counts, values="count", names="label",
                     color_discrete_sequence=["#3498db", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("장르별 이탈률")
        genre_churn = df.groupby("GameGenre")["is_churned"].mean().sort_values(ascending=False)
        fig = px.bar(x=genre_churn.index, y=genre_churn.values,
                     labels={"x": "장르", "y": "이탈률"},
                     color=genre_churn.values, color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

    # 주요 피처 분포
    st.subheader("주요 피처 분포 (이탈 vs 유지)")
    feat = st.selectbox("피처 선택", [
        "PlayTimeHours", "SessionsPerWeek", "AvgSessionDurationMinutes",
        "PlayerLevel", "AchievementsUnlocked", "InGamePurchases",
        "activity_score", "playtime_per_session",
    ])
    fig = px.histogram(df, x=feat, color="is_churned", barmode="overlay",
                       color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                       labels={"is_churned": "이탈 여부"},
                       opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)


def page_model_performance():
    """모델 성능 페이지."""
    st.title("📊 모델 성능")

    model = load_model()
    if model is None:
        st.error("학습된 모델이 없습니다. `python scripts/train.py`를 실행하세요.")
        return

    st.info(f"모델: **{type(model).__name__}**")

    y_test, y_pred, y_proba = get_test_predictions()
    if y_test is None:
        st.error("예측 데이터를 생성할 수 없습니다.")
        return

    # 메트릭
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    c3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    c4.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
    c5.metric("AUC-ROC", f"{roc_auc_score(y_test, y_proba):.4f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash"), name="Random"))
        fig.update_layout(xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Precision-Recall Curve")
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="Model"))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=["유지", "이탈"], y=["유지", "이탈"],
                    color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        feature_path = MODEL_DIR / "feature_names.txt"
        if feature_path.exists():
            names = feature_path.read_text().strip().split("\n")
            imp = model.feature_importances_
            feat_df = pd.DataFrame({"feature": names, "importance": imp})
            feat_df = feat_df.sort_values("importance", ascending=True).tail(15)
            fig = px.bar(feat_df, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

    # SHAP 이미지
    st.subheader("SHAP Analysis")
    shap_path = MODEL_DIR / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP Summary Plot")
    else:
        st.warning("SHAP 분석 결과가 없습니다. `python scripts/evaluate.py`를 실행하세요.")


def page_prediction():
    """실시간 예측 페이지."""
    st.title("🔮 이탈 예측")

    model = load_model()
    if model is None:
        st.error("학습된 모델이 없습니다.")
        return

    st.markdown("유저 정보를 입력하면 이탈 확률을 예측합니다.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("나이", 10, 100, 25)
        gender = st.selectbox("성별", ["Male", "Female"])
        location = st.selectbox("지역", [
            "USA", "Europe", "Asia", "South America",
            "Africa", "Australia", "Other",
        ])
        game_genre = st.selectbox("게임 장르", [
            "RPG", "FPS", "MOBA", "Sports", "Strategy",
            "Simulation", "Action", "Adventure",
        ])

    with col2:
        difficulty = st.selectbox("난이도", ["Easy", "Medium", "Hard"])
        playtime = st.number_input("총 플레이 시간(시간)", 0.0, 10000.0, 100.0)
        sessions = st.number_input("주간 세션 수", 0, 50, 5)
        session_dur = st.number_input("평균 세션 시간(분)", 0.0, 600.0, 45.0)

    with col3:
        level = st.number_input("플레이어 레벨", 1, 200, 30)
        achievements = st.number_input("업적 수", 0, 500, 15)
        purchases = st.number_input("인게임 구매 횟수", 0, 500, 5)

    if st.button("이탈 예측", type="primary"):
        raw = pd.DataFrame([{
            "Age": age, "Gender": gender, "Location": location,
            "GameGenre": game_genre, "GameDifficulty": difficulty,
            "PlayTimeHours": playtime, "SessionsPerWeek": sessions,
            "AvgSessionDurationMinutes": session_dur,
            "PlayerLevel": level, "AchievementsUnlocked": achievements,
            "InGamePurchases": purchases,
        }])

        raw = engineer_gaming_behavior_features(raw)
        feature_cols = get_feature_columns("kaggle")
        for col in feature_cols["categorical"]:
            le = LabelEncoder()
            le.fit(raw[col].astype(str))
            raw[col] = le.transform(raw[col].astype(str))

        feature_path = MODEL_DIR / "feature_names.txt"
        feature_names = feature_path.read_text().strip().split("\n")
        X = raw[feature_names]

        proba = model.predict_proba(X)[0, 1]

        if proba < 0.3:
            risk, color = "LOW", "green"
        elif proba < 0.5:
            risk, color = "MEDIUM", "orange"
        elif proba < 0.7:
            risk, color = "HIGH", "red"
        else:
            risk, color = "CRITICAL", "red"

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("이탈 확률", f"{proba:.1%}")
        c2.markdown(f"### 위험 등급: :{color}[{risk}]")

        # 게이지 차트
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "이탈 확률 (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "#2ecc71"},
                    {"range": [30, 50], "color": "#f39c12"},
                    {"range": [50, 70], "color": "#e67e22"},
                    {"range": [70, 100], "color": "#e74c3c"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)


# 사이드바 네비게이션
page = st.sidebar.radio("페이지", ["개요", "모델 성능", "이탈 예측"])

if page == "개요":
    page_overview()
elif page == "모델 성능":
    page_model_performance()
elif page == "이탈 예측":
    page_prediction()
