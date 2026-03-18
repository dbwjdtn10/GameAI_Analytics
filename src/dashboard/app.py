"""Streamlit 대시보드 - 게임 유저 이탈 분석."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
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


def _encode_for_model(df):
    """모델 입력용 인코딩."""
    feature_cols = get_feature_columns("kaggle")
    for col in feature_cols["categorical"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        df[col] = le.transform(df[col].astype(str))
    feature_path = MODEL_DIR / "feature_names.txt"
    feature_names = feature_path.read_text().strip().split("\n")
    return df[feature_names], feature_names


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
        genre_churn = (
            df.groupby("GameGenre")["is_churned"].mean().sort_values(ascending=False)
        )
        fig = px.bar(x=genre_churn.index, y=genre_churn.values,
                     labels={"x": "장르", "y": "이탈률"},
                     color=genre_churn.values, color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

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

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=["유지", "이탈"], y=["유지", "이탈"],
                    color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

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

    st.subheader("SHAP Analysis")
    shap_path = MODEL_DIR / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP Summary Plot")
    else:
        st.warning("SHAP 분석 결과가 없습니다. `python scripts/evaluate.py`를 실행하세요.")


def page_segment():
    """유저 세그먼트 페이지."""
    st.title("👥 유저 세그먼트")

    seg_path = MODEL_DIR / "segmenter.joblib"
    if not seg_path.exists():
        st.error("세그먼트 모델이 없습니다. `python scripts/segment.py`를 실행하세요.")
        return

    from src.models.segmenter import SEGMENT_LABELS, get_segment_summary

    df = load_data()

    try:
        summary = get_segment_summary(df)
    except Exception as e:
        st.error(f"세그먼트 분석 실패: {e}")
        return

    # KPI
    st.subheader("세그먼트 분포")
    c1, c2, c3, c4 = st.columns(4)
    for i, (seg_name, row) in enumerate(summary.iterrows()):
        col = [c1, c2, c3, c4][i % 4]
        col.metric(
            seg_name.upper(),
            f"{row['count']:,.0f}명 ({row['ratio']:.1%})",
            f"이탈률 {row['churn_rate']:.1%}",
        )

    st.divider()

    # 세그먼트별 비교 차트
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("세그먼트별 이탈률")
        fig = px.bar(
            summary.reset_index(), x="segment", y="churn_rate",
            color="churn_rate", color_continuous_scale="RdYlGn_r",
            text_auto=".1%",
        )
        fig.update_layout(yaxis_title="이탈률")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("세그먼트 크기")
        fig = px.pie(
            summary.reset_index(), values="count", names="segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 세그먼트별 피처 비교
    st.subheader("세그먼트별 주요 지표 비교")
    metrics_to_show = ["avg_playtime", "avg_level", "avg_purchases"]
    fig = go.Figure()
    for metric in metrics_to_show:
        fig.add_trace(go.Bar(
            x=summary.index, y=summary[metric], name=metric,
        ))
    fig.update_layout(barmode="group", xaxis_title="세그먼트")
    st.plotly_chart(fig, use_container_width=True)

    # 리텐션 전략
    st.subheader("세그먼트별 리텐션 전략")
    for seg_name, info in SEGMENT_LABELS.items():
        with st.expander(f"**{seg_name.upper()}** - {info['description']}"):
            for action in info["strategy"]:
                st.markdown(f"- {action}")


def page_monitoring():
    """드리프트 모니터링 페이지."""
    st.title("📡 모델 모니터링")

    from src.monitoring.drift import calculate_psi, detect_data_drift

    df = load_data()
    feature_cols = get_feature_columns("kaggle")
    numeric_features = feature_cols["numeric"]

    split_idx = int(len(df) * 0.7)
    ref_data = df.iloc[:split_idx]
    cur_data = df.iloc[split_idx:]

    st.info(
        f"기준 데이터: {len(ref_data):,}행 | "
        f"최근 데이터: {len(cur_data):,}행"
    )

    # 데이터 드리프트
    st.subheader("데이터 드리프트 (KS Test)")
    drift_result = detect_data_drift(ref_data, cur_data, numeric_features)

    c1, c2, c3 = st.columns(3)
    c1.metric("검사 피처 수", drift_result["total_features"])
    c2.metric("드리프트 피처 수", len(drift_result["drifted_features"]))
    drift_pct = drift_result["drift_ratio"]
    c3.metric("드리프트 비율", f"{drift_pct:.1%}",
              delta="정상" if drift_pct < 0.3 else "경고",
              delta_color="normal" if drift_pct < 0.3 else "inverse")

    details = drift_result["details"]
    drift_df = pd.DataFrame([
        {"피처": feat, "KS 통계량": d["statistic"], "p-value": d["p_value"],
         "드리프트": "⚠️ YES" if d["drifted"] else "✅ NO"}
        for feat, d in details.items()
    ]).sort_values("p-value")
    st.dataframe(drift_df, use_container_width=True, hide_index=True)

    st.divider()

    # PSI 분석
    st.subheader("PSI (Population Stability Index)")
    psi_results = []
    for feat in numeric_features:
        ref_vals = ref_data[feat].dropna().values
        cur_vals = cur_data[feat].dropna().values
        if len(ref_vals) > 0 and len(cur_vals) > 0:
            psi = calculate_psi(ref_vals, cur_vals)
            if psi < 0.1:
                status = "✅ 안정"
            elif psi < 0.2:
                status = "⚠️ 약간 변화"
            else:
                status = "🚨 유의미한 변화"
            psi_results.append({"피처": feat, "PSI": round(psi, 4), "상태": status})

    psi_df = pd.DataFrame(psi_results).sort_values("PSI", ascending=False)
    st.dataframe(psi_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        psi_df, x="PSI", y="피처", orientation="h",
        color="PSI", color_continuous_scale="RdYlGn_r",
        title="피처별 PSI",
    )
    fig.add_vline(x=0.1, line_dash="dash", line_color="orange",
                  annotation_text="약간 변화")
    fig.add_vline(x=0.2, line_dash="dash", line_color="red",
                  annotation_text="유의미한 변화")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 예측 분포 비교
    st.divider()
    st.subheader("예측 분포 비교 (기준 vs 최근)")

    model = load_model()
    if model is not None:
        cat_features = feature_cols["categorical"]
        all_features = numeric_features + cat_features

        ref_enc = ref_data.copy()
        cur_enc = cur_data.copy()
        for col in cat_features:
            le = LabelEncoder()
            le.fit(pd.concat([ref_enc[col].astype(str), cur_enc[col].astype(str)]))
            ref_enc[col] = le.transform(ref_enc[col].astype(str))
            cur_enc[col] = le.transform(cur_enc[col].astype(str))

        ref_proba = model.predict_proba(ref_enc[all_features])[:, 1]
        cur_proba = model.predict_proba(cur_enc[all_features])[:, 1]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ref_proba, name="기준 데이터", opacity=0.7, nbinsx=30,
        ))
        fig.add_trace(go.Histogram(
            x=cur_proba, name="최근 데이터", opacity=0.7, nbinsx=30,
        ))
        fig.update_layout(barmode="overlay", xaxis_title="이탈 확률",
                          yaxis_title="빈도", title="예측 확률 분포 비교")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("기준 평균 이탈 확률", f"{np.mean(ref_proba):.4f}")
        c2.metric("최근 평균 이탈 확률", f"{np.mean(cur_proba):.4f}")


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


def page_whatif():
    """What-If 분석 페이지."""
    st.title("🔬 What-If 분석")
    st.markdown(
        "피처 값을 변경하면 이탈 확률이 어떻게 변하는지 실시간으로 확인합니다. "
        "**개입(Intervention) 효과를 시뮬레이션**하세요."
    )

    model = load_model()
    if model is None:
        st.error("학습된 모델이 없습니다.")
        return

    feature_path = MODEL_DIR / "feature_names.txt"
    if not feature_path.exists():
        st.error("피처 이름 파일이 없습니다.")
        return

    # 기본 유저 프로필
    st.subheader("기준 유저 프로필")
    col1, col2, col3 = st.columns(3)
    with col1:
        base_age = st.number_input("나이", 10, 100, 25, key="wi_age")
        base_gender = st.selectbox("성별", ["Male", "Female"], key="wi_gender")
        base_genre = st.selectbox("장르", [
            "RPG", "FPS", "MOBA", "Sports", "Strategy",
            "Simulation", "Action", "Adventure",
        ], key="wi_genre")
        base_diff = st.selectbox("난이도", ["Easy", "Medium", "Hard"], key="wi_diff")
    with col2:
        base_playtime = st.number_input("플레이 시간(h)", 0.0, 5000.0, 50.0, key="wi_pt")
        base_sessions = st.number_input("주간 세션", 0, 50, 3, key="wi_sess")
        base_duration = st.number_input("세션 시간(min)", 0.0, 600.0, 30.0, key="wi_dur")
        base_location = st.selectbox("지역", [
            "USA", "Europe", "Asia", "South America",
            "Africa", "Australia", "Other",
        ], key="wi_loc")
    with col3:
        base_level = st.number_input("레벨", 1, 200, 15, key="wi_lvl")
        base_achievements = st.number_input("업적", 0, 500, 5, key="wi_ach")
        base_purchases = st.number_input("구매 횟수", 0, 500, 1, key="wi_pur")

    base_data = {
        "Age": base_age, "Gender": base_gender, "Location": base_location,
        "GameGenre": base_genre, "GameDifficulty": base_diff,
        "PlayTimeHours": base_playtime, "SessionsPerWeek": base_sessions,
        "AvgSessionDurationMinutes": base_duration,
        "PlayerLevel": base_level, "AchievementsUnlocked": base_achievements,
        "InGamePurchases": base_purchases,
    }

    # 기준 예측
    raw_base = pd.DataFrame([base_data])
    raw_base = engineer_gaming_behavior_features(raw_base)
    X_base, feature_names = _encode_for_model(raw_base)
    base_proba = model.predict_proba(X_base)[0, 1]

    st.metric("기준 이탈 확률", f"{base_proba:.1%}")

    st.divider()

    # What-If: 수치형 피처 슬라이더
    st.subheader("개입 시뮬레이션")
    st.markdown("아래 슬라이더로 값을 변경하면 이탈 확률 변화를 실시간으로 확인합니다.")

    c1, c2 = st.columns(2)
    with c1:
        wi_sessions = st.slider(
            "주간 세션 수 변경", 0, 30, base_sessions, key="wi_s_sessions"
        )
        wi_playtime = st.slider(
            "플레이 시간(h) 변경", 0.0, 3000.0, base_playtime, step=10.0, key="wi_s_pt"
        )
        wi_purchases = st.slider(
            "구매 횟수 변경", 0, 200, base_purchases, key="wi_s_pur"
        )
    with c2:
        wi_duration = st.slider(
            "세션 시간(min) 변경", 0.0, 300.0, base_duration, step=5.0, key="wi_s_dur"
        )
        wi_level = st.slider(
            "레벨 변경", 1, 200, base_level, key="wi_s_lvl"
        )
        wi_achievements = st.slider(
            "업적 수 변경", 0, 300, base_achievements, key="wi_s_ach"
        )

    # What-If 예측
    wi_data = base_data.copy()
    wi_data.update({
        "SessionsPerWeek": wi_sessions,
        "PlayTimeHours": wi_playtime,
        "InGamePurchases": wi_purchases,
        "AvgSessionDurationMinutes": wi_duration,
        "PlayerLevel": wi_level,
        "AchievementsUnlocked": wi_achievements,
    })

    raw_wi = pd.DataFrame([wi_data])
    raw_wi = engineer_gaming_behavior_features(raw_wi)
    X_wi, _ = _encode_for_model(raw_wi)
    wi_proba = model.predict_proba(X_wi)[0, 1]

    delta = wi_proba - base_proba
    c1, c2, c3 = st.columns(3)
    c1.metric("변경 후 이탈 확률", f"{wi_proba:.1%}")
    c2.metric("변화량", f"{delta:+.1%}",
              delta_color="inverse" if delta > 0 else "normal")
    c3.metric(
        "효과",
        "개선" if delta < -0.05 else ("악화" if delta > 0.05 else "미미"),
    )

    # 피처별 민감도 분석 (Partial Dependence)
    st.divider()
    st.subheader("피처 민감도 분석 (Partial Dependence)")
    st.markdown("선택한 피처를 변화시켰을 때 이탈 확률이 어떻게 변하는지 보여줍니다.")

    sensitivity_feature = st.selectbox(
        "분석할 피처",
        ["SessionsPerWeek", "PlayTimeHours", "AvgSessionDurationMinutes",
         "PlayerLevel", "AchievementsUnlocked", "InGamePurchases"],
        key="wi_sens_feat",
    )

    # 범위 설정
    ranges = {
        "SessionsPerWeek": (0, 30, 31),
        "PlayTimeHours": (0, 2000, 50),
        "AvgSessionDurationMinutes": (5, 300, 50),
        "PlayerLevel": (1, 150, 50),
        "AchievementsUnlocked": (0, 200, 50),
        "InGamePurchases": (0, 100, 50),
    }
    lo, hi, n_points = ranges[sensitivity_feature]
    values = np.linspace(lo, hi, n_points)

    probas = []
    for v in values:
        tmp = base_data.copy()
        tmp[sensitivity_feature] = float(v)
        raw_tmp = pd.DataFrame([tmp])
        raw_tmp = engineer_gaming_behavior_features(raw_tmp)
        X_tmp, _ = _encode_for_model(raw_tmp)
        p = model.predict_proba(X_tmp)[0, 1]
        probas.append(p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values, y=probas, mode="lines+markers",
        name="이탈 확률",
        line=dict(width=3),
        marker=dict(size=4),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="이탈 임계값 (0.5)")
    fig.add_vline(x=base_data[sensitivity_feature], line_dash="dot",
                  line_color="blue", annotation_text="현재 값")
    fig.update_layout(
        xaxis_title=sensitivity_feature,
        yaxis_title="이탈 확률",
        yaxis_range=[0, 1],
        title=f"{sensitivity_feature} 변화에 따른 이탈 확률",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_business_impact():
    """비즈니스 임팩트 분석 페이지."""
    st.title("💰 비즈니스 임팩트 분석")
    st.markdown("이탈 예측 모델의 **비즈니스 가치(ROI)** 를 정량화합니다.")

    model = load_model()
    y_test, y_pred, y_proba = get_test_predictions()
    if model is None or y_test is None:
        st.error("모델 또는 테스트 데이터가 없습니다.")
        return

    # 비즈니스 파라미터 설정
    st.subheader("비즈니스 파라미터 설정")
    c1, c2, c3 = st.columns(3)
    with c1:
        avg_ltv = st.number_input("유저당 평균 LTV ($)", 10, 10000, 150)
        monthly_active = st.number_input("월간 활성 유저 (MAU)", 1000, 10000000, 100000)
    with c2:
        retention_cost = st.number_input("유저당 리텐션 비용 ($)", 0.1, 100.0, 5.0)
        retention_success_rate = st.slider("리텐션 성공률 (%)", 5, 80, 25) / 100
    with c3:
        current_churn_rate = st.slider("현재 월간 이탈률 (%)", 1, 50, 25) / 100

    st.divider()

    # 임계값별 비용-편익 분석
    st.subheader("임계값(Threshold) 최적화")
    st.markdown("이탈 예측 임계값에 따른 비용-편익 트레이드오프를 분석합니다.")

    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tp = ((preds == 1) & (y_test.values == 1)).sum()  # 이탈 정확 예측
        fp = ((preds == 1) & (y_test.values == 0)).sum()  # 오탐 (유지 유저에 불필요 개입)
        fn = ((preds == 0) & (y_test.values == 1)).sum()  # 미탐 (이탈 놓침)

        # 비용 계산
        intervention_cost = (tp + fp) * retention_cost
        saved_revenue = tp * retention_success_rate * avg_ltv
        lost_revenue = fn * avg_ltv
        net_benefit = saved_revenue - intervention_cost

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        results.append({
            "threshold": round(t, 2),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "intervention_cost": round(intervention_cost, 0),
            "saved_revenue": round(saved_revenue, 0),
            "lost_revenue": round(lost_revenue, 0),
            "net_benefit": round(net_benefit, 0),
        })

    results_df = pd.DataFrame(results)
    best_idx = results_df["net_benefit"].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]
    best_benefit = results_df.loc[best_idx, "net_benefit"]

    st.success(
        f"최적 임계값: **{best_threshold:.2f}** "
        f"(순 이익: **${best_benefit:,.0f}** / 테스트 세트 기준)"
    )

    # 차트: 순이익 vs 임계값
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df["threshold"], y=results_df["net_benefit"],
        mode="lines+markers", name="순 이익 ($)",
        line=dict(width=3, color="#2ecc71"),
    ))
    fig.add_trace(go.Scatter(
        x=results_df["threshold"], y=results_df["intervention_cost"],
        mode="lines", name="개입 비용 ($)",
        line=dict(dash="dash", color="#e74c3c"),
    ))
    fig.add_trace(go.Scatter(
        x=results_df["threshold"], y=results_df["saved_revenue"],
        mode="lines", name="보존 매출 ($)",
        line=dict(dash="dash", color="#3498db"),
    ))
    fig.add_vline(x=best_threshold, line_dash="dot", line_color="green",
                  annotation_text=f"최적: {best_threshold}")
    fig.update_layout(
        xaxis_title="이탈 예측 임계값",
        yaxis_title="금액 ($)",
        title="임계값별 비용-편익 분석",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 상세 테이블
    with st.expander("임계값별 상세 결과"):
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.divider()

    # 연간 ROI 추정
    st.subheader("연간 ROI 추정")

    monthly_at_risk = monthly_active * current_churn_rate
    from sklearn.metrics import precision_score, recall_score

    opt_preds = (y_proba >= best_threshold).astype(int)
    opt_precision = precision_score(y_test, opt_preds)
    opt_recall = recall_score(y_test, opt_preds)

    monthly_detected = monthly_at_risk * opt_recall
    monthly_false_alerts = (monthly_active - monthly_at_risk) * (1 - opt_precision) * (
        opt_recall / max(1 - opt_recall, 0.01)
    )
    monthly_saved = monthly_detected * retention_success_rate
    monthly_revenue_saved = monthly_saved * avg_ltv
    monthly_cost = (monthly_detected + monthly_false_alerts) * retention_cost
    monthly_net = monthly_revenue_saved - monthly_cost
    annual_net = monthly_net * 12
    roi = (monthly_revenue_saved - monthly_cost) / max(monthly_cost, 1) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("월간 이탈 위험 유저", f"{monthly_at_risk:,.0f}")
    c2.metric("월간 탐지 유저", f"{monthly_detected:,.0f}")
    c3.metric("월간 보존 유저", f"{monthly_saved:,.0f}")
    c4.metric("ROI", f"{roi:,.0f}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("월간 보존 매출", f"${monthly_revenue_saved:,.0f}")
    c2.metric("월간 개입 비용", f"${monthly_cost:,.0f}")
    c3.metric("연간 순 이익", f"${annual_net:,.0f}",
              delta=f"월 ${monthly_net:,.0f}")


# 사이드바 네비게이션
page = st.sidebar.radio("페이지", [
    "개요", "모델 성능", "유저 세그먼트", "모니터링",
    "이탈 예측", "What-If 분석", "비즈니스 임팩트",
])

if page == "개요":
    page_overview()
elif page == "모델 성능":
    page_model_performance()
elif page == "유저 세그먼트":
    page_segment()
elif page == "모니터링":
    page_monitoring()
elif page == "이탈 예측":
    page_prediction()
elif page == "What-If 분석":
    page_whatif()
elif page == "비즈니스 임팩트":
    page_business_impact()
