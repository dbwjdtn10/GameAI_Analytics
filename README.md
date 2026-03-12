# GameAI Analytics

게임 유저 행동 분석 & 이탈 예측 시스템

## Overview

게임 유저의 행동 데이터를 분석하여 이탈(churn)을 예측하고, 실시간 API 및 대시보드를 통해 인사이트를 제공하는 end-to-end ML 파이프라인 프로젝트입니다.

### Key Features

- **EDA & 통계 분석**: t-test, chi-square, A/B testing (Z-test, Bootstrap, Cohen's h)
- **ML 파이프라인**: Logistic Regression, XGBoost, LightGBM, Ensemble (Voting/Stacking)
- **하이퍼파라미터 튜닝**: Optuna 기반 자동 튜닝
- **모델 해석**: SHAP (Summary, Bar, Waterfall)
- **유저 세그먼트**: K-Means 클러스터링 + 세그먼트별 리텐션 전략
- **API 서빙**: FastAPI + API Key 인증 (예측, 세그먼트, 피처 중요도)
- **대시보드**: Streamlit + Plotly 인터랙티브 시각화
- **MLOps**: MLflow 실험 추적, 드리프트 탐지 (KS test, PSI), Airflow 재학습
- **CI/CD**: GitHub Actions, Docker, docker-compose
- **DB**: PostgreSQL 데이터 적재 + SQL 분석 쿼리

## Project Structure

```
GameAI_Analytics/
├── data/
│   ├── raw/              # 원본 데이터 (Kaggle)
│   ├── processed/        # 전처리된 데이터
│   └── synthetic/        # 합성 데이터
├── models/               # 학습된 모델, 차트
├── notebooks/
│   ├── 01_eda_gaming_behavior.ipynb   # EDA + 통계 검정
│   ├── 02_eda_cookie_cats.ipynb       # A/B 테스트 EDA
│   ├── 03_feature_engineering.ipynb   # 피처 엔지니어링 분석
│   ├── 04_model_comparison.ipynb      # 모델 성능 비교
│   └── 05_ab_test_analysis.ipynb      # A/B 테스트 통계 분석
├── scripts/
│   ├── train.py              # 모델 학습
│   ├── evaluate.py           # SHAP 분석
│   ├── segment.py            # K-Means 세그먼트
│   ├── generate_synthetic.py # 합성 데이터 생성
│   └── load_to_postgres.py   # PostgreSQL 적재
├── sql/
│   ├── schema.sql        # DB 스키마
│   └── queries.sql       # 분석 쿼리
├── src/
│   ├── api/              # FastAPI 서버
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── dependencies.py
│   │   ├── middleware.py
│   │   └── routes/       # predict, segment, model_info, health
│   ├── dashboard/        # Streamlit 대시보드
│   ├── data/             # 데이터 로딩/전처리/DB
│   ├── features/         # 피처 엔지니어링/선택
│   ├── models/           # 학습/평가/레지스트리/세그먼트
│   └── monitoring/       # 드리프트 탐지
├── dags/
│   └── retrain_dag.py    # Airflow 재학습 DAG
├── tests/                # pytest 테스트 (45개)
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
└── pyproject.toml
```

## Quick Start

### 1. Setup

```bash
pip install ".[dev]"
```

### 2. 모델 학습

```bash
# 기본 학습 (5개 모델)
python scripts/train.py

# Optuna 튜닝 포함
python scripts/train.py --tune --tune-trials 50

# SHAP 분석
python scripts/evaluate.py

# 유저 세그먼트
python scripts/segment.py
```

### 3. API 서버

```bash
python -m src.api.main
```

**API 엔드포인트:**

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/predict/single` | 단일 유저 이탈 예측 |
| POST | `/api/v1/predict/batch` | 배치 예측 (최대 1000명) |
| POST | `/api/v1/segment/classify` | 유저 세그먼트 분류 |
| GET | `/api/v1/model/info` | 모델 정보 |
| GET | `/api/v1/model/features/importance` | 피처 중요도 |
| GET | `/health` | 헬스 체크 |

**예측 요청 예시:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "X-API-Key: dev-key-gameai-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25, "Gender": "Male", "Location": "USA",
    "GameGenre": "RPG", "GameDifficulty": "Medium",
    "PlayTimeHours": 120.5, "SessionsPerWeek": 5,
    "AvgSessionDurationMinutes": 45.0, "PlayerLevel": 32,
    "AchievementsUnlocked": 15, "InGamePurchases": 8
  }'
```

**응답:**
```json
{
  "churn_probability": 0.73,
  "churn_prediction": true,
  "risk_level": "critical",
  "top_risk_factors": [
    {"feature": "activity_score", "impact": 0.18, "description": "종합 활동 점수"},
    {"feature": "SessionsPerWeek", "impact": 0.15, "description": "주간 세션 수"}
  ],
  "recommended_actions": [
    "긴급 복귀 보상 지급 (7일 미접속 시)",
    "1:1 맞춤 이벤트 쿠폰 제공"
  ]
}
```

### 4. 대시보드

```bash
streamlit run src/dashboard/app.py
```

### 5. Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# MLflow: http://localhost:5000
# PostgreSQL: localhost:5432
```

### 6. PostgreSQL

```bash
docker-compose up -d postgres
python scripts/load_to_postgres.py
```

## Model Performance

| Model | AUC-ROC | Accuracy | F1-Score |
|-------|---------|----------|----------|
| LogisticRegression | 0.9233 | 0.8423 | 0.7475 |
| **XGBoost** | **0.9409** | **0.9535** | **0.9083** |
| LightGBM | 0.9392 | 0.9518 | 0.9045 |
| VotingEnsemble | 0.9383 | 0.9510 | 0.9039 |
| StackingEnsemble | 0.9402 | 0.9514 | 0.9044 |

Best model: **XGBoost** (AUC-ROC 0.9409)

## Tech Stack

| 영역 | 기술 |
|------|------|
| ML | scikit-learn, XGBoost, LightGBM, SHAP, Optuna |
| API | FastAPI, Pydantic, uvicorn |
| Dashboard | Streamlit, Plotly |
| DB | PostgreSQL, SQLAlchemy |
| MLOps | MLflow, DVC, Airflow |
| Infra | Docker, docker-compose, GitHub Actions |
| Testing | pytest (45 tests), Ruff linter |

## Datasets

1. **Online Gaming Behavior Dataset** (Kaggle) - 40,034 rows
   - 이탈 정의: EngagementLevel = "Low"
2. **Cookie Cats A/B Test** (Kaggle) - 90,189 rows
   - 리텐션 분석, A/B 테스트 통계 검정
3. **Synthetic Data** - 10,000 rows
   - 시계열 유저 행동 시뮬레이션

## Tests

```bash
pytest tests/ -v        # 전체 테스트 (45개)
ruff check src/ tests/  # 린트
```
