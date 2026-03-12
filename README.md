# GameAI Analytics

> 게임 유저 행동 분석 & 이탈 예측 시스템
> End-to-End ML Pipeline for Game Player Churn Prediction

[![CI](https://github.com/dbwjdtn10/GameAI_Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/dbwjdtn10/GameAI_Analytics/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Overview

게임 유저의 플레이 로그를 분석하여 **이탈(churn)을 예측**하고, **유저 세그먼트별 맞춤 리텐션 전략**을 제안하는 프로젝트입니다.
데이터 수집부터 모델 학습, API 서빙, 대시보드, MLOps까지 **프로덕션 수준의 전체 파이프라인**을 구현했습니다.

### Highlights

- **XGBoost 기반 이탈 예측** — AUC-ROC 0.9409, 5개 모델 비교 실험
- **K-Means 유저 세그먼트** — 4개 클러스터별 리텐션 전략 자동 제안
- **SHAP 모델 해석** — 피처 기여도 시각화로 예측 근거 설명
- **FastAPI 실시간 서빙** — 단일/배치 예측 API + API Key 인증
- **Streamlit 대시보드** — 5페이지 인터랙티브 모니터링
- **MLOps 파이프라인** — MLflow 실험 추적, Airflow 재학습, 드리프트 탐지

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GameAI Analytics                             │
├─────────────┬──────────────┬──────────────┬────────────────────────┤
│  Data Layer │  ML Pipeline │  Serving     │  MLOps                 │
├─────────────┼──────────────┼──────────────┼────────────────────────┤
│ Kaggle CSV  │ Feature Eng  │ FastAPI      │ MLflow Tracking        │
│ Synthetic   │ XGBoost      │  - Predict   │ DVC Pipeline           │
│ PostgreSQL  │ LightGBM     │  - Segment   │ Airflow DAG            │
│             │ Ensemble     │  - Model Info│ Drift Detection        │
│             │ SHAP         │ Streamlit    │  - KS Test             │
│             │ Optuna       │  - 5 Pages   │  - PSI                 │
│             │ K-Means      │              │ GitHub Actions CI      │
└─────────────┴──────────────┴──────────────┴────────────────────────┘
```

```
유저 데이터 (CSV/DB)
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  데이터 로드   │───▶│  피처 엔지니어링 │───▶│   모델 학습    │
│  전처리/검증   │    │  12개 파생 피처  │    │  5개 모델 비교  │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐            │
                    │  SHAP 해석    │◀───────────┤
                    │  피처 중요도   │            │
                    └──────────────┘            ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Streamlit   │◀───│  FastAPI      │◀───│  모델 저장     │
│  대시보드     │    │  REST API     │    │  (joblib)     │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                    ┌──────────────┐            │
                    │  MLflow      │◀───────────┘
                    │  실험 추적    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐    ┌──────────────┐
                    │  드리프트 탐지 │───▶│  Airflow      │
                    │  KS / PSI    │    │  자동 재학습   │
                    └──────────────┘    └──────────────┘
```

---

## Tech Stack

| 영역 | 기술 | 용도 |
|------|------|------|
| **ML** | scikit-learn, XGBoost, LightGBM | 이탈 예측 모델 학습 |
| **해석** | SHAP | 모델 예측 설명 |
| **튜닝** | Optuna | 하이퍼파라미터 자동 최적화 |
| **세그먼트** | K-Means | 유저 군집화 + 리텐션 전략 |
| **통계** | SciPy | t-test, chi-square, A/B testing |
| **API** | FastAPI, Pydantic | 실시간 예측 서빙 |
| **대시보드** | Streamlit, Plotly | 인터랙티브 시각화 |
| **DB** | PostgreSQL, SQLAlchemy | 유저 데이터 적재 + SQL 분석 |
| **MLOps** | MLflow, DVC, Airflow | 실험 추적, 데이터 버전, 재학습 |
| **모니터링** | KS test, PSI | 데이터/예측 드리프트 탐지 |
| **인프라** | Docker, docker-compose | 컨테이너 오케스트레이션 |
| **CI/CD** | GitHub Actions, pre-commit | 자동 테스트 + 린트 |
| **테스트** | pytest (45 tests), Ruff | 품질 보증 |

---

## Model Performance

5개 모델을 동일 조건에서 학습/평가한 결과:

| Model | AUC-ROC | Accuracy | F1-Score | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
| LogisticRegression | 0.9233 | 0.8423 | 0.7475 | 0.7081 | 0.7918 |
| **XGBoost** | **0.9409** | **0.9535** | **0.9083** | **0.9198** | **0.8971** |
| LightGBM | 0.9392 | 0.9518 | 0.9045 | 0.9184 | 0.8909 |
| VotingEnsemble | 0.9383 | 0.9510 | 0.9039 | 0.9032 | 0.9046 |
| StackingEnsemble | 0.9402 | 0.9514 | 0.9044 | 0.9143 | 0.8948 |

> Best: **XGBoost** (AUC-ROC 0.9409)
> 데이터 분할: Train 60% / Validation 20% / Test 20% (stratified)

---

## Project Structure

```
GameAI_Analytics/
│
├── src/                          # 소스 코드
│   ├── config.py                 # 프로젝트 설정 (경로, 상수)
│   ├── api/                      # FastAPI 서버
│   │   ├── main.py               #   앱 진입점 + lifespan
│   │   ├── schemas.py            #   Pydantic 요청/응답 모델
│   │   ├── dependencies.py       #   ModelService + API Key 인증
│   │   ├── middleware.py          #   요청 로깅 미들웨어
│   │   └── routes/               #   라우터 모듈
│   │       ├── predict.py        #     단일/배치 예측
│   │       ├── segment.py        #     유저 세그먼트 분류
│   │       ├── model_info.py     #     모델 정보/피처 중요도
│   │       └── health.py         #     헬스 체크
│   ├── dashboard/                # Streamlit 대시보드
│   │   └── app.py                #   5페이지 (개요/모델/세그먼트/모니터링/예측)
│   ├── data/                     # 데이터 레이어
│   │   ├── loader.py             #   CSV 데이터 로딩
│   │   ├── preprocessor.py       #   전처리 파이프라인
│   │   ├── synthetic.py          #   합성 데이터 생성기
│   │   └── database.py           #   PostgreSQL 연동
│   ├── features/                 # 피처 엔지니어링
│   │   ├── engineer.py           #   12개 파생 피처 생성
│   │   └── selector.py           #   피처 선택 (MI, VIF)
│   ├── models/                   # ML 모델
│   │   ├── trainer.py            #   5개 모델 + Optuna 튜닝
│   │   ├── evaluator.py          #   평가 리포트 + 시각화
│   │   ├── registry.py           #   MLflow 실험 기록
│   │   └── segmenter.py          #   K-Means 세그먼트
│   └── monitoring/               # 모니터링
│       └── drift.py              #   KS test + PSI 드리프트 탐지
│
├── scripts/                      # 실행 스크립트
│   ├── train.py                  #   모델 학습 파이프라인
│   ├── evaluate.py               #   SHAP 분석
│   ├── segment.py                #   세그먼트 클러스터링
│   ├── generate_synthetic.py     #   합성 데이터 생성
│   └── load_to_postgres.py       #   PostgreSQL 적재
│
├── notebooks/                    # 분석 노트북 (출력 포함)
│   ├── 01_eda_gaming_behavior.ipynb  # EDA + 통계 검정
│   ├── 02_eda_cookie_cats.ipynb      # A/B 테스트 EDA
│   ├── 03_feature_engineering.ipynb  # 피처 분석 (상관관계, VIF, MI)
│   ├── 04_model_comparison.ipynb     # 모델 성능 비교
│   └── 05_ab_test_analysis.ipynb     # A/B 테스트 통계 분석
│
├── sql/                          # SQL
│   ├── schema.sql                #   DB 스키마 (players, predictions, model_registry)
│   └── queries.sql               #   분석 쿼리 (이탈률, 세그먼트, 코호트)
│
├── dags/                         # Airflow
│   └── retrain_dag.py            #   주간 재학습 DAG (드리프트 → 학습 → 배포)
│
├── tests/                        # 테스트 (45개)
│   ├── test_api.py               #   API 엔드포인트 (9 tests)
│   ├── test_data_quality.py      #   데이터 품질 (13 tests)
│   ├── test_drift.py             #   드리프트 탐지 (5 tests)
│   ├── test_features.py          #   피처 엔지니어링 (5 tests)
│   ├── test_model_regression.py  #   모델 성능 회귀 (4 tests)
│   ├── test_models.py            #   모델 학습 (5 tests)
│   └── test_preprocessor.py      #   전처리 (4 tests)
│
├── data/
│   ├── raw/                      # 원본 데이터 (Kaggle)
│   ├── processed/                # 전처리 데이터
│   └── synthetic/                # 합성 데이터
│
├── models/                       # 학습 결과물
│   ├── best_model.joblib         #   최고 성능 모델
│   ├── roc_comparison.png        #   ROC 커브 비교
│   ├── confusion_matrix.png      #   혼동 행렬
│   └── feature_importance.png    #   피처 중요도
│
├── Dockerfile                    # API 컨테이너
├── docker-compose.yml            # 4 서비스 (API/Dashboard/MLflow/PostgreSQL)
├── pyproject.toml                # 프로젝트 메타데이터 + 의존성
├── dvc.yaml                      # DVC 파이프라인 (prepare → train → evaluate)
├── .pre-commit-config.yaml       # pre-commit 훅 (ruff + 기본 검사)
├── .github/workflows/ci.yml      # GitHub Actions CI
└── .env.example                  # 환경 변수 템플릿
```

---

## Quick Start

### 1. 설치

```bash
git clone https://github.com/dbwjdtn10/GameAI_Analytics.git
cd GameAI_Analytics
pip install ".[dev]"
```

### 2. 모델 학습

```bash
# 기본 학습 (5개 모델: LR, XGBoost, LightGBM, Voting, Stacking)
python scripts/train.py

# Optuna 하이퍼파라미터 튜닝 포함
python scripts/train.py --tune --tune-trials 50

# SHAP 모델 해석
python scripts/evaluate.py

# 유저 세그먼트 (K-Means 4 clusters)
python scripts/segment.py
```

### 3. API 서버

```bash
python -m src.api.main
# http://localhost:8000/docs (Swagger UI)
```

### 4. 대시보드

```bash
streamlit run src/dashboard/app.py
# http://localhost:8501
```

### 5. Docker (전체 서비스)

```bash
docker-compose up --build
```

| 서비스 | URL | 설명 |
|--------|-----|------|
| API | http://localhost:8000 | FastAPI 예측 서버 |
| Dashboard | http://localhost:8501 | Streamlit 대시보드 |
| MLflow | http://localhost:5000 | 실험 추적 UI |
| PostgreSQL | localhost:5432 | 데이터베이스 |

---

## API Endpoints

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/v1/predict/single` | 단일 유저 이탈 예측 |
| `POST` | `/api/v1/predict/batch` | 배치 예측 (최대 1,000명) |
| `POST` | `/api/v1/segment/classify` | 유저 세그먼트 분류 |
| `GET` | `/api/v1/model/info` | 모델 메타 정보 |
| `GET` | `/api/v1/model/features/importance` | 피처 중요도 |
| `GET` | `/health` | 헬스 체크 |

**인증:** `X-API-Key` 헤더 (기본 키: `dev-key-gameai-2024`)

<details>
<summary><b>예측 요청/응답 예시</b></summary>

**Request:**
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

**Response:**
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

</details>

---

## Notebooks

| # | 노트북 | 내용 |
|---|--------|------|
| 01 | EDA - Gaming Behavior | 40,034건 데이터 탐색, 분포 분석, t-test, chi-square 검정 |
| 02 | EDA - Cookie Cats | 90,189건 A/B 테스트 데이터 탐색, 리텐션 분석 |
| 03 | Feature Engineering | 상관관계 히트맵, VIF 다중공선성, Mutual Information |
| 04 | Model Comparison | 5개 모델 ROC/PR 커브, 혼동 행렬 비교 |
| 05 | A/B Test Analysis | Z-test, Bootstrap, Cohen's h 효과 크기 분석 |

> 모든 노트북은 실행 결과(출력, 차트)가 포함되어 있습니다.

---

## Datasets

| 데이터셋 | 출처 | 크기 | 용도 |
|----------|------|------|------|
| Online Gaming Behavior | Kaggle | 40,034 rows | 이탈 예측 (EngagementLevel=Low → churn) |
| Cookie Cats A/B Test | Kaggle | 90,189 rows | A/B 테스트 통계 분석 |
| Synthetic Data | 자체 생성 | 10,000 rows | 시계열 유저 행동 시뮬레이션 |

---

## MLOps Pipeline

```
DVC Pipeline:  prepare ──▶ train ──▶ evaluate
                  │           │          │
                  ▼           ▼          ▼
              CSV 전처리   모델 학습   SHAP 분석

Airflow DAG (Weekly):
  check_drift ──▶ decide ──▶ retrain ──▶ notify
       │                        │
       ▼                        ▼
   KS/PSI 검정          성능 향상 시 자동 배포
```

- **MLflow**: 모든 실험의 파라미터, 메트릭, 아티팩트 자동 기록
- **DVC**: 데이터 + 모델 버전 관리 (3 stage pipeline)
- **Airflow**: 매주 월요일 02:00 드리프트 체크 → 조건부 재학습
- **Drift Detection**: KS test (피처별) + PSI (예측 분포) 기반 모니터링

---

## Testing

```bash
# 전체 테스트 (45개)
pytest tests/ -v

# 린트 검사
ruff check src/ tests/

# 커버리지
pytest tests/ --cov=src --cov-report=term-missing
```

| 테스트 모듈 | 항목 수 | 범위 |
|-------------|---------|------|
| test_api | 9 | API 엔드포인트, 인증, 에러 처리 |
| test_data_quality | 13 | 데이터 스키마, 결측치, 범위 검증 |
| test_drift | 5 | KS test, PSI, 드리프트 판정 |
| test_features | 5 | 피처 생성, 타입, 값 범위 |
| test_model_regression | 4 | AUC≥0.90, Accuracy≥0.85 성능 보장 |
| test_models | 5 | 모델 학습, 앙상블 동작 |
| test_preprocessor | 4 | 전처리 파이프라인 |

> CI 환경에서는 데이터/모델 파일이 없는 테스트를 자동 스킵합니다.

---

## User Segments

K-Means 클러스터링으로 유저를 4개 세그먼트로 분류하고, 세그먼트별 리텐션 전략을 제안합니다.

| 세그먼트 | 특성 | 리텐션 전략 |
|----------|------|-------------|
| Hardcore | 높은 플레이타임, 높은 레벨 | 엔드게임 콘텐츠, 경쟁 리더보드 |
| Casual | 짧은 세션, 낮은 빈도 | 일일 퀘스트, 쉬운 보상 |
| At Risk | 활동량 감소 추세 | 복귀 보상, 개인화 이벤트 |
| New User | 낮은 레벨, 짧은 경력 | 튜토리얼 개선, 초기 보상 |

---

## License

This project is for portfolio/educational purposes.
