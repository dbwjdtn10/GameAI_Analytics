# GameAI Analytics

> 게임 유저 행동 분석 & 이탈 예측 시스템
> End-to-End ML Pipeline for Game Player Churn Prediction

[![CI](https://github.com/dbwjdtn10/GameAI_Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/dbwjdtn10/GameAI_Analytics/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-63%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-pytest--cov-blue.svg)]()

---

## Overview

게임 유저의 플레이 로그를 분석하여 **이탈(churn)을 예측**하고, **유저 세그먼트별 맞춤 리텐션 전략**을 제안하는 프로젝트입니다.
데이터 수집부터 모델 학습, API 서빙, 대시보드, MLOps까지 **프로덕션 수준의 전체 파이프라인**을 구현했습니다.

### Highlights

- **XGBoost 기반 이탈 예측** — AUC-ROC 0.9409, 5개 모델 비교 실험
- **K-Means 유저 세그먼트** — 4개 클러스터별 리텐션 전략 자동 제안
- **SHAP 모델 해석** — 피처 기여도 시각화로 예측 근거 설명
- **FastAPI 실시간 서빙** — 단일/배치 예측 + JWT 인증 + Redis 캐싱 + Rate Limiting
- **Streamlit 대시보드** — 7페이지 (What-If 분석, 비즈니스 임팩트 ROI 포함)
- **Observability** — Prometheus 메트릭 + Grafana 대시보드 + 구조화된 로깅 (structlog)
- **데이터 품질** — Pandera 스키마 검증 + Feature Store (Training-Serving Skew 방지)
- **모델 최적화** — ONNX Runtime 변환 + p50/p95/p99 벤치마크
- **MLOps 파이프라인** — MLflow 실험 추적, DVC 파이프라인, Airflow 재학습, 드리프트 탐지
- **부하 테스트** — Locust 시나리오 (100~1000 동시 유저) + API 벤치마크

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GameAI Analytics                                    │
├──────────────┬───────────────┬────────────────┬─────────────────────────────┤
│  Data Layer  │  ML Pipeline  │  Serving       │  Observability & MLOps      │
├──────────────┼───────────────┼────────────────┼─────────────────────────────┤
│ Kaggle CSV   │ Feature Eng   │ FastAPI        │ Prometheus + Grafana        │
│ PostgreSQL   │ Feature Store │  - JWT Auth    │ Structlog (JSON)            │
│ Pandera      │ XGBoost       │  - Redis Cache │ MLflow Tracking             │
│  Validation  │ ONNX Runtime  │  - Rate Limit  │ DVC Pipeline                │
│              │ LightGBM      │ Streamlit      │ Airflow DAG                 │
│              │ SHAP / Optuna │  - 7 Pages     │ Drift Detection (KS/PSI)    │
│              │ K-Means       │  - What-If     │ Alerting Rules              │
│              │               │  - ROI 분석    │ GitHub Actions CI           │
└──────────────┴───────────────┴────────────────┴─────────────────────────────┘
```

```
유저 데이터 (CSV/DB)
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Pandera     │───▶│  피처 엔지니어링 │───▶│   모델 학습    │
│  데이터 검증  │    │  Feature Store │    │  5개 모델 비교  │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐            │
                    │  SHAP 해석    │◀───────────┤
                    │  ONNX 변환    │            │
                    └──────────────┘            ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Streamlit   │◀───│  FastAPI      │◀───│  Redis 캐시   │
│  7페이지      │    │  + JWT + Rate │    │  + ONNX 서빙  │
└──────────────┘    └──────┬───────┘    └──────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │  Prometheus  │ │  MLflow      │ │  Airflow     │
   │  + Grafana   │ │  실험 추적    │ │  자동 재학습   │
   └──────────────┘ └──────────────┘ └──────────────┘
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
| **데이터 검증** | Pandera | 스키마 검증 (Raw, Engineered, Inference) |
| **피처 관리** | Feature Store (Custom) | Training-Serving Skew 방지, 인코더 일관성 |
| **모델 최적화** | ONNX Runtime, skl2onnx | 추론 속도 최적화, 벤치마크 |
| **API** | FastAPI, Pydantic | 실시간 예측 서빙 |
| **인증** | JWT (python-jose), API Key | 토큰 기반 인증 + API Key 병행 |
| **캐싱** | Redis | 중복 추론 방지, 응답 속도 향상 |
| **Rate Limiting** | slowapi | API 요청 제한 (DDoS 방지) |
| **대시보드** | Streamlit, Plotly | 7페이지 인터랙티브 시각화 |
| **DB** | PostgreSQL, SQLAlchemy | 유저 데이터 적재 + SQL 분석 |
| **로깅** | structlog | JSON 구조화 로깅, request_id 추적 |
| **메트릭** | Prometheus, Grafana | 실시간 메트릭 수집 + 대시보드 |
| **알림** | Prometheus Alertmanager | 고장애율, 높은 레이턴시, 드리프트 알림 |
| **MLOps** | MLflow, DVC, Airflow | 실험 추적, 데이터 버전, 재학습 |
| **모니터링** | KS test, PSI | 데이터/예측 드리프트 탐지 |
| **인프라** | Docker, docker-compose | 7 서비스 컨테이너 오케스트레이션 |
| **CI/CD** | GitHub Actions, pre-commit | 린트 + 테스트 + 커버리지 + 모델 게이트 + Docker 빌드 |
| **테스트** | pytest (63 tests), Ruff | 품질 보증 |
| **부하 테스트** | Locust | 동시 사용자 부하 테스트 시나리오 |

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
├── src/                              # 소스 코드
│   ├── config.py                     # 프로젝트 설정 (경로, Redis, JWT, Rate Limit)
│   ├── api/                          # FastAPI 서버
│   │   ├── main.py                   #   앱 진입점 + Prometheus + CORS + Rate Limit
│   │   ├── schemas.py                #   Pydantic 요청/응답 모델
│   │   ├── dependencies.py           #   ModelService + API Key 인증
│   │   ├── middleware.py             #   structlog 로깅 + request_id 미들웨어
│   │   ├── auth_jwt.py               #   JWT 토큰 발급/검증
│   │   ├── cache.py                  #   Redis 비동기 캐싱 레이어
│   │   └── routes/                   #   라우터 모듈
│   │       ├── predict.py            #     단일/배치 예측 + 캐싱 + 메트릭
│   │       ├── segment.py            #     유저 세그먼트 분류
│   │       ├── model_info.py         #     모델 정보/피처 중요도
│   │       └── health.py             #     헬스 체크 (모델 + Redis 상태)
│   ├── dashboard/                    # Streamlit 대시보드
│   │   └── app.py                    #   7페이지 (개요/모델/세그먼트/모니터링/예측/What-If/ROI)
│   ├── data/                         # 데이터 레이어
│   │   ├── loader.py                 #   CSV 데이터 로딩
│   │   ├── preprocessor.py           #   전처리 파이프라인
│   │   ├── validation.py             #   Pandera 스키마 검증 (3종)
│   │   ├── synthetic.py              #   합성 데이터 생성기
│   │   └── database.py               #   PostgreSQL 연동
│   ├── features/                     # 피처 엔지니어링
│   │   ├── engineer.py               #   12개 파생 피처 생성
│   │   ├── store.py                  #   Feature Store (인코더 일관성 보장)
│   │   └── selector.py               #   피처 선택 (MI, VIF)
│   ├── models/                       # ML 모델
│   │   ├── trainer.py                #   5개 모델 + Optuna 튜닝
│   │   ├── evaluator.py              #   평가 리포트 + 시각화
│   │   ├── onnx_converter.py         #   ONNX 변환 + 추론 벤치마크
│   │   ├── registry.py               #   MLflow 실험 기록
│   │   └── segmenter.py              #   K-Means 세그먼트
│   └── monitoring/                   # 모니터링 & Observability
│       ├── drift.py                  #   KS test + PSI 드리프트 탐지
│       ├── metrics.py                #   Prometheus 커스텀 메트릭 (10종)
│       └── logging_config.py         #   structlog 구조화 로깅 설정
│
├── scripts/                          # 실행 스크립트
│   ├── train.py                      #   모델 학습 파이프라인
│   ├── evaluate.py                   #   SHAP 분석
│   ├── segment.py                    #   세그먼트 클러스터링
│   ├── benchmark.py                  #   API 응답 시간 벤치마크 (p50/p95/p99)
│   ├── generate_synthetic.py         #   합성 데이터 생성
│   └── load_to_postgres.py           #   PostgreSQL 적재
│
├── notebooks/                        # 분석 노트북 (출력 포함)
│   ├── 01_eda_gaming_behavior.ipynb  #   EDA + 통계 검정
│   ├── 02_eda_cookie_cats.ipynb      #   A/B 테스트 EDA
│   ├── 03_feature_engineering.ipynb  #   피처 분석 (상관관계, VIF, MI)
│   ├── 04_model_comparison.ipynb     #   모델 성능 비교
│   ├── 05_ab_test_analysis.ipynb     #   A/B 테스트 통계 분석
│   └── 06_business_impact.ipynb      #   비즈니스 임팩트 (LTV, ROI, 임계값 최적화)
│
├── tests/                            # 테스트 (63개)
│   ├── test_api.py                   #   API 엔드포인트 (9 tests)
│   ├── test_auth_jwt.py              #   JWT 인증 (4 tests)
│   ├── test_validation.py            #   Pandera 데이터 검증 (6 tests)
│   ├── test_feature_store.py         #   Feature Store (4 tests)
│   ├── test_metrics.py               #   Prometheus + 헬스체크 (4 tests)
│   ├── test_data_quality.py          #   데이터 품질 (13 tests)
│   ├── test_drift.py                 #   드리프트 탐지 (5 tests)
│   ├── test_features.py              #   피처 엔지니어링 (5 tests)
│   ├── test_model_regression.py      #   모델 성능 회귀 (4 tests)
│   ├── test_models.py                #   모델 학습 (5 tests)
│   └── test_preprocessor.py          #   전처리 (4 tests)
│
├── grafana/                          # Grafana 설정
│   ├── dashboards/gameai.json        #   ML 모니터링 대시보드 (10패널)
│   └── provisioning/                 #   자동 프로비저닝 (데이터소스 + 대시보드)
│
├── sql/                              # SQL
│   ├── schema.sql                    #   DB 스키마 (players, predictions, model_registry)
│   └── queries.sql                   #   분석 쿼리 (이탈률, 세그먼트, 코호트)
│
├── dags/                             # Airflow
│   └── retrain_dag.py                #   주간 재학습 DAG (드리프트 → 학습 → 배포)
│
├── Dockerfile                        # API 컨테이너
├── docker-compose.yml                # 7 서비스 (API/Dashboard/MLflow/PostgreSQL/Redis/Prometheus/Grafana)
├── Makefile                          # 편의 명령어 (make train, make test, make up 등)
├── pyproject.toml                    # 프로젝트 메타데이터 + 의존성
├── locustfile.py                     # Locust 부하 테스트 시나리오
├── prometheus.yml                    # Prometheus 스크랩 설정
├── alerts.yml                        # Prometheus 알림 규칙
├── dvc.yaml                          # DVC 파이프라인 (validate → prepare → train → evaluate)
├── .pre-commit-config.yaml           # pre-commit 훅 (ruff + 기본 검사)
├── .github/workflows/ci.yml          # GitHub Actions CI (린트 + 테스트 + 모델 게이트 + Docker)
└── .env.example                      # 환경 변수 템플릿
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
make train                    # 기본 학습 (5개 모델)
make train-tune               # Optuna 하이퍼파라미터 튜닝 포함
make evaluate                 # SHAP 모델 해석
make segment                  # 유저 세그먼트 (K-Means)
```

### 3. API 서버

```bash
make api
# http://localhost:8000/docs    (Swagger UI)
# http://localhost:8000/metrics (Prometheus 메트릭)
```

### 4. 대시보드

```bash
make dashboard
# http://localhost:8501
```

### 5. Docker (전체 서비스)

```bash
make up                       # 7 서비스 기동
make down                     # 종료
make logs                     # 로그 확인
```

| 서비스 | URL | 설명 |
|--------|-----|------|
| API | http://localhost:8000 | FastAPI 예측 서버 |
| Dashboard | http://localhost:8501 | Streamlit 대시보드 (7페이지) |
| MLflow | http://localhost:5000 | 실험 추적 UI |
| Grafana | http://localhost:3000 | ML 모니터링 대시보드 (admin/gameai2024) |
| Prometheus | http://localhost:9090 | 메트릭 수집 UI |
| PostgreSQL | localhost:5432 | 데이터베이스 |
| Redis | localhost:6379 | 예측 캐시 |

### 6. 테스트 & 품질 검사

```bash
make test                     # 전체 테스트 (63개)
make lint                     # Ruff 린트
make coverage                 # 테스트 커버리지
make benchmark                # API 응답 시간 벤치마크
```

---

## API Endpoints

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/v1/predict/single` | 단일 유저 이탈 예측 (Redis 캐싱) |
| `POST` | `/api/v1/predict/batch` | 배치 예측 (최대 1,000명) |
| `POST` | `/api/v1/segment/classify` | 유저 세그먼트 분류 |
| `GET` | `/api/v1/model/info` | 모델 메타 정보 |
| `GET` | `/api/v1/model/features/importance` | 피처 중요도 |
| `POST` | `/auth/token` | JWT 토큰 발급 |
| `GET` | `/health` | 헬스 체크 (모델 + Redis 상태) |
| `GET` | `/metrics` | Prometheus 메트릭 |

**인증 (2가지 방식 병행):**
- API Key: `X-API-Key` 헤더 (기본 키: `dev-key-gameai-2024`)
- JWT: `Authorization: Bearer <token>` (발급: `POST /auth/token`)

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

<details>
<summary><b>JWT 토큰 발급 예시</b></summary>

```bash
# 토큰 발급
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=gameai2024"

# 응답: {"access_token": "eyJ...", "token_type": "bearer", "expires_in": 1800}
```

</details>

---

## Observability

### Prometheus 메트릭 (10종)

| 메트릭 | 타입 | 설명 |
|--------|------|------|
| `gameai_prediction_requests_total` | Counter | 예측 요청 수 (엔드포인트별, 리스크별) |
| `gameai_prediction_latency_seconds` | Histogram | 모델 추론 레이턴시 (p50/p95/p99) |
| `gameai_prediction_errors_total` | Counter | 예측 에러 수 |
| `gameai_model_loaded` | Gauge | 모델 로드 상태 (1=loaded) |
| `gameai_model_inference_total` | Counter | 모델 추론 횟수 |
| `gameai_cache_hits_total` | Counter | Redis 캐시 히트 수 |
| `gameai_cache_misses_total` | Counter | Redis 캐시 미스 수 |
| `gameai_high_risk_users_total` | Counter | 고위험 유저 탐지 수 |
| `gameai_batch_prediction_size` | Histogram | 배치 예측 크기 분포 |
| `gameai_drift_ratio` | Gauge | 데이터 드리프트 비율 |

### Grafana 대시보드

10개 패널로 구성된 ML 모니터링 대시보드:
- 예측 요청률 (req/min) / 추론 레이턴시 (p50, p95)
- 고위험 유저 탐지 현황 / 모델 상태
- 캐시 히트율 / 에러율
- HTTP 요청 분포 / 배치 크기 분포
- 모델 추론 처리량 / 데이터 드리프트 게이지

### Alerting Rules

| 알림 | 조건 | 심각도 |
|------|------|--------|
| HighErrorRate | 에러율 > 5% (5분) | critical |
| HighLatency | p95 > 1초 (5분) | warning |
| ModelDown | 모델 미로드 (1분) | critical |
| HighDrift | 드리프트 > 50% | warning |
| CacheDown | 캐시 히트율 = 0 (5분) | warning |

---

## Data Validation

Pandera 기반 3단계 데이터 검증:

| 스키마 | 적용 시점 | 검증 내용 |
|--------|-----------|-----------|
| `RawGamingBehaviorSchema` | 데이터 로드 | 타입, 범위, 허용값, 중복률 < 5% |
| `EngineeredFeaturesSchema` | 피처 생성 후 | 파생 피처 존재, 양수, 타겟 값 검증 |
| `InferenceInputSchema` | API 추론 입력 | 입력 범위, 필수 필드 검증 |

DVC 파이프라인에 `validate` 스테이지로 통합:
```
validate → prepare → train → evaluate
```

---

## Feature Store

경량 Feature Store로 Training-Serving Skew를 방지합니다:

```
feature_store/
├── metadata.json          # 피처 메타데이터 (타입, 스키마 해시)
├── feature_stats.json     # 학습 데이터 통계 (mean, std, min, max, IQR)
└── encoders/              # 범주형 인코더 매핑 (학습 시 저장 → 서빙 시 재사용)
    ├── Gender.json
    ├── Location.json
    ├── GameGenre.json
    └── GameDifficulty.json
```

- **학습 시**: `register_training_data()` → 피처 통계 + 인코더 매핑 저장
- **서빙 시**: `transform_for_serving()` → 저장된 인코더로 동일한 변환 보장
- **검증**: `validate_serving_data()` → 입력 데이터가 학습 분포에서 벗어나는지 체크

---

## Model Optimization (ONNX)

XGBoost/LightGBM 모델을 ONNX Runtime으로 변환하여 추론 속도를 최적화합니다.

```bash
# ONNX 변환 + 벤치마크
python src/models/onnx_converter.py

# API 엔드포인트 벤치마크 (p50/p95/p99)
python scripts/benchmark.py --url http://localhost:8000 --n-requests 200
```

벤치마크 출력 예시:
```
[Single Prediction] POST /api/v1/predict/single
  p50: 12.34ms | p95: 25.67ms | p99: 45.12ms
  Mean: 15.23ms | Throughput: 65.7 req/s
```

---

## Load Testing

Locust 기반 4가지 시나리오로 부하 테스트:

```bash
# 웹 UI 모드
locust -f locustfile.py --host http://localhost:8000
# http://localhost:8089 에서 유저 수/초당 생성 설정

# CLI 모드 (100 유저, 30초)
locust -f locustfile.py --host http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 30s --headless
```

| 시나리오 | 비율 | 설명 |
|----------|------|------|
| 단일 예측 | 70% | `/predict/single` 랜덤 유저 |
| 배치 예측 | 15% | `/predict/batch` 10~50명 |
| 세그먼트 분류 | 10% | `/segment/classify` |
| 모델 정보 | 5% | `/model/info` |

---

## Dashboard (7 Pages)

| 페이지 | 내용 |
|--------|------|
| 개요 | KPI 카드 (유저 수, 이탈률), 장르/난이도별 이탈률, 피처 분포 |
| 모델 성능 | ROC/PR 커브, 혼동 행렬, 피처 중요도, SHAP 분석 |
| 유저 세그먼트 | K-Means 분포, 세그먼트별 이탈률, 리텐션 전략 카드 |
| 모니터링 | KS test 드리프트, PSI, 예측 분포 비교 |
| 이탈 예측 | 유저 입력 → 실시간 이탈 확률 + 게이지 차트 |
| **What-If 분석** | 피처 변경 시 이탈 확률 변화 시뮬레이션, Partial Dependence Plot |
| **비즈니스 임팩트** | 임계값 최적화, ROI 계산, 연간 순이익 추정, 세그먼트별 비용-편익 |

---

## Notebooks

| # | 노트북 | 내용 |
|---|--------|------|
| 01 | EDA - Gaming Behavior | 40,034건 데이터 탐색, 분포 분석, t-test, chi-square 검정 |
| 02 | EDA - Cookie Cats | 90,189건 A/B 테스트 데이터 탐색, 리텐션 분석 |
| 03 | Feature Engineering | 상관관계 히트맵, VIF 다중공선성, Mutual Information |
| 04 | Model Comparison | 5개 모델 ROC/PR 커브, 혼동 행렬 비교 |
| 05 | A/B Test Analysis | Z-test, Bootstrap, Cohen's h 효과 크기 분석 |
| **06** | **Business Impact** | **LTV 추정, 임계값 최적화, 세그먼트별 ROI, 민감도 분석** |

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
DVC Pipeline:  validate ──▶ prepare ──▶ train ──▶ evaluate
                  │            │           │          │
                  ▼            ▼           ▼          ▼
             Pandera 검증  CSV 전처리   모델 학습   SHAP 분석

Airflow DAG (Weekly):
  check_drift ──▶ decide ──▶ retrain ──▶ notify
       │                        │
       ▼                        ▼
   KS/PSI 검정          성능 향상 시 자동 배포
```

- **MLflow**: 모든 실험의 파라미터, 메트릭, 아티팩트 자동 기록
- **DVC**: 데이터 + 모델 버전 관리 (4 stage pipeline)
- **Airflow**: 매주 월요일 02:00 드리프트 체크 → 조건부 재학습
- **Drift Detection**: KS test (피처별) + PSI (예측 분포) 기반 모니터링
- **Prometheus Alerting**: 에러율/레이턴시/드리프트 기반 자동 알림

---

## Testing

```bash
make test                     # 전체 테스트 (63개)
make lint                     # Ruff 린트
make coverage                 # 테스트 커버리지 리포트
```

| 테스트 모듈 | 항목 수 | 범위 |
|-------------|---------|------|
| test_api | 9 | API 엔드포인트, 인증, 에러 처리 |
| test_auth_jwt | 4 | JWT 토큰 발급, 검증, 페이로드 |
| test_validation | 6 | Pandera 스키마 검증 (유효/무효 데이터) |
| test_feature_store | 4 | Feature Store 등록, 로드, 변환, 일관성 |
| test_metrics | 4 | Prometheus 메트릭, 헬스체크, 헤더 |
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
