# Architecture

GameAI Analytics 시스템 아키텍처 상세 문서

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐    │
│  │  curl / SDK  │  │  Streamlit UI    │  │  Swagger UI (/docs)         │    │
│  │              │  │  (7 pages)       │  │  Grafana (:3000)            │    │
│  └──────┬──────┘  └────────┬─────────┘  └──────────────┬──────────────┘    │
└─────────┼──────────────────┼───────────────────────────┼───────────────────┘
          │                  │                           │
          ▼                  ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                                    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ structlog    │  │ API Key Auth │  │ JWT Auth     │  │ CORS         │   │
│  │ Middleware   │  │ (X-API-Key)  │  │ (Bearer)     │  │ Middleware   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│  ┌──────────────┐  ┌──────────────┐                                       │
│  │ Prometheus   │  │ Rate Limiter │                                       │
│  │ Instrumentor │  │ (slowapi)    │                                       │
│  └──────────────┘  └──────────────┘                                       │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ /predict     │  │ /segment     │  │ /model       │  │ /health      │   │
│  │  single      │  │  classify    │  │  info        │  │ (components) │   │
│  │  batch       │  │              │  │  importance  │  │              │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────────┘   │
│         │                 │                  │                              │
│  ┌──────▼─────────────────▼──────────────────▼──────────────────────┐      │
│  │              Redis Cache Layer (async)                            │      │
│  │  - 동일 입력 캐싱 (TTL 5min)  - graceful fallback               │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└─────────┼──────────────────┼──────────────────┼────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Service Layer                                         │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  ModelService (src/api/dependencies.py)                            │     │
│  │  - predict(): 확률 예측 + 리스크 레벨 + 추천 액션                    │     │
│  │  - _get_risk_factors(): SHAP 기반 Top-5 리스크 요인                │     │
│  │  - _get_recommended_actions(): 리스크 레벨별 맞춤 전략              │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Trainer      │  │ Segmenter    │  │ DriftDetector│  │ FeatureStore │   │
│  │ (5 models)   │  │ (K-Means)    │  │ (KS + PSI)  │  │ (encoders)   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐                        │
│  │ ONNX Converter       │  │ Pandera Validator    │                        │
│  │ (XGBoost → ONNX)     │  │ (3 schemas)          │                        │
│  └──────────────────────┘  └──────────────────────┘                        │
└─────────┼──────────────────┼──────────────────┼────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Data Layer                                            │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ CSV Files    │  │ PostgreSQL   │  │ MLflow DB    │  │ Model Files  │   │
│  │ (raw/proc)   │  │ (players,   │  │ (sqlite)     │  │ (joblib/onnx)│   │
│  │              │  │  predictions)│  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐                                        │
│  │ Redis        │  │ Feature Store│                                        │
│  │ (cache)      │  │ (JSON files) │                                        │
│  └──────────────┘  └──────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Observability Layer                                       │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Prometheus   │  │ Grafana      │  │ structlog    │  │ Alerting     │   │
│  │ (메트릭 수집) │  │ (대시보드)   │  │ (JSON 로깅)   │  │ (알림 규칙)   │   │
│  │ :9090        │  │ :3000        │  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### 1. Data Module (`src/data/`)

데이터 수집, 검증, 전처리, 저장을 담당합니다.

```
src/data/
├── loader.py        # 데이터 로딩 (CSV → DataFrame)
├── validation.py    # Pandera 스키마 검증 (3종)
├── preprocessor.py  # 전처리 파이프라인 (결측치, 인코딩, 스케일링)
├── synthetic.py     # 합성 데이터 생성 (시계열 유저 행동 시뮬레이션)
└── database.py      # PostgreSQL 연동 (SQLAlchemy)
```

**데이터 흐름:**
```
Kaggle CSV ──▶ loader.py ──▶ validation.py ──▶ preprocessor.py ──▶ engineer.py ──▶ 학습 데이터
                  │              (Pandera)
                  ▼
             database.py ──▶ PostgreSQL (players 테이블)
```

**Pandera 검증 스키마 (3단계):**

| 스키마 | 적용 시점 | 검증 내용 |
|--------|-----------|-----------|
| `RawGamingBehaviorSchema` | 데이터 로드 | 타입, 범위, 허용값, 중복률 < 5% |
| `EngineeredFeaturesSchema` | 피처 생성 후 | 파생 피처 존재, 양수, 타겟 값 검증 |
| `InferenceInputSchema` | API 추론 입력 | 입력 범위, 필수 필드 검증 |

**이탈 정의:**
- `EngagementLevel == "Low"` → `is_churned = 1` (이탈)
- `EngagementLevel != "Low"` → `is_churned = 0` (유지)

### 2. Feature Module (`src/features/`)

12개 파생 피처 생성 + Feature Store로 Training-Serving 일관성 보장

```
src/features/
├── engineer.py  # 피처 생성
├── store.py     # Feature Store (인코더 매핑 저장/로드)
└── selector.py  # 피처 선택 (Mutual Information, VIF)
```

**Feature Store 구조:**
```
feature_store/
├── metadata.json          # 피처 메타데이터 (타입, 스키마 해시, 샘플 수)
├── feature_stats.json     # 학습 데이터 통계 (mean, std, min, max, Q25, Q75)
└── encoders/              # 범주형 인코더 매핑
    ├── Gender.json        #   {"Female": 0, "Male": 1}
    ├── Location.json      #   {"Africa": 0, "Asia": 1, ...}
    ├── GameGenre.json     #   {"Action": 0, "Adventure": 1, ...}
    └── GameDifficulty.json
```

- 학습 시 `register_training_data()` → 인코더 매핑 저장
- 서빙 시 `transform_for_serving()` → 저장된 매핑으로 동일한 변환

**파생 피처 목록:**

| 피처 | 수식 | 의미 |
|------|------|------|
| `playtime_per_session` | PlayTimeHours / SessionsPerWeek | 세션당 플레이 강도 |
| `weekly_activity_intensity` | SessionsPerWeek * AvgSessionDurationMinutes | 주간 총 활동 시간 |
| `session_engagement_score` | AvgSessionDuration / global_mean | 세션 몰입도 |
| `level_efficiency` | PlayerLevel / PlayTimeHours | 시간당 레벨업 속도 |
| `achievement_rate` | AchievementsUnlocked / PlayerLevel | 레벨 대비 달성률 |
| `purchase_per_hour` | InGamePurchases / PlayTimeHours | 시간당 결제 비율 |
| `activity_score` | 가중 합산 (0.3/0.3/0.2/0.2) | 종합 활동 점수 |
| `age_group` | 구간화 (teen/young_adult/adult/middle/senior) | 연령대 |

### 3. Model Module (`src/models/`)

5개 모델 학습, 평가, ONNX 변환, 세그먼트를 담당합니다.

```
src/models/
├── trainer.py         # 모델 정의 + Optuna 튜닝
├── evaluator.py       # 메트릭 계산 + 시각화 (ROC, CM, Feature Importance)
├── onnx_converter.py  # ONNX 변환 + 추론 벤치마크
├── registry.py        # MLflow 실험 기록
└── segmenter.py       # K-Means 유저 세그먼트
```

**모델 파이프라인:**
```
                    ┌─ LogisticRegression (baseline)
                    │
                    ├─ XGBoost ──────────┐
X_train, y_train ──▶├─ LightGBM ─────────┼──▶ VotingEnsemble
                    │                    │
                    │                    └──▶ StackingEnsemble
                    │
                    └────▶ 비교 → Best 저장 (joblib) → ONNX 변환 (선택)
```

**ONNX 추론 최적화:**
- `convert_model_to_onnx()`: XGBoost/LightGBM → ONNX 변환
- `ONNXModelService`: ONNX Runtime 기반 추론 (graph optimization 적용)
- `benchmark_inference()`: joblib vs ONNX p50/p95/p99 비교

### 4. API Module (`src/api/`)

FastAPI 기반 REST API 서버 (인증, 캐싱, 메트릭 포함)

```
src/api/
├── main.py           # 앱 생성 + Prometheus + CORS + Rate Limit + JWT
├── schemas.py        # Pydantic 모델 (요청/응답 스키마)
├── dependencies.py   # ModelService (싱글톤), API Key 인증
├── middleware.py      # structlog 로깅 + request_id 추적
├── auth_jwt.py       # JWT 토큰 발급/검증 (OAuth2PasswordBearer)
├── cache.py          # Redis 비동기 캐싱 (graceful fallback)
└── routes/
    ├── predict.py    # POST /predict/single (캐싱), POST /predict/batch
    ├── segment.py    # POST /segment/classify
    ├── model_info.py # GET /model/info, GET /model/features/importance
    └── health.py     # GET /health (모델 + Redis 상태)
```

**요청 처리 흐름:**
```
Client Request
    │
    ▼
structlog Middleware (request_id 발급, 처리 시간 측정)
    │
    ▼
Rate Limiter (slowapi, 60/min 기본)
    │
    ▼
인증 (API Key 또는 JWT Bearer Token)
    │
    ▼
Pydantic 스키마 검증 (PlayerFeatures)
    │
    ▼
Redis 캐시 조회 ──▶ [HIT] → 즉시 응답
    │
    │ [MISS]
    ▼
ModelService.predict()
    ├── 피처 엔지니어링 (동적 생성)
    ├── 모델 추론 (predict_proba)
    ├── 리스크 레벨 분류 (low/medium/high/critical)
    ├── Top-5 리스크 요인 추출
    └── 리스크별 추천 액션 매핑
    │
    ▼
Redis 캐시 저장 (TTL 5분)
    │
    ▼
Prometheus 메트릭 기록 (latency, count, risk_level)
    │
    ▼
PredictionResponse (JSON) + X-Request-ID + X-Process-Time-Ms
```

### 5. Dashboard Module (`src/dashboard/`)

Streamlit + Plotly 기반 7페이지 인터랙티브 대시보드

| 페이지 | 내용 |
|--------|------|
| 개요 | KPI 카드, 장르/난이도별 이탈률, 피처 분포 |
| 모델 성능 | ROC/PR 커브, 혼동 행렬, 피처 중요도, SHAP |
| 유저 세그먼트 | K-Means 분포, 세그먼트별 이탈률, 리텐션 전략 |
| 모니터링 | KS test 드리프트, PSI, 예측 분포 비교 |
| 이탈 예측 | 입력 폼 → 실시간 확률 + 게이지 |
| **What-If 분석** | 피처 슬라이더 → 확률 변화, Partial Dependence Plot |
| **비즈니스 임팩트** | 임계값 최적화, ROI 계산, 연간 순이익 추정 |

### 6. Monitoring & Observability Module (`src/monitoring/`)

```
src/monitoring/
├── drift.py           # KS test + PSI 드리프트 탐지
├── metrics.py         # Prometheus 커스텀 메트릭 정의 (10종)
└── logging_config.py  # structlog 구조화 로깅 설정
```

**Prometheus 메트릭 (10종):**

| 메트릭 | 타입 | 라벨 |
|--------|------|------|
| `gameai_prediction_requests_total` | Counter | endpoint, risk_level |
| `gameai_prediction_latency_seconds` | Histogram | endpoint |
| `gameai_prediction_errors_total` | Counter | endpoint, error_type |
| `gameai_model_loaded` | Gauge | - |
| `gameai_model_inference_total` | Counter | model_type |
| `gameai_cache_hits_total` | Counter | - |
| `gameai_cache_misses_total` | Counter | - |
| `gameai_high_risk_users_total` | Counter | risk_level |
| `gameai_batch_prediction_size` | Histogram | - |
| `gameai_drift_ratio` | Gauge | - |

---

## Infrastructure

### Docker Compose 구성 (7 서비스)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  docker-compose.yml                                                      │
│                                                                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────────────────┐  │
│  │ PostgreSQL │  │ Redis      │  │ API (FastAPI)                    │  │
│  │ :5432      │  │ :6379      │  │ :8000                            │  │
│  │            │  │            │  │ + Prometheus metrics              │  │
│  │ - players  │  │ - cache    │  │ + JWT + Rate Limit + CORS        │  │
│  │ - predict  │  │            │  │ depends: postgres, redis         │  │
│  │ - registry │  │            │  │ healthcheck: /health             │  │
│  └────────────┘  └────────────┘  └──────────┬───────────────────────┘  │
│                                              │                          │
│  ┌────────────┐               ┌──────────────▼─────────┐               │
│  │ MLflow UI  │               │ Dashboard (Streamlit)  │               │
│  │ :5000      │               │ :8501 (7 pages)        │               │
│  │ - sqlite   │               │ depends: api           │               │
│  └────────────┘               └────────────────────────┘               │
│                                                                          │
│  ┌────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │ Prometheus                 │  │ Grafana                          │  │
│  │ :9090                      │  │ :3000 (admin/gameai2024)         │  │
│  │ scrape: api:8000/metrics   │  │ datasource: prometheus:9090      │  │
│  │ alerts: alerts.yml         │  │ dashboard: ML Monitoring (10p)   │  │
│  └────────────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### CI/CD Pipeline (3 Jobs)

```
Push / PR to main
    │
    ▼
┌─ Job 1: lint-and-test ──────────────────────┐
│  ├── Setup Python 3.12                       │
│  ├── Cache pip dependencies                  │
│  ├── Install dependencies                    │
│  ├── Ruff lint check (src/ tests/)           │
│  └── pytest + coverage (63 tests)            │
└──────────────┬───────────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼                     ▼
┌─ Job 2: model-validation ──┐  ┌─ Job 3: docker-build ──────────┐
│  (PR only)                  │  │  (main branch only)             │
│  ├── Data validation        │  │  ├── Setup Docker Buildx        │
│  │   (Pandera schemas)      │  │  └── Build + cache (GHA)        │
│  └── Model performance gate │  └─────────────────────────────────┘
│      (AUC, accuracy check)  │
└─────────────────────────────┘
```

### DVC Pipeline (4 Stages)

```
validate ──────────▶ prepare ──────────▶ train ──────────▶ evaluate
   │                    │                  │                  │
   │ deps:              │ deps:            │ deps:            │ deps:
   │ - loader.py        │ - loader.py      │ - train.py       │ - evaluate.py
   │ - validation.py    │ - engineer.py    │ - trainer.py     │ - best_model.joblib
   │ - raw CSV          │ - validation.py  │ - features CSV   │
   │                    │ - raw CSV        │                  │ plots:
   │ (Pandera check)    │                  │ outs:            │ - shap_summary.png
   │                    │ outs:            │ - model.joblib   │ - shap_bar.png
   │                    │ - features.csv   │ plots:           │ - shap_waterfall.png
   │                    │                  │ - roc.png        │
   │                    │                  │ - cm.png         │
   │                    │                  │ - importance.png │
```

---

## Database Schema

```sql
players               predictions            model_registry
┌──────────────┐     ┌──────────────────┐    ┌──────────────────┐
│ id (PK)      │     │ id (PK)          │    │ id (PK)          │
│ age          │     │ player_id (FK)   │    │ model_name       │
│ gender       │     │ churn_proba      │    │ version          │
│ location     │     │ churn_prediction │    │ auc_roc          │
│ game_genre   │     │ risk_level       │    │ accuracy         │
│ play_time    │     │ segment          │    │ f1_score         │
│ sessions/wk  │     │ model_version    │    │ is_active        │
│ avg_session  │     │ created_at       │    │ created_at       │
│ player_level │     └──────────────────┘    └──────────────────┘
│ achievements │
│ purchases    │
│ is_churned   │
│ created_at   │
└──────────────┘
```

---

## Key Design Decisions

| 결정 | 이유 |
|------|------|
| EngagementLevel=Low를 이탈로 정의 | 데이터셋에 직접적인 이탈 레이블이 없어 Low engagement를 proxy로 사용 |
| XGBoost를 최종 모델로 선택 | AUC-ROC 0.9409로 최고 성능, 앙상블 대비 단일 모델이 해석/서빙에 유리 |
| K-Means 4 clusters | Elbow method + 도메인 지식으로 4개 세그먼트가 가장 해석 가능 |
| API Key + JWT 병행 인증 | 간단한 개발용 API Key + 프로덕션 수준 JWT (역할 기반) |
| Redis 캐싱 (graceful fallback) | Redis 미사용 시에도 서비스 정상 동작, 성능 향상은 보너스 |
| Pandera 데이터 검증 | 학습/추론 양쪽에서 데이터 품질 보장, DVC 파이프라인 통합 |
| Feature Store (커스텀 경량) | Feast 등 무거운 도구 대신 JSON 기반으로 Training-Serving Skew 방지 |
| ONNX 선택적 사용 | joblib 기본 + USE_ONNX=true 시 ONNX Runtime 전환 |
| structlog JSON 로깅 | request_id 추적, 구조화된 로그로 운영 편의성 향상 |
| Prometheus + Grafana | 실시간 메트릭 + 시각화 + 알림 규칙으로 완전한 Observability |
| pre-commit + GitHub Actions | 로컬 린트(pre-commit) + CI 검증(Actions) 이중 품질 게이트 |
