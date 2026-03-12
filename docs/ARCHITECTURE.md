# Architecture

GameAI Analytics 시스템 아키텍처 상세 문서

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                   │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐    │
│  │  curl / SDK  │  │  Streamlit UI    │  │  Swagger UI (/docs)         │    │
│  └──────┬──────┘  └────────┬─────────┘  └──────────────┬──────────────┘    │
└─────────┼──────────────────┼───────────────────────────┼──────────────────-─┘
          │                  │                           │
          ▼                  ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API Layer (FastAPI)                                │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ RequestLogging   │  │ API Key Auth │  │ CORS         │                  │
│  │ Middleware       │  │ (X-API-Key)  │  │ Middleware   │                  │
│  └──────────────────┘  └──────────────┘  └──────────────┘                  │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ /predict     │  │ /segment     │  │ /model       │  │ /health      │   │
│  │  single      │  │  classify    │  │  info        │  │              │   │
│  │  batch       │  │              │  │  importance  │  │              │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────────┘   │
└─────────┼──────────────────┼──────────────────┼───────────────────────────-─┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Service Layer                                        │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  ModelService (src/api/dependencies.py)                            │     │
│  │  - predict(): 확률 예측 + 리스크 레벨 + 추천 액션                    │     │
│  │  - _get_risk_factors(): SHAP 기반 Top-5 리스크 요인                │     │
│  │  - _get_recommended_actions(): 리스크 레벨별 맞춤 전략              │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐      │
│  │ Trainer          │  │ Segmenter        │  │ DriftDetector        │      │
│  │ (5 models)       │  │ (K-Means)        │  │ (KS test + PSI)     │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Data Layer                                           │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ CSV Files    │  │ PostgreSQL   │  │ MLflow DB    │  │ Model Files  │   │
│  │ (raw/proc)   │  │ (players,   │  │ (sqlite)     │  │ (joblib)     │   │
│  │              │  │  predictions)│  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### 1. Data Module (`src/data/`)

데이터 수집, 전처리, 저장을 담당합니다.

```
src/data/
├── loader.py        # 데이터 로딩 (CSV → DataFrame)
├── preprocessor.py  # 전처리 파이프라인 (결측치, 인코딩, 스케일링)
├── synthetic.py     # 합성 데이터 생성 (시계열 유저 행동 시뮬레이션)
└── database.py      # PostgreSQL 연동 (SQLAlchemy)
```

**데이터 흐름:**
```
Kaggle CSV ──▶ loader.py ──▶ preprocessor.py ──▶ engineer.py ──▶ 학습 데이터
                  │
                  ▼
             database.py ──▶ PostgreSQL (players 테이블)
```

**이탈 정의:**
- `EngagementLevel == "Low"` → `is_churned = 1` (이탈)
- `EngagementLevel != "Low"` → `is_churned = 0` (유지)

### 2. Feature Module (`src/features/`)

12개의 파생 피처를 생성하고, 피처 중요도를 분석합니다.

```
src/features/
├── engineer.py  # 피처 생성
└── selector.py  # 피처 선택 (Mutual Information, VIF)
```

**파생 피처 목록:**

| 피처 | 수식 | 의미 |
|------|------|------|
| `play_intensity` | PlayTimeHours / SessionsPerWeek | 세션당 플레이 강도 |
| `session_efficiency` | AchievementsUnlocked / AvgSessionDurationMinutes | 세션 효율성 |
| `achievement_rate` | AchievementsUnlocked / PlayerLevel | 레벨 대비 달성률 |
| `purchase_per_hour` | InGamePurchases / PlayTimeHours | 시간당 결제 비율 |
| `engagement_index` | SessionsPerWeek * AvgSessionDurationMinutes | 주간 총 플레이 시간 |
| `level_per_hour` | PlayerLevel / PlayTimeHours | 시간당 레벨업 속도 |
| `activity_score` | 정규화 종합 | 종합 활동 점수 (0~1) |
| `is_high_spender` | InGamePurchases > median | 고과금 유저 여부 |
| `is_hardcore` | PlayTimeHours > Q75 | 하드코어 유저 여부 |
| `play_time_bin` | 구간화 (low/mid/high/very_high) | 플레이타임 구간 |
| `age_group` | 구간화 (teen/young_adult/adult/senior) | 연령대 |
| `session_frequency_bin` | 구간화 (rare/occasional/regular/daily) | 접속 빈도 구간 |

### 3. Model Module (`src/models/`)

5개 모델 학습, 평가, 실험 관리, 세그먼트를 담당합니다.

```
src/models/
├── trainer.py    # 모델 정의 + Optuna 튜닝
├── evaluator.py  # 메트릭 계산 + 시각화 (ROC, CM, Feature Importance)
├── registry.py   # MLflow 실험 기록
└── segmenter.py  # K-Means 유저 세그먼트
```

**모델 파이프라인:**
```
                    ┌─ LogisticRegression (baseline)
                    │
                    ├─ XGBoost ──────────┐
X_train, y_train ──▶├─ LightGBM ─────────┼──▶ VotingEnsemble
                    │                    │
                    │                    └──▶ StackingEnsemble
                    │                         (meta: LR)
                    └─────────────────────────────────────────▶ 비교 → Best 저장
```

**Optuna 튜닝 (선택적):**
- XGBoost: max_depth, learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda
- LightGBM: num_leaves, learning_rate, n_estimators, min_child_samples, subsample, colsample_bytree

### 4. API Module (`src/api/`)

FastAPI 기반 REST API 서버입니다.

```
src/api/
├── main.py           # 앱 생성, lifespan (모델 로드), 라우터 등록
├── schemas.py        # Pydantic 모델 (요청/응답 스키마)
├── dependencies.py   # ModelService (싱글톤), API Key 인증
├── middleware.py      # RequestLoggingMiddleware (처리 시간 기록)
└── routes/
    ├── predict.py    # POST /predict/single, POST /predict/batch
    ├── segment.py    # POST /segment/classify
    ├── model_info.py # GET /model/info, GET /model/features/importance
    └── health.py     # GET /health
```

**요청 처리 흐름:**
```
Client Request
    │
    ▼
RequestLoggingMiddleware (처리 시간 측정)
    │
    ▼
API Key 검증 (X-API-Key 헤더)
    │
    ▼
Pydantic 스키마 검증 (PlayerFeatures)
    │
    ▼
ModelService.predict()
    ├── 피처 엔지니어링 (동적 생성)
    ├── 모델 추론 (predict_proba)
    ├── 리스크 레벨 분류 (low/medium/high/critical)
    ├── Top-5 리스크 요인 추출
    └── 리스크별 추천 액션 매핑
    │
    ▼
PredictionResponse (JSON)
```

**리스크 레벨 분류:**
| 확률 범위 | 레벨 | 설명 |
|-----------|------|------|
| 0.0 - 0.3 | low | 안전 |
| 0.3 - 0.5 | medium | 주의 |
| 0.5 - 0.7 | high | 위험 |
| 0.7 - 1.0 | critical | 긴급 |

### 5. Dashboard Module (`src/dashboard/`)

Streamlit + Plotly 기반 인터랙티브 대시보드입니다.

**5개 페이지 구성:**

| 페이지 | 내용 |
|--------|------|
| 개요 | KPI 카드 (유저 수, 이탈률, 평균 플레이타임), 장르/난이도별 이탈률 |
| 모델 성능 | ROC 커브, 혼동 행렬, 피처 중요도 차트 |
| 유저 세그먼트 | K-Means 클러스터 분포, 세그먼트별 이탈률, 리텐션 전략 카드 |
| 모니터링 | KS test 드리프트 결과, PSI 점수, 예측 분포 비교 |
| 이탈 예측 | 유저 정보 입력 폼 → 실시간 이탈 확률 + 리스크 요인 |

### 6. Monitoring Module (`src/monitoring/`)

데이터/모델 드리프트를 탐지합니다.

```
src/monitoring/
└── drift.py
    ├── detect_data_drift()       # 피처별 KS test (p < 0.05 → drift)
    ├── detect_prediction_drift() # 예측 분포 KS test
    └── calculate_psi()           # Population Stability Index
```

**드리프트 판정 기준:**
| 지표 | 기준 | 판정 |
|------|------|------|
| KS test p-value | < 0.05 | 해당 피처 드리프트 발생 |
| PSI | < 0.1 | 안정 |
| PSI | 0.1 - 0.25 | 주의 |
| PSI | > 0.25 | 심각한 드리프트 |

---

## Infrastructure

### Docker Compose 구성

```
┌─────────────────────────────────────────────────────────┐
│  docker-compose.yml                                      │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ PostgreSQL │  │ MLflow UI  │  │ API (FastAPI)      │ │
│  │ :5432      │  │ :5000      │  │ :8000              │ │
│  │            │  │            │  │                    │ │
│  │ - players  │  │ - sqlite   │  │ depends: postgres  │ │
│  │ - predict  │  │   backend  │  │ healthcheck: /heal │ │
│  │ - registry │  │            │  │                    │ │
│  └────────────┘  └────────────┘  └─────────┬──────────┘ │
│                                             │            │
│                                  ┌──────────▼─────────┐  │
│                                  │ Dashboard          │  │
│                                  │ (Streamlit) :8501  │  │
│                                  │                    │  │
│                                  │ depends: api       │  │
│                                  └────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### CI/CD Pipeline

```
Push / PR to main
    │
    ▼
GitHub Actions
    ├── Setup Python 3.12
    ├── Cache pip dependencies
    ├── Install dependencies
    ├── Ruff lint check
    └── pytest (45 tests, auto-skip for missing data)
```

### DVC Pipeline

```
prepare ──────────▶ train ──────────▶ evaluate
   │                  │                  │
   │ deps:            │ deps:            │ deps:
   │ - loader.py      │ - train.py       │ - evaluate.py
   │ - engineer.py    │ - trainer.py     │ - best_model.joblib
   │ - raw CSV        │ - features CSV   │
   │                  │                  │ plots:
   │ outs:            │ outs:            │ - shap_summary.png
   │ - features.csv   │ - model.joblib   │ - shap_bar.png
   │                  │ plots:           │ - shap_waterfall.png
   │                  │ - roc.png        │
   │                  │ - cm.png         │
   │                  │ - importance.png │
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
│ play_time    │     │ model_version    │    │ f1_score         │
│ sessions/wk  │     │ created_at       │    │ is_active        │
│ avg_session  │     └──────────────────┘    │ created_at       │
│ player_level │                              └──────────────────┘
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
| SQLite for MLflow | 로컬 개발 환경에 적합, 프로덕션에서는 PostgreSQL로 교체 가능 |
| API Key 인증 | MVP 수준의 간단한 인증, 프로덕션에서는 OAuth2/JWT로 확장 가능 |
| pre-commit + GitHub Actions | 로컬 린트(pre-commit) + CI 검증(Actions) 이중 품질 게이트 |
