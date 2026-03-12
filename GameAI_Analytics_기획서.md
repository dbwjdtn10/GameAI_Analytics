# GameAI Analytics - 게임 유저 행동 분석 & 이탈 예측 시스템

## 프로젝트 개요

게임 유저 로그 데이터를 분석하여 이탈을 예측하고, 유저 세그먼트별 맞춤 리텐션 전략을 제안하는 엔드투엔드 ML 파이프라인 프로젝트.

**타겟 공고:** 네오플 전략분석실 AI 엔지니어 / 넥슨 인텔리전스랩스 AI 엔지니어
**핵심 어필:** Kaggle 데이터셋 + 합성 데이터 활용, Scikit-Learn/XGBoost 모델링, FastAPI 서빙, 게임 도메인 이해

---

## 타겟 공고 JD 분석 및 정렬

### 넥슨/네오플 AI 관련 직무에서 공통 요구사항 (조사 기반)

| JD 요구사항 | 본 프로젝트에서의 대응 |
|------------|----------------------|
| Python, SQL 필수 | Python 전체 파이프라인 + PostgreSQL 쿼리 |
| Pandas, Scikit-learn 활용 | 전처리/피처 엔지니어링/모델링 전반 |
| 통계 기반 분석 (요인분석, 인과관계, 시계열) | EDA 통계 분석, 시계열 플레이 패턴, A/B 테스트(Cookie Cats) |
| 시각화 (Tableau 등) | Streamlit + Plotly 대시보드 (Tableau 대체) |
| 대용량 데이터 처리 (Spark, Snowflake) | 합성 데이터 대량 생성으로 스케일 시연 (Spark는 선택적 확장) |
| 게임에 대한 관심/이해 | 게임 도메인 특화 피처 설계, 유저 세그먼트 전략 |
| MLOps (컨테이너, 파이프라인) | Docker, MLflow, GitHub Actions CI/CD |
| 데이터를 통계적으로 이해하고 논리적으로 설명하는 역량 | SHAP 기반 모델 해석, EDA 리포트 |

### JD 대비 보완 포인트

- **Airflow:** 스케줄링 기반 재학습 파이프라인에 간단히라도 적용 → JD 정렬도 높임
- **SQL 역량:** PostgreSQL에 유저 데이터 적재 후 SQL 기반 분석 쿼리도 노트북에 포함
- **통계 분석:** EDA에 가설 검정(t-test, chi-square), A/B 테스트 통계적 유의성 검증 포함

> **참고 소스:**
> - [넥슨 인텔리전스랩스 데이터 분석가 JD (원티드)](https://www.wanted.co.kr/wd/106989)
> - [넥슨 인텔리전스랩스 테크블로그](https://www.intelligencelabs.tech)
> - [넥슨 AI 활용 사례](https://www.intelligencelabs.tech/6adc3e4b-646a-4b6b-81b6-63b60b71330e)

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 언어 | Python 3.11+ |
| 데이터 처리 | Pandas, NumPy, SQL |
| ML/모델링 | Scikit-Learn, XGBoost, LightGBM |
| 모델 해석 | SHAP |
| 실험 관리 | MLflow |
| 하이퍼파라미터 튜닝 | Optuna |
| 워크플로우 | Airflow (재학습 스케줄링) |
| API 서빙 | FastAPI, Pydantic |
| 시각화/대시보드 | Streamlit, Plotly |
| DB | PostgreSQL (유저 데이터), Redis (캐싱) |
| 컨테이너 | Docker, docker-compose |
| CI/CD | GitHub Actions (lint, test, 모델 성능 체크) |
| 데이터 버전 관리 | DVC (Data Version Control) |
| 테스트 | pytest |

---

## 데이터셋

### 트랙 1: Kaggle 실제 데이터 (모델 성능 검증용)

- **[Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)** (메인)
  - 40,034행 x 13컬럼
  - 피처: PlayTimeHours, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked, InGamePurchases 등
  - 타겟: `EngagementLevel` (High/Medium/Low) → **Low를 이탈 proxy로 사용**
  - 용도: 실제 데이터 기반 모델 학습 및 성능 보고

- **[Mobile Games AB Testing (Cookie Cats)](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing)** (보조)
  - 90,189행 x 5컬럼
  - 타겟: `retention_1`, `retention_7` (실제 리텐션 라벨)
  - 용도: 리텐션 예측 모델 교차 검증, A/B 테스트 통계 분석 사례

### 트랙 2: 합성 데이터 (시스템 시연 + 데이터 엔지니어링 역량)

Kaggle 데이터에 없는 피처(login_streak, guild_activity, purchase_history 등)를 포함한 **실제 게임 로그 형태**의 합성 데이터를 직접 생성.

- 유저 타입별 행동 프로파일 (하드코어/캐주얼/이탈예정/복귀유저)
- 파라미터로 데이터 볼륨, 이탈률, 시즌 이벤트 효과 등 조절 가능
- **합성 데이터 vs 실제 데이터 분포 비교 분석** 포함 → 합성 데이터 신뢰성 검증

```bash
# 사용 예시
python scripts/generate_synthetic.py \
  --num_users 50000 \
  --days 90 \
  --churn_rate 0.25 \
  --event_effect 0.15
```

> **포지셔닝:** Kaggle 데이터로 "모델이 실제로 동작한다"를 증명하고, 합성 데이터로 "실제 게임 환경에서의 전체 파이프라인"을 시연한다.

---

## 프로젝트 구조

```
gameai-analytics/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions (lint + test + model perf check)
├── dvc.yaml                      # DVC 파이프라인 정의
├── .dvc/                         # DVC 설정
│
├── data/
│   ├── raw/                      # 원본 데이터
│   │   ├── gaming_behavior/      # Kaggle - Online Gaming Behavior
│   │   └── mobile_ab/            # Kaggle - Cookie Cats
│   ├── processed/                # 전처리된 데이터
│   └── synthetic/                # 합성 데이터
│
├── notebooks/                    # EDA 노트북
│   ├── 01_eda_gaming_behavior.ipynb
│   ├── 02_eda_cookie_cats.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_ab_test_analysis.ipynb # A/B 테스트 통계 분석
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # 설정 관리
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # 데이터 로드
│   │   ├── preprocessor.py       # 전처리 파이프라인
│   │   └── synthetic.py          # 합성 데이터 생성기
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer.py           # 피처 엔지니어링
│   │   └── selector.py           # 피처 선택
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py            # 모델 학습
│   │   ├── evaluator.py          # 평가 메트릭
│   │   └── registry.py           # 모델 레지스트리 (MLflow)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI 앱
│   │   ├── schemas.py            # Pydantic 스키마
│   │   ├── routes/
│   │   │   ├── predict.py        # 이탈 예측 엔드포인트
│   │   │   ├── segment.py        # 유저 세그먼트 분석
│   │   │   └── health.py         # 헬스체크
│   │   └── middleware.py         # 로깅, 에러 핸들링
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift_detector.py     # 모델/데이터 드리프트 감지
│   │
│   └── dashboard/
│       └── app.py                # Streamlit 대시보드
│
├── tests/
│   ├── test_preprocessor.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_api.py
│   ├── test_data_quality.py      # 데이터 스키마/품질 검증
│   └── test_model_regression.py  # 모델 성능 regression 테스트
│
├── dags/
│   └── retrain_pipeline.py       # Airflow DAG: 재학습 파이프라인
│
├── scripts/
│   ├── train.py                  # 학습 실행 스크립트
│   ├── evaluate.py               # 평가 스크립트
│   └── generate_synthetic.py     # 합성 데이터 생성
│
└── mlflow/                       # MLflow 실험 트래킹 저장소
```

---

## 핵심 기능 명세

### 1. 데이터 파이프라인 (`src/data/`)

**전처리 파이프라인:**
- 결측치 처리 (KNN Imputer / 도메인 기반 룰)
- 이상치 탐지 (IQR, Z-score)
- 범주형 변수 인코딩 (Label / Target Encoding)
- 타겟 변수 가공: EngagementLevel → 이진 분류 (Low=이탈, Medium+High=유지)

**합성 데이터 생성기 (차별화 포인트):**
- 실제 게임 로그 패턴을 모방하는 시뮬레이터
- 유저 타입별 행동 프로파일 (하드코어/캐주얼/이탈예정/복귀유저)
- 시계열 로그 생성 (일별 로그인, 세션, 구매, 소셜 활동)
- 파라미터로 데이터 볼륨, 이탈률, 시즌 이벤트 효과 등 조절 가능

**데이터 버전 관리 (DVC):**
- raw/processed/synthetic 데이터를 DVC로 버전 관리
- `dvc.yaml`로 데이터 → 전처리 → 피처 → 학습 파이프라인 정의
- Git에는 메타데이터만, 실제 데이터는 remote storage (로컬 or S3)

### 2. 피처 엔지니어링 (`src/features/`)

**Kaggle 데이터에서 직접 사용 가능한 피처:**

| 카테고리 | 원본 컬럼 | 파생 피처 | 설명 |
|----------|-----------|-----------|------|
| 활동 패턴 | `PlayTimeHours` | `playtime_per_session` | 세션당 플레이 시간 |
| | `SessionsPerWeek` | `weekly_activity_intensity` | 주간 활동 강도 |
| | `AvgSessionDurationMinutes` | `session_engagement_score` | 세션 몰입도 |
| 진행도 | `PlayerLevel` | `level_efficiency` | 플레이 시간 대비 레벨 효율 |
| | `AchievementsUnlocked` | `achievement_rate` | 레벨 대비 업적 달성률 |
| 소비 | `InGamePurchases` | `purchase_per_hour` | 시간당 구매 빈도 |
| 프로필 | `GameGenre`, `GameDifficulty` | 인코딩 | 장르/난이도 선호 |

**합성 데이터 전용 피처 (확장):**

| 카테고리 | 피처 | 설명 |
|----------|------|------|
| 활동 패턴 | `login_streak` | 연속 로그인 일수 |
| | `days_since_last_login` | 마지막 접속 후 경과일 |
| | `playtime_trend_7d` | 최근 7일 플레이 시간 추세 (기울기) |
| 소비 패턴 | `purchase_frequency` | 구매 빈도 |
| | `avg_purchase_amount` | 평균 구매 금액 |
| | `days_since_last_purchase` | 마지막 구매 후 경과일 |
| 소셜 | `friend_count` | 친구 수 |
| | `guild_activity_rate` | 길드 활동 비율 |
| | `chat_messages_per_session` | 세션당 채팅 메시지 수 |
| 세션 | `session_count_7d` | 최근 7일 세션 수 |
| | `peak_hour_ratio` | 피크 시간대 접속 비율 |

### 3. 유저 세그먼트 분류

**세그먼트 방법론:**
- K-Means 클러스터링 (활동/소비/진행도 기반)
- Elbow Method + Silhouette Score로 최적 K 결정
- 클러스터 결과에 도메인 라벨 부여:

| 세그먼트 | 특성 | 리텐션 전략 |
|----------|------|------------|
| 하드코어 | 높은 플레이타임, 높은 레벨, 활발한 구매 | VIP 보상, 경쟁 콘텐츠 |
| 캐주얼 | 낮은 세션, 간헐적 접속 | 가벼운 일일 미션, 로그인 보상 |
| 이탈 위험 | 감소 추세, 낮은 활동 | 복귀 보상, 맞춤 알림 |
| 신규 유저 | 낮은 레벨, 짧은 플레이 기간 | 튜토리얼 보상, 초보자 가이드 |

### 4. 모델링 (`src/models/`)

**학습 파이프라인:**
- Baseline: Logistic Regression
- Main: XGBoost, LightGBM
- Ensemble: Voting/Stacking
- 하이퍼파라미터 튜닝: Optuna

**평가 메트릭:**
- 이탈 예측: AUC-ROC, Precision-Recall, F1-Score
- 피처 중요도: SHAP values (모델 해석 가능성)
- 비즈니스 메트릭: 예측 기반 예상 리텐션 효과 (시뮬레이션)
- Cold Start 대응: 신규 유저(데이터 부족)는 세그먼트 기반 디폴트 예측값 적용

**MLflow 실험 트래킹:**
```
각 실험마다 자동 기록:
- 하이퍼파라미터
- 메트릭 (train/val/test)
- 피처 중요도 차트
- SHAP 분석 결과
- 모델 아티팩트
```

### 5. 모델 재학습 & 모니터링

**재학습 전략 (Airflow DAG):**
```
스케줄: 주 1회 (또는 새 데이터 유입 시 트리거)
파이프라인:
1. 새 데이터 검증 (스키마, 품질 체크)
2. 전처리 + 피처 엔지니어링
3. 모델 재학습
4. 성능 비교 (신규 모델 vs 현재 배포 모델)
5. 성능이 개선된 경우에만 모델 교체 (자동 승격)
6. MLflow에 실험 기록
```

**모델/데이터 드리프트 모니터링 (`src/monitoring/`):**
- 입력 데이터 분포 변화 감지 (PSI: Population Stability Index)
- 예측 분포 변화 감지 (예측값의 평균/분산 추이)
- 성능 메트릭 추이 대시보드 (Streamlit 내 탭으로 통합)
- 임계치 초과 시 알림 (로그 기반, 추후 Slack 연동 가능)

### 6. 예측 API (`src/api/`)

**엔드포인트:**

```
POST /api/v1/predict/churn
  → 단일 유저 이탈 확률 예측

POST /api/v1/predict/batch
  → 다수 유저 일괄 예측

GET  /api/v1/segment/{user_id}
  → 유저 세그먼트 분류 + 리텐션 전략 제안

GET  /api/v1/features/importance
  → 현재 모델의 피처 중요도

GET  /api/v1/model/info
  → 모델 버전, 성능 메트릭, 학습 일시

GET  /api/v1/health
  → 서비스 상태 확인
```

**인증:** API Key 기반 인증 (헤더: `X-API-Key`)

**응답 예시 (POST /api/v1/predict/churn):**
```json
{
  "user_id": "user_12345",
  "churn_probability": 0.73,
  "risk_level": "HIGH",
  "top_risk_factors": [
    {"feature": "playtime_trend_7d", "impact": -0.32, "description": "최근 7일 플레이 시간 급감"},
    {"feature": "days_since_last_purchase", "impact": 0.28, "description": "마지막 결제 후 45일 경과"},
    {"feature": "friend_count", "impact": -0.15, "description": "활성 친구 수 감소"}
  ],
  "recommended_actions": [
    "복귀 보상 지급 (7일 미접속 시)",
    "길드 활동 알림 발송",
    "맞춤 이벤트 쿠폰 제공"
  ]
}
```

### 7. 대시보드 (`src/dashboard/`)

**Streamlit 페이지 구성:**

- **Overview:** 전체 유저 이탈률 트렌드, KPI 요약
- **Cohort Analysis:** 가입 시기별 리텐션 커브
- **Segment View:** 유저 세그먼트별 분포 + 행동 패턴 비교
- **Individual Lookup:** 특정 유저 검색 → 이탈 확률 + SHAP 워터폴 차트
- **Model Performance:** AUC-ROC 커브, Confusion Matrix, 피처 중요도 차트
- **Model Monitoring:** 데이터/예측 드리프트 추이, 재학습 이력
- **A/B Simulation:** 리텐션 전략별 예상 효과 시뮬레이션

---

## 구현 순서

### Phase 1: 프로젝트 셋업 & 데이터 탐색 (2~3일)
```
1. 프로젝트 초기화 (pyproject.toml, 구조 생성, Git, DVC)
2. 데이터 로더 구현 (Gaming Behavior + Cookie Cats)
3. PostgreSQL에 데이터 적재 + SQL 기반 탐색 쿼리
4. EDA 노트북 작성 (분포, 상관관계, 이탈 패턴, 통계 검정)
5. 합성 데이터 생성기 구현
6. 합성 데이터 vs 실제 데이터 분포 비교 분석
```

### Phase 2: 피처 엔지니어링 (2일)
```
7. 전처리 파이프라인 구현 (Scikit-Learn Pipeline)
8. Kaggle 데이터 기반 피처 생성
9. 합성 데이터 기반 확장 피처 생성
10. 피처 선택 (상관관계, VIF, mutual information)
```

### Phase 3: 모델링 (3~4일)
```
11. Baseline 모델 (Logistic Regression)
12. XGBoost / LightGBM 학습
13. Optuna 하이퍼파라미터 튜닝
14. SHAP 분석 + 모델 해석
15. MLflow 실험 트래킹 연동
16. 앙상블 모델 구현
17. 유저 세그먼트 클러스터링 (K-Means)
18. Cookie Cats A/B 테스트 통계 분석
```

### Phase 4: API 서빙 (2~3일)
```
19. FastAPI 앱 구조 셋업
20. 예측 엔드포인트 구현
21. 세그먼트 분석 엔드포인트
22. API Key 인증 + 에러 핸들링
23. pytest 테스트 작성 (단위 + 통합 + 모델 regression)
```

### Phase 5: MLOps & 모니터링 (2~3일)
```
24. 데이터/모델 드리프트 감지 모듈 구현
25. Airflow DAG 작성 (재학습 파이프라인)
26. DVC 파이프라인 정의 (데이터 → 학습 → 평가)
27. 모델 성능 regression 테스트 (CI에 통합)
```

### Phase 6: 대시보드 & 배포 (2~3일)
```
28. Streamlit 대시보드 구현 (모니터링 탭 포함)
29. Docker + docker-compose 구성
30. GitHub Actions CI 설정 (lint + test + model perf check)
31. README 작성 (스크린샷 포함)
32. GitHub 배포
```

**예상 총 소요: 3~4주**

---

## README에 포함할 내용 (포트폴리오 어필용)

```markdown
## 프로젝트 동기
게임 산업에서 유저 이탈은 매출과 직결되는 핵심 문제입니다.
이 프로젝트는 ML 모델을 통해 이탈 위험 유저를 사전에 식별하고,
데이터 기반 리텐션 전략을 제안하는 시스템을 구축합니다.

## 주요 성과
- 이탈 예측 AUC-ROC: 0.XX (baseline 대비 XX% 향상)
- SHAP 기반 피처 중요도 분석으로 이탈 원인 해석
- K-Means 유저 세그먼트별 맞춤 리텐션 전략 자동 생성
- FastAPI 기반 실시간 예측 API (응답 시간 < 50ms)
- 합성 데이터 생성기로 실제 게임 로그 환경 시뮬레이션
- Airflow 기반 자동 재학습 + 드리프트 모니터링

## 실행 방법
docker-compose up --build
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
# MLflow: http://localhost:5000
```

---

## 차별화 포인트 (면접 대비)

1. **게임 도메인 이해:** 단순 ML이 아니라 게임 유저 행동 패턴을 반영한 피처 설계
2. **엔드투엔드:** 데이터 수집 → 전처리 → 모델링 → 서빙 → 모니터링 → 재학습 전체 사이클
3. **모델 해석:** SHAP으로 "왜 이탈할 것인가"를 설명할 수 있음 → 비즈니스 의사결정 지원
4. **2트랙 데이터 전략:** 실제 데이터(Kaggle)로 성능 검증 + 합성 데이터로 실제 게임 환경 시뮬레이션
5. **유저 세그먼테이션:** K-Means 기반 세그먼트 분류 + 세그먼트별 맞춤 리텐션 전략 자동 제안
6. **통계 분석:** A/B 테스트 유의성 검증, 가설 검정 등 통계 기반 의사결정 역량
7. **MLOps:** MLflow 실험 관리, DVC 데이터 버전 관리, Airflow 재학습, Docker 컨테이너화, CI/CD
8. **모니터링:** 데이터/모델 드리프트 감지 + 성능 추이 대시보드 → 프로덕션 운영 감각

---

## 면접 예상 질문 & 답변 준비

| 예상 질문 | 답변 포인트 |
|-----------|------------|
| "왜 EngagementLevel=Low를 이탈로 정의했나?" | 실제 이탈 라벨이 없는 상황에서 proxy 타겟 설정 → 실무에서도 이탈 정의는 도메인에 따라 다름 (7일 미접속, 30일 미접속 등) |
| "합성 데이터로 학습한 모델을 실무에 쓸 수 있나?" | 합성 데이터는 시스템 검증용, 실제 성능은 Kaggle 데이터로 보고. 합성 데이터의 분포 비교 분석으로 신뢰성도 검증 |
| "모델 성능이 안 나오면 어떻게 하나?" | 피처 엔지니어링 반복, 타겟 정의 변경, 데이터 불균형 처리 (SMOTE, class weight), 모델 앙상블 순서로 접근 |
| "실시간 예측이 필요한 상황에서 지연은?" | Redis 캐싱으로 반복 요청 처리, 모델 경량화 (피처 수 제한), 배치 예측 vs 실시간 예측 분리 |
| "드리프트 감지 후 어떻게 대응하나?" | PSI 기반 데이터 드리프트 감지 → 임계치 초과 시 Airflow DAG 트리거 → 재학습 → 성능 비교 후 자동 승격 |
| "왜 XGBoost/LightGBM인가? 딥러닝은?" | 정형 데이터에서는 부스팅 모델이 효율적, 해석 가능성도 높음. 시계열 패턴이 중요해지면 LSTM 확장 가능 |
