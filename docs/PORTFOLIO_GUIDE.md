# GameAI Analytics — 포트폴리오 가이드

> 면접/포트폴리오 제출 시 참고할 개인 문서

---

## 프로젝트 한줄 요약

**게임 유저 이탈 예측 E2E ML 시스템** — 데이터 검증부터 모델 학습, API 서빙, 모니터링, 비즈니스 임팩트 분석까지 프로덕션 수준의 전체 파이프라인을 구현한 프로젝트

---

## 면접 대비 — 질문별 답변 포인트

### "이 프로젝트를 간단히 설명해주세요"

> 게임 유저 40,034명의 행동 데이터를 분석하여 이탈을 예측하는 ML 시스템입니다.
> XGBoost 모델이 AUC-ROC 0.94를 달성했고, FastAPI로 실시간 서빙하며,
> Prometheus/Grafana로 모니터링하고, Airflow로 자동 재학습하는 구조입니다.
> 비즈니스 관점에서 이탈 예측 모델의 ROI도 정량화했습니다.

### "왜 XGBoost를 선택했나요?"

> 5개 모델(LR, XGBoost, LightGBM, Voting, Stacking)을 비교한 결과
> XGBoost가 AUC-ROC 0.9409로 최고 성능이었습니다.
> 앙상블(Stacking 0.9402)과 큰 차이가 없어서
> 단일 모델의 해석성과 서빙 편의성을 고려해 XGBoost를 선택했습니다.

### "Training-Serving Skew는 어떻게 방지하나요?"

> Feature Store를 직접 구현했습니다.
> 학습 시 범주형 인코더 매핑(Gender→{Female:0, Male:1})을 JSON으로 저장하고,
> 서빙 시 동일한 매핑을 로드하여 변환합니다.
> 입력 데이터가 학습 분포에서 벗어나는지도 IQR 기반으로 검증합니다.

### "모니터링은 어떻게 하나요?"

> 3단계 모니터링입니다:
> 1. **실시간**: Prometheus 10종 커스텀 메트릭 (예측 수, 레이턴시, 캐시 히트율, 에러율)
> 2. **시각화**: Grafana 10패널 대시보드 (자동 프로비저닝)
> 3. **알림**: 6개 alerting rule (에러율>5%, p95>1초, 모델다운, 드리프트>50%)
>
> 데이터 드리프트는 KS test + PSI로 탐지하고,
> Airflow DAG이 매주 자동으로 체크 → 조건부 재학습합니다.

### "비즈니스 가치를 어떻게 측정했나요?"

> 노트북 06과 대시보드 비즈니스 임팩트 페이지에서:
> - 유저당 LTV($150) × 예측 정확도 × 리텐션 성공률(25%)로 보존 매출 계산
> - 임계값(threshold) 0.1~0.9 범위에서 비용-편익 최적화
> - 최적 임계값에서의 연간 ROI 추정
> - 세그먼트별(하드코어/캐주얼/위험/신규) 차별화된 ROI 분석

### "API 성능은 어떻게 보장하나요?"

> 4가지 방법:
> 1. **Redis 캐싱** — 동일 입력 중복 추론 방지 (TTL 5분, graceful fallback)
> 2. **ONNX Runtime** — XGBoost→ONNX 변환으로 추론 속도 최적화
> 3. **Rate Limiting** — slowapi로 DDoS 방지 (60req/min)
> 4. **벤치마크** — p50/p95/p99 레이턴시 측정 스크립트 + Locust 부하 테스트

### "테스트 전략은?"

> 63개 테스트, 7개 카테고리:
> - **API 테스트**: 엔드포인트, 인증, 에러 처리
> - **데이터 품질**: Pandera 스키마 검증 (유효/무효 데이터)
> - **모델 회귀**: AUC≥0.90, Accuracy≥0.85 성능 게이트
> - **드리프트**: KS test, PSI 로직 검증
> - **보안**: JWT 토큰 발급/검증
> - **Feature Store**: 인코더 일관성
> - **메트릭**: Prometheus 엔드포인트, 헬스체크
>
> CI에서 데이터/모델 없으면 자동 스킵. PR에서 모델 성능 게이트 적용.

---

## 기술 스택 정리 (면접용)

| 기술 | 왜 이 기술을 선택했는지 |
|------|------------------------|
| XGBoost | 정형 데이터 최고 성능, SHAP 호환, ONNX 변환 가능 |
| FastAPI | 비동기 지원, 자동 Swagger, Pydantic 검증 |
| Streamlit | 빠른 프로토타이핑, 인터랙티브 위젯 |
| Pandera | 코드 기반 스키마 검증, CI 통합 용이 |
| structlog | JSON 구조화 로깅, request_id 추적 |
| Prometheus + Grafana | 업계 표준 메트릭 수집 + 시각화 |
| Redis | 고성능 비동기 캐싱, graceful fallback |
| JWT | 프로덕션 수준 토큰 인증 (API Key와 병행) |
| ONNX Runtime | 추론 최적화, 프레임워크 독립적 서빙 |
| Locust | Python 기반 부하 테스트, 웹 UI |
| DVC | 데이터/모델 버전 관리, 재현 가능한 파이프라인 |
| Airflow | 스케줄링된 재학습, 조건부 실행 |
| Docker Compose | 7 서비스 오케스트레이션, 1-command 기동 |

---

## 프로젝트 수치 요약

| 항목 | 수치 |
|------|------|
| 학습 데이터 | 40,034 rows × 11 features + 12 derived |
| 모델 성능 | AUC-ROC 0.9409 / Accuracy 95.35% / F1 0.9083 |
| 비교 모델 | 5개 (LR, XGBoost, LightGBM, Voting, Stacking) |
| API 엔드포인트 | 8개 (predict, batch, segment, model info, health, metrics, auth) |
| 대시보드 페이지 | 7개 (개요, 모델, 세그먼트, 모니터링, 예측, What-If, ROI) |
| 테스트 | 63개 (11 모듈) |
| Docker 서비스 | 7개 (API, Dashboard, DB, Redis, MLflow, Prometheus, Grafana) |
| Prometheus 메트릭 | 10종 커스텀 + FastAPI 자동 수집 |
| 알림 규칙 | 6개 (에러율, 레이턴시, 모델, 드리프트, 캐시, 위험유저) |
| 노트북 | 6개 (EDA, Feature Eng, Model Comparison, A/B Test, Business Impact) |
| CI/CD Jobs | 3개 (lint+test, model-validation, docker-build) |

---

## 데모 시나리오 (라이브 시연 시)

### 1. 전체 서비스 기동 (2분)
```bash
make up
# 7 서비스 기동 확인
```

### 2. API 예측 데모 (1분)
```bash
# 단일 예측
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "X-API-Key: dev-key-gameai-2024" \
  -H "Content-Type: application/json" \
  -d '{"Age":25,"Gender":"Male","Location":"USA","GameGenre":"RPG","GameDifficulty":"Medium","PlayTimeHours":120.5,"SessionsPerWeek":5,"AvgSessionDurationMinutes":45.0,"PlayerLevel":32,"AchievementsUnlocked":15,"InGamePurchases":8}'

# JWT 토큰 발급
curl -X POST http://localhost:8000/auth/token -d "username=admin&password=gameai2024"
```

### 3. 대시보드 시연 (3분)
- http://localhost:8501 → 7페이지 순서대로
- **What-If 분석**: 세션 수를 3→10으로 올리면 이탈 확률이 어떻게 변하는지 시연
- **비즈니스 임팩트**: LTV/리텐션비용 조절 → 연간 ROI 변화

### 4. Grafana 모니터링 (1분)
- http://localhost:3000 (admin/gameai2024)
- 예측 요청률, 레이턴시 차트 실시간 확인

### 5. 부하 테스트 (1분)
```bash
make load-test
# http://localhost:8089 에서 100 유저 설정 → 결과 차트
```

---

## 디렉토리별 핵심 파일 (코드 리뷰 대비)

면접관이 코드를 볼 때 가장 인상적인 파일들:

| 우선순위 | 파일 | 포인트 |
|----------|------|--------|
| 1 | `src/api/routes/predict.py` | 캐싱 + 메트릭 + 비즈니스 로직 통합 |
| 2 | `src/features/store.py` | Feature Store 직접 구현, Training-Serving Skew 방지 |
| 3 | `src/monitoring/metrics.py` | 10종 Prometheus 커스텀 메트릭 설계 |
| 4 | `src/data/validation.py` | Pandera 3단계 데이터 검증 |
| 5 | `src/dashboard/app.py` | What-If 분석 + 비즈니스 임팩트 페이지 |
| 6 | `src/models/onnx_converter.py` | ONNX 변환 + 벤치마크 |
| 7 | `notebooks/06_business_impact.ipynb` | LTV, ROI, 임계값 최적화 |
| 8 | `docker-compose.yml` | 7 서비스 오케스트레이션 |
| 9 | `.github/workflows/ci.yml` | 3-job CI (테스트+모델게이트+Docker) |
| 10 | `alerts.yml` | Prometheus 알림 규칙 6종 |

---

## 알려진 한계점 (면접에서 솔직하게 말할 것)

| 한계 | 이유 | 개선 방향 |
|------|------|-----------|
| LabelEncoder가 요청마다 재생성됨 | 기존 구조 유지 (Feature Store로 대체 가능하지만 기존 API 호환성) | Feature Store의 `transform_for_serving()` 연동 |
| Redis 싱글톤 race condition | async 환경에서 첫 연결 시 이론적 중복 가능 | asyncio.Lock 또는 startup event에서 초기화 |
| JWT 시크릿 하드코딩 기본값 | 개발 편의 (env var 미설정 시 기본값 사용) | 프로덕션에서는 반드시 env var 설정 필수 |
| ONNX 변환은 수동 | 자동화 미적용 | DVC 파이프라인에 ONNX 변환 스테이지 추가 |
| 데이터가 정적 CSV | 실시간 스트리밍 없음 | Kafka/Kinesis 연동 가능 |

---

## 이력서/포트폴리오 기재 예시

### 한국어
**GameAI Analytics — 게임 유저 이탈 예측 시스템**
- XGBoost 기반 이탈 예측 모델 (AUC-ROC 0.94, 5모델 비교 실험)
- FastAPI REST API (JWT 인증, Redis 캐싱, Rate Limiting)
- Prometheus + Grafana 실시간 모니터링 (10종 커스텀 메트릭, 6개 알림 규칙)
- Pandera 데이터 검증 + Feature Store로 Training-Serving Skew 방지
- Streamlit 대시보드 7페이지 (What-If 분석, 비즈니스 임팩트 ROI)
- Docker Compose 7서비스 + GitHub Actions CI/CD (63개 테스트)

### English
**GameAI Analytics — Game Player Churn Prediction System**
- Built XGBoost churn prediction model (AUC-ROC 0.94, 5-model comparison)
- Designed FastAPI REST API with JWT auth, Redis caching, rate limiting
- Implemented Prometheus + Grafana monitoring (10 custom metrics, 6 alert rules)
- Created Pandera validation pipeline + lightweight Feature Store
- Built 7-page Streamlit dashboard with What-If analysis and ROI calculator
- Containerized with Docker Compose (7 services) + CI/CD (63 tests)
