# GameAI Analytics

게임 유저 행동 분석 & 이탈 예측 시스템

## Overview

게임 유저의 행동 데이터를 분석하여 이탈(churn)을 예측하고, 실시간 API 및 대시보드를 통해 인사이트를 제공하는 end-to-end ML 파이프라인 프로젝트입니다.

### Key Features

- **EDA & 통계 분석**: t-test, chi-square, A/B testing (Z-test, Bootstrap, Cohen's h)
- **ML 파이프라인**: Logistic Regression, XGBoost, LightGBM, Ensemble (Voting/Stacking)
- **하이퍼파라미터 튜닝**: Optuna 기반 자동 튜닝
- **모델 해석**: SHAP (Summary, Bar, Waterfall)
- **API 서빙**: FastAPI + API Key 인증
- **대시보드**: Streamlit + Plotly 인터랙티브 시각화
- **MLOps**: MLflow 실험 추적, 드리프트 탐지 (KS test, PSI)
- **CI/CD**: GitHub Actions, Docker, docker-compose

## Project Structure

```
GameAI_Analytics/
├── data/
│   ├── raw/              # 원본 데이터 (Kaggle)
│   ├── processed/        # 전처리된 데이터
│   └── synthetic/        # 합성 데이터
├── models/               # 학습된 모델, 차트
├── notebooks/
│   ├── 01_eda_gaming_behavior.ipynb
│   └── 02_eda_cookie_cats.ipynb
├── scripts/
│   ├── train.py          # 모델 학습
│   ├── evaluate.py       # SHAP 분석
│   └── generate_synthetic.py
├── src/
│   ├── api/              # FastAPI 서버
│   ├── dashboard/        # Streamlit 대시보드
│   ├── data/             # 데이터 로딩/전처리
│   ├── features/         # 피처 엔지니어링
│   ├── models/           # 모델 학습/평가/레지스트리
│   └── monitoring/       # 드리프트 탐지
├── tests/                # pytest 테스트 (38개)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Quick Start

### 1. Setup

```bash
# 의존성 설치
pip install ".[dev]"

# 데이터 준비 (data/raw/ 에 CSV 파일 배치)
```

### 2. 모델 학습

```bash
# 기본 학습 (5개 모델)
python scripts/train.py

# Optuna 튜닝 포함
python scripts/train.py --tune --tune-trials 50

# SHAP 분석
python scripts/evaluate.py
```

### 3. API 서버

```bash
# 개발 서버
python -m src.api.main

# 예측 요청
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
```

## Model Performance

| Model | AUC-ROC | F1-Score |
|-------|---------|----------|
| LogisticRegression | 0.9233 | - |
| **XGBoost** | **0.9409** | - |
| LightGBM | 0.9392 | - |
| VotingEnsemble | 0.9383 | - |
| StackingEnsemble | 0.9402 | - |

Best model: **XGBoost** (AUC-ROC 0.9409)

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, SHAP, Optuna
- **API**: FastAPI, Pydantic, uvicorn
- **Dashboard**: Streamlit, Plotly
- **MLOps**: MLflow
- **Infra**: Docker, GitHub Actions
- **Testing**: pytest (38 tests), Ruff linter

## Datasets

1. **Online Gaming Behavior Dataset** (Kaggle) - 40,034 rows
   - 이탈 정의: EngagementLevel = "Low"
2. **Cookie Cats A/B Test** (Kaggle) - 90,189 rows
   - 리텐션 분석, A/B 테스트 통계 검정
3. **Synthetic Data** - 10,000 rows
   - 시계열 유저 행동 시뮬레이션

## Tests

```bash
# 전체 테스트
pytest tests/ -v

# 린트
ruff check src/ tests/
```
