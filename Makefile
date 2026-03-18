.PHONY: help install train train-tune evaluate segment api dashboard test lint coverage benchmark load-test up down logs clean onnx

help: ## 도움말
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── 설치 ───────────────────────────────────────
install: ## 의존성 설치
	pip install ".[dev]"

# ─── ML 파이프라인 ──────────────────────────────
train: ## 모델 학습 (5개 모델 비교)
	python scripts/train.py

train-tune: ## 모델 학습 + Optuna 튜닝 (50 trials)
	python scripts/train.py --tune --tune-trials 50

evaluate: ## SHAP 모델 해석
	python scripts/evaluate.py

segment: ## K-Means 유저 세그먼트
	python scripts/segment.py

onnx: ## ONNX 모델 변환 + 벤치마크
	python src/models/onnx_converter.py

# ─── 서비스 ─────────────────────────────────────
api: ## FastAPI 서버 실행 (localhost:8000)
	python -m src.api.main

dashboard: ## Streamlit 대시보드 실행 (localhost:8501)
	streamlit run src/dashboard/app.py

# ─── Docker ─────────────────────────────────────
up: ## Docker Compose 기동 (7 서비스)
	docker-compose up --build -d

down: ## Docker Compose 종료
	docker-compose down

logs: ## Docker 로그 확인
	docker-compose logs -f

# ─── 테스트 & 품질 ──────────────────────────────
test: ## 전체 테스트 실행 (63개)
	pytest tests/ -v --tb=short

lint: ## Ruff 린트 검사
	ruff check src/ tests/

lint-fix: ## Ruff 린트 자동 수정
	ruff check src/ tests/ --fix

coverage: ## 테스트 커버리지 리포트
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=xml

# ─── 성능 테스트 ────────────────────────────────
benchmark: ## API 응답 시간 벤치마크 (p50/p95/p99)
	python scripts/benchmark.py --url http://localhost:8000 --n-requests 200

load-test: ## Locust 부하 테스트 (웹 UI)
	locust -f locustfile.py --host http://localhost:8000

load-test-headless: ## Locust 부하 테스트 (CLI, 100유저 30초)
	locust -f locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10 --run-time 30s --headless

# ─── DVC ────────────────────────────────────────
dvc-run: ## DVC 파이프라인 실행 (validate → prepare → train → evaluate)
	dvc repro

# ─── 유틸 ───────────────────────────────────────
clean: ## 캐시 및 임시 파일 정리
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml
