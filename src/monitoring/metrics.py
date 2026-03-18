"""Prometheus 메트릭 수집 모듈."""

from prometheus_client import Counter, Gauge, Histogram

# 예측 요청 메트릭
PREDICTION_REQUEST_COUNT = Counter(
    "gameai_prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint", "risk_level"],
)

PREDICTION_LATENCY = Histogram(
    "gameai_prediction_latency_seconds",
    "Prediction inference latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

PREDICTION_ERROR_COUNT = Counter(
    "gameai_prediction_errors_total",
    "Total number of prediction errors",
    ["endpoint", "error_type"],
)

# 모델 메트릭
MODEL_LOADED = Gauge(
    "gameai_model_loaded",
    "Whether the ML model is loaded (1=loaded, 0=not loaded)",
)

MODEL_INFERENCE_COUNT = Counter(
    "gameai_model_inference_total",
    "Total number of model inferences",
    ["model_type"],
)

# 캐시 메트릭
CACHE_HIT_COUNT = Counter(
    "gameai_cache_hits_total",
    "Total number of cache hits",
)

CACHE_MISS_COUNT = Counter(
    "gameai_cache_misses_total",
    "Total number of cache misses",
)

# 비즈니스 메트릭
HIGH_RISK_USERS_DETECTED = Counter(
    "gameai_high_risk_users_total",
    "Total number of high/critical risk users detected",
    ["risk_level"],
)

BATCH_SIZE = Histogram(
    "gameai_batch_prediction_size",
    "Size of batch prediction requests",
    buckets=(1, 5, 10, 50, 100, 500, 1000),
)

# 드리프트 메트릭
DRIFT_SCORE = Gauge(
    "gameai_drift_ratio",
    "Current data drift ratio (0-1)",
)
