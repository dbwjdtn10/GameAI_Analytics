from pathlib import Path

# 프로젝트 루트
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# 데이터셋 경로
GAMING_BEHAVIOR_PATH = RAW_DIR / "gaming_behavior" / "online_gaming_behavior_dataset.csv"
COOKIE_CATS_PATH = RAW_DIR / "mobile_ab" / "cookie_cats.csv"

# MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "gameai-churn-prediction"

# 모델
MODEL_DIR = ROOT_DIR / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15

# 타겟 정의
CHURN_LABEL = "is_churned"  # EngagementLevel=Low → 1, else → 0

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
