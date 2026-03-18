"""Locust 부하 테스트.

사용법:
    locust -f locustfile.py --host http://localhost:8000
    # 웹 UI: http://localhost:8089

시나리오:
    - 단일 예측 (70%)
    - 배치 예측 (15%)
    - 세그먼트 분류 (10%)
    - 모델 정보 조회 (5%)
"""

import random

from locust import HttpUser, between, task

API_KEY = "dev-key-gameai-2024"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

GENRES = ["RPG", "FPS", "MOBA", "Sports", "Strategy", "Simulation", "Action", "Adventure"]
LOCATIONS = ["USA", "Europe", "Asia", "South America", "Africa", "Australia", "Other"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]


def random_player():
    """랜덤 플레이어 데이터 생성."""
    return {
        "Age": random.randint(10, 65),
        "Gender": random.choice(["Male", "Female"]),
        "Location": random.choice(LOCATIONS),
        "GameGenre": random.choice(GENRES),
        "GameDifficulty": random.choice(DIFFICULTIES),
        "PlayTimeHours": round(random.uniform(1, 2000), 1),
        "SessionsPerWeek": random.randint(1, 30),
        "AvgSessionDurationMinutes": round(random.uniform(10, 300), 1),
        "PlayerLevel": random.randint(1, 150),
        "AchievementsUnlocked": random.randint(0, 200),
        "InGamePurchases": random.randint(0, 100),
    }


class GameAIUser(HttpUser):
    """일반 API 사용자 시뮬레이션."""

    wait_time = between(0.5, 2)

    @task(70)
    def predict_single(self):
        """단일 유저 이탈 예측."""
        self.client.post(
            "/api/v1/predict/single",
            json=random_player(),
            headers=HEADERS,
        )

    @task(15)
    def predict_batch(self):
        """배치 예측 (10~50명)."""
        batch_size = random.randint(10, 50)
        self.client.post(
            "/api/v1/predict/batch",
            json={"players": [random_player() for _ in range(batch_size)]},
            headers=HEADERS,
        )

    @task(10)
    def classify_segment(self):
        """세그먼트 분류."""
        self.client.post(
            "/api/v1/segment/classify",
            json=random_player(),
            headers=HEADERS,
        )

    @task(5)
    def model_info(self):
        """모델 정보 조회."""
        self.client.get(
            "/api/v1/model/info",
            headers=HEADERS,
        )

    def on_start(self):
        """시작 시 헬스 체크."""
        self.client.get("/health")


class HeavyUser(HttpUser):
    """대량 배치 요청 사용자 시뮬레이션."""

    wait_time = between(2, 5)
    weight = 1  # 전체 사용자 중 비율

    @task
    def predict_large_batch(self):
        """대규모 배치 예측 (100~500명)."""
        batch_size = random.randint(100, 500)
        self.client.post(
            "/api/v1/predict/batch",
            json={"players": [random_player() for _ in range(batch_size)]},
            headers=HEADERS,
        )
