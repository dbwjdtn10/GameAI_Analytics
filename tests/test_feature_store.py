"""Feature Store 테스트."""


import pandas as pd
import pytest

from src.features.store import FeatureStore


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Age": [25, 30, 22, 35, 28],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Location": ["USA", "Europe", "Asia", "USA", "Europe"],
        "GameGenre": ["RPG", "FPS", "MOBA", "RPG", "Strategy"],
        "GameDifficulty": ["Medium", "Hard", "Easy", "Medium", "Hard"],
        "PlayTimeHours": [120.0, 50.0, 200.0, 80.0, 150.0],
        "SessionsPerWeek": [5, 3, 7, 2, 6],
        "AvgSessionDurationMinutes": [45.0, 30.0, 60.0, 20.0, 55.0],
        "PlayerLevel": [32, 15, 55, 10, 40],
        "AchievementsUnlocked": [15, 5, 30, 3, 20],
        "InGamePurchases": [8, 2, 15, 0, 10],
        "EngagementLevel": ["High", "Medium", "High", "Low", "High"],
    })


@pytest.fixture
def store(tmp_path):
    return FeatureStore(store_dir=tmp_path / "test_store")


def test_register_and_load(store, sample_data):
    store.register_training_data(sample_data)

    # 메타데이터 파일 생성 확인
    assert (store.store_dir / "metadata.json").exists()
    assert (store.store_dir / "feature_stats.json").exists()
    assert (store.store_dir / "encoders").exists()

    # 리로드
    store2 = FeatureStore(store_dir=store.store_dir)
    store2.load()
    assert store2.metadata["n_samples"] == 5
    assert len(store2.encoder_maps) > 0


def test_transform_for_serving(store, sample_data):
    store.register_training_data(sample_data)

    single = sample_data.iloc[:1].copy()
    result = store.transform_for_serving(single)
    assert len(result) == 1
    # 범주형이 int로 변환되었는지 확인
    assert result["Gender"].dtype in ("int64", "int32", "float64")


def test_validate_serving_data(store, sample_data):
    store.register_training_data(sample_data)

    result = store.validate_serving_data(sample_data)
    assert isinstance(result, dict)
    assert "valid" in result
    assert "warnings" in result


def test_encoder_consistency(store, sample_data):
    """학습과 서빙에서 동일한 인코딩이 적용되는지 확인."""
    store.register_training_data(sample_data)

    # 동일 데이터 변환 시 일관된 결과
    r1 = store.transform_for_serving(sample_data.iloc[:1].copy())
    r2 = store.transform_for_serving(sample_data.iloc[:1].copy())
    assert r1["Gender"].values[0] == r2["Gender"].values[0]
