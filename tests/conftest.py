"""공통 테스트 설정."""

import pytest

from src.config import COOKIE_CATS_PATH, GAMING_BEHAVIOR_PATH


def pytest_collection_modifyitems(config, items):
    """데이터 파일이 없으면 해당 테스트를 자동 스킵."""
    skip_gaming = pytest.mark.skip(reason="Gaming behavior data file not found")
    skip_cookie = pytest.mark.skip(reason="Cookie Cats data file not found")
    skip_model = pytest.mark.skip(reason="Trained model file not found")

    from src.config import MODEL_DIR

    for item in items:
        if "gaming_df" in getattr(item, "fixturenames", []):
            if not GAMING_BEHAVIOR_PATH.exists():
                item.add_marker(skip_gaming)
        if "cookie_df" in getattr(item, "fixturenames", []):
            if not COOKIE_CATS_PATH.exists():
                item.add_marker(skip_cookie)
        if "requires_data" in {m.name for m in item.iter_markers()}:
            if not GAMING_BEHAVIOR_PATH.exists():
                item.add_marker(skip_gaming)
        if "requires_model" in {m.name for m in item.iter_markers()}:
            if not (MODEL_DIR / "best_model.joblib").exists():
                item.add_marker(skip_model)
