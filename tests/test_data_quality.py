"""데이터 품질 검증 테스트."""

import pytest

from src.config import COOKIE_CATS_PATH, GAMING_BEHAVIOR_PATH
from src.data.loader import load_cookie_cats, load_gaming_behavior
from src.data.synthetic import generate_synthetic_data


@pytest.mark.skipif(not GAMING_BEHAVIOR_PATH.exists(), reason="Data file not found")
class TestGamingBehaviorQuality:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_gaming_behavior()

    def test_no_duplicate_player_ids(self):
        assert self.df["PlayerID"].is_unique

    def test_expected_columns_exist(self):
        expected = [
            "PlayerID", "Age", "Gender", "PlayTimeHours",
            "SessionsPerWeek", "PlayerLevel", "EngagementLevel",
        ]
        for col in expected:
            assert col in self.df.columns

    def test_no_null_values_in_key_columns(self):
        key_cols = ["PlayerID", "PlayTimeHours", "SessionsPerWeek", "EngagementLevel"]
        for col in key_cols:
            assert self.df[col].isna().sum() == 0, f"{col} has null values"

    def test_engagement_level_values(self):
        valid = {"Low", "Medium", "High"}
        actual = set(self.df["EngagementLevel"].unique())
        assert actual == valid

    def test_numeric_ranges(self):
        assert self.df["Age"].between(0, 120).all()
        assert (self.df["PlayTimeHours"] >= 0).all()
        assert (self.df["SessionsPerWeek"] >= 0).all()
        assert (self.df["PlayerLevel"] >= 0).all()


@pytest.mark.skipif(not COOKIE_CATS_PATH.exists(), reason="Data file not found")
class TestCookieCatsQuality:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_cookie_cats()

    def test_expected_columns(self):
        expected = ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]
        for col in expected:
            assert col in self.df.columns

    def test_version_values(self):
        valid = {"gate_30", "gate_40"}
        actual = set(self.df["version"].unique())
        assert actual == valid

    def test_retention_is_boolean(self):
        assert self.df["retention_1"].dtype == bool
        assert self.df["retention_7"].dtype == bool


class TestSyntheticDataQuality:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = generate_synthetic_data(num_users=1000, days=30, seed=42)

    def test_correct_row_count(self):
        assert len(self.df) == 1000

    def test_user_id_unique(self):
        assert self.df["user_id"].is_unique

    def test_churn_label_binary(self):
        assert set(self.df["is_churned"].unique()).issubset({0, 1})

    def test_no_negative_playtime(self):
        assert (self.df["avg_daily_playtime"] >= 0).all()
        assert (self.df["total_playtime"] >= 0).all()

    def test_user_types_valid(self):
        valid = {"hardcore", "casual", "at_risk", "returning"}
        actual = set(self.df["user_type"].unique())
        assert actual.issubset(valid)
