"""데이터 검증 스키마 테스트 (Pandera)."""

import pandas as pd
import pytest

from src.data.validation import (
    validate_inference_input,
    validate_raw_data,
)


@pytest.fixture
def valid_raw_data():
    return pd.DataFrame([{
        "PlayerID": 1,
        "Age": 25,
        "Gender": "Male",
        "Location": "USA",
        "GameGenre": "RPG",
        "GameDifficulty": "Medium",
        "PlayTimeHours": 120.5,
        "SessionsPerWeek": 5,
        "AvgSessionDurationMinutes": 45.0,
        "PlayerLevel": 32,
        "AchievementsUnlocked": 15,
        "InGamePurchases": 8,
        "EngagementLevel": "High",
    }])


@pytest.fixture
def valid_inference_data():
    return pd.DataFrame([{
        "Age": 25,
        "Gender": "Male",
        "Location": "USA",
        "GameGenre": "RPG",
        "GameDifficulty": "Medium",
        "PlayTimeHours": 120.5,
        "SessionsPerWeek": 5,
        "AvgSessionDurationMinutes": 45.0,
        "PlayerLevel": 32,
        "AchievementsUnlocked": 15,
        "InGamePurchases": 8,
    }])


def test_valid_raw_data_passes(valid_raw_data):
    result = validate_raw_data(valid_raw_data)
    assert len(result) == 1


def test_invalid_age_fails(valid_raw_data):
    valid_raw_data.loc[0, "Age"] = 5  # below minimum
    with pytest.raises(Exception):
        validate_raw_data(valid_raw_data)


def test_invalid_gender_fails(valid_raw_data):
    valid_raw_data.loc[0, "Gender"] = "Unknown"
    with pytest.raises(Exception):
        validate_raw_data(valid_raw_data)


def test_invalid_difficulty_fails(valid_raw_data):
    valid_raw_data.loc[0, "GameDifficulty"] = "Extreme"
    with pytest.raises(Exception):
        validate_raw_data(valid_raw_data)


def test_valid_inference_input(valid_inference_data):
    result = validate_inference_input(valid_inference_data)
    assert len(result) == 1


def test_inference_negative_playtime_fails(valid_inference_data):
    valid_inference_data.loc[0, "PlayTimeHours"] = -10
    with pytest.raises(Exception):
        validate_inference_input(valid_inference_data)
