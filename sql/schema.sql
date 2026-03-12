-- GameAI Analytics - PostgreSQL 스키마 정의
-- 게임 유저 행동 데이터 테이블

CREATE TABLE IF NOT EXISTS players (
    player_id       SERIAL PRIMARY KEY,
    age             INTEGER NOT NULL CHECK (age BETWEEN 10 AND 100),
    gender          VARCHAR(10) NOT NULL,
    location        VARCHAR(50) NOT NULL,
    game_genre      VARCHAR(30) NOT NULL,
    game_difficulty  VARCHAR(10) NOT NULL CHECK (game_difficulty IN ('Easy', 'Medium', 'Hard')),
    play_time_hours  FLOAT NOT NULL DEFAULT 0,
    sessions_per_week INTEGER NOT NULL DEFAULT 0,
    avg_session_duration_minutes FLOAT NOT NULL DEFAULT 0,
    player_level     INTEGER NOT NULL DEFAULT 1,
    achievements_unlocked INTEGER NOT NULL DEFAULT 0,
    in_game_purchases INTEGER NOT NULL DEFAULT 0,
    engagement_level VARCHAR(10) NOT NULL CHECK (engagement_level IN ('Low', 'Medium', 'High')),
    is_churned       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_players_churned ON players(is_churned);
CREATE INDEX IF NOT EXISTS idx_players_genre ON players(game_genre);
CREATE INDEX IF NOT EXISTS idx_players_engagement ON players(engagement_level);
CREATE INDEX IF NOT EXISTS idx_players_level ON players(player_level);

-- 예측 결과 저장 테이블
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    SERIAL PRIMARY KEY,
    player_id        INTEGER REFERENCES players(player_id),
    churn_probability FLOAT NOT NULL,
    risk_level       VARCHAR(10) NOT NULL,
    segment          VARCHAR(20),
    model_version    VARCHAR(50),
    predicted_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_level);

-- 모델 메타데이터 테이블
CREATE TABLE IF NOT EXISTS model_registry (
    model_id         SERIAL PRIMARY KEY,
    model_name       VARCHAR(100) NOT NULL,
    model_type       VARCHAR(50) NOT NULL,
    auc_roc          FLOAT,
    f1_score         FLOAT,
    accuracy         FLOAT,
    feature_count    INTEGER,
    is_active        BOOLEAN DEFAULT FALSE,
    trained_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
