"""PostgreSQL 데이터 적재 및 쿼리 모듈."""

import os

import pandas as pd

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://gameai:gameai@localhost:5432/gameai",
)


def get_engine():
    """SQLAlchemy 엔진 생성."""
    from sqlalchemy import create_engine

    return create_engine(DATABASE_URL)


def load_to_postgres(df: pd.DataFrame, table_name: str = "players"):
    """DataFrame을 PostgreSQL에 적재.

    Args:
        df: 적재할 데이터
        table_name: 테이블명
    """
    engine = get_engine()

    # 컬럼명을 snake_case로 변환
    column_map = {
        "PlayerID": "player_id",
        "Age": "age",
        "Gender": "gender",
        "Location": "location",
        "GameGenre": "game_genre",
        "GameDifficulty": "game_difficulty",
        "PlayTimeHours": "play_time_hours",
        "SessionsPerWeek": "sessions_per_week",
        "AvgSessionDurationMinutes": "avg_session_duration_minutes",
        "PlayerLevel": "player_level",
        "AchievementsUnlocked": "achievements_unlocked",
        "InGamePurchases": "in_game_purchases",
        "EngagementLevel": "engagement_level",
        "is_churned": "is_churned",
    }

    df_pg = df.rename(columns=column_map)
    cols = [c for c in column_map.values() if c in df_pg.columns]
    df_pg = df_pg[cols]

    df_pg.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"  {len(df_pg)}행을 {table_name} 테이블에 적재 완료")


def query_churn_by_genre() -> pd.DataFrame:
    """장르별 이탈률 조회."""
    engine = get_engine()
    sql = """
    SELECT
        game_genre,
        COUNT(*) AS user_count,
        ROUND(AVG(is_churned::int) * 100, 2) AS churn_rate_pct,
        ROUND(AVG(play_time_hours)::numeric, 1) AS avg_playtime
    FROM players
    GROUP BY game_genre
    ORDER BY churn_rate_pct DESC
    """
    return pd.read_sql(sql, engine)


def query_high_risk_users(limit: int = 100) -> pd.DataFrame:
    """고위험 유저 조회."""
    engine = get_engine()
    sql = f"""
    SELECT
        p.player_id,
        p.game_genre,
        p.player_level,
        p.play_time_hours,
        pr.churn_probability,
        pr.risk_level,
        pr.segment
    FROM players p
    JOIN predictions pr ON p.player_id = pr.player_id
    WHERE pr.risk_level IN ('high', 'critical')
    ORDER BY pr.churn_probability DESC
    LIMIT {limit}
    """
    return pd.read_sql(sql, engine)


def save_predictions(predictions_df: pd.DataFrame):
    """예측 결과를 PostgreSQL에 저장."""
    engine = get_engine()
    predictions_df.to_sql("predictions", engine, if_exists="append", index=False)
    print(f"  {len(predictions_df)}건 예측 결과 저장 완료")


def init_schema():
    """SQL 스키마 파일을 실행하여 테이블 생성."""
    from pathlib import Path

    from sqlalchemy import text

    engine = get_engine()
    schema_path = Path(__file__).resolve().parent.parent.parent / "sql" / "schema.sql"

    if not schema_path.exists():
        print(f"스키마 파일 없음: {schema_path}")
        return

    with engine.connect() as conn:
        conn.execute(text(schema_path.read_text()))
        conn.commit()
    print("스키마 초기화 완료")
