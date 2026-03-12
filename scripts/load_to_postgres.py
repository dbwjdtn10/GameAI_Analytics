"""PostgreSQL에 게임 데이터 적재 스크립트.

사용법:
    docker-compose up -d postgres
    python scripts/load_to_postgres.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import init_schema, load_to_postgres
from src.data.loader import load_gaming_behavior


def main():
    print("PostgreSQL 스키마 초기화...")
    try:
        init_schema()
    except Exception as e:
        print(f"  스키마 초기화 실패: {e}")
        print("  PostgreSQL이 실행 중인지 확인하세요.")
        print("  실행: docker-compose up -d postgres")
        return

    print("데이터 로드 중...")
    df = load_gaming_behavior()
    print(f"  {len(df)}행 로드 완료")

    print("PostgreSQL에 적재 중...")
    load_to_postgres(df, table_name="players")

    print("\n적재 완료! SQL 쿼리로 확인:")
    print("  docker exec -it gameai_analytics-postgres-1 psql -U gameai -d gameai")
    print("  SELECT game_genre, COUNT(*), ROUND(AVG(is_churned::int)*100, 2) AS churn_pct")
    print("  FROM players GROUP BY game_genre ORDER BY churn_pct DESC;")


if __name__ == "__main__":
    main()
