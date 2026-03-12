"""합성 데이터 생성 스크립트."""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SYNTHETIC_DIR
from src.data.synthetic import generate_synthetic_data


def main():
    parser = argparse.ArgumentParser(description="합성 게임 유저 데이터 생성")
    parser.add_argument("--num_users", type=int, default=10000, help="생성할 유저 수")
    parser.add_argument("--days", type=int, default=90, help="관찰 기간 (일)")
    parser.add_argument("--churn_rate", type=float, default=0.25, help="전체 이탈률")
    parser.add_argument("--event_effect", type=float, default=0.15, help="이벤트 효과")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    print(f"합성 데이터 생성 시작: {args.num_users}명, {args.days}일")

    df = generate_synthetic_data(
        num_users=args.num_users,
        days=args.days,
        churn_rate=args.churn_rate,
        event_effect=args.event_effect,
        seed=args.seed,
    )

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SYNTHETIC_DIR / "synthetic_users.csv"
    df.to_csv(output_path, index=False)

    print(f"생성 완료: {output_path}")
    print(f"  총 유저: {len(df)}")
    print(f"  이탈률: {df['is_churned'].mean():.2%}")
    print(f"  유저 타입 분포:")
    for user_type, count in df["user_type"].value_counts().items():
        print(f"    {user_type}: {count} ({count/len(df):.1%})")


if __name__ == "__main__":
    main()
