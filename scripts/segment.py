"""유저 세그먼트 클러스터링 실행 스크립트."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.data.loader import load_gaming_behavior
from src.features.engineer import engineer_gaming_behavior_features
from src.models.segmenter import get_segment_summary, train_segmenter


def main():
    print("데이터 로드 중...")
    df = load_gaming_behavior()
    df = engineer_gaming_behavior_features(df)

    print("K-Means 세그먼트 학습 중...")
    result = train_segmenter(df, n_clusters=4)

    print(f"클러스터 매핑: {result['cluster_map']}")

    print("\n세그먼트 분포:")
    summary = get_segment_summary(df)
    print(summary.to_string())

    print("\n세그먼트 모델 저장 완료!")


if __name__ == "__main__":
    main()
