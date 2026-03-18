"""경량 Feature Store.

오프라인(학습)과 온라인(서빙) 피처의 일관성을 보장하여
Training-Serving Skew를 방지한다.

구조:
  feature_store/
    metadata.json        # 피처 메타데이터 (타입, 통계, 스키마 해시)
    feature_stats.json   # 학습 데이터 기반 피처 통계 (mean, std, min, max)
    encoders/            # 범주형 인코더 매핑 저장
"""

import json
from pathlib import Path

import pandas as pd

from src.config import FEATURE_STORE_DIR
from src.features.engineer import engineer_gaming_behavior_features, get_feature_columns


class FeatureStore:
    """오프라인/온라인 피처 일관성을 보장하는 Feature Store."""

    def __init__(self, store_dir: Path | None = None):
        self.store_dir = store_dir or FEATURE_STORE_DIR
        self.metadata: dict = {}
        self.feature_stats: dict = {}
        self.encoder_maps: dict = {}

    def register_training_data(self, df: pd.DataFrame):
        """학습 데이터에서 피처 메타데이터와 통계를 등록.

        학습 시 1회 호출하여 피처 스키마와 통계를 저장한다.
        이후 서빙 시에도 동일한 변환이 적용되도록 한다.
        """
        self.store_dir.mkdir(parents=True, exist_ok=True)
        (self.store_dir / "encoders").mkdir(exist_ok=True)

        df_feat = engineer_gaming_behavior_features(df)
        feature_cols = get_feature_columns("kaggle")
        numeric = feature_cols["numeric"]
        categorical = feature_cols["categorical"]

        # 피처 통계 저장
        stats = {}
        for col in numeric:
            if col in df_feat.columns:
                vals = df_feat[col].dropna()
                stats[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "q25": float(vals.quantile(0.25)),
                    "q75": float(vals.quantile(0.75)),
                    "null_ratio": float(df_feat[col].isna().mean()),
                }
        self.feature_stats = stats

        # 범주형 인코더 매핑 저장 (정렬 기반 LabelEncoder와 동일한 결과)
        encoder_maps = {}
        for col in categorical:
            if col in df_feat.columns:
                unique_vals = sorted(df_feat[col].astype(str).unique())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                encoder_maps[col] = mapping

                # 개별 인코더 파일 저장
                with open(self.store_dir / "encoders" / f"{col}.json", "w") as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)

        self.encoder_maps = encoder_maps

        # 메타데이터 저장
        self.metadata = {
            "version": "1.0",
            "n_samples": len(df),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "all_features": numeric + categorical,
            "n_features": len(numeric) + len(categorical),
            "schema_hash": _compute_schema_hash(df_feat, numeric + categorical),
        }

        with open(self.store_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        with open(self.store_dir / "feature_stats.json", "w") as f:
            json.dump(self.feature_stats, f, ensure_ascii=False, indent=2)

        print(f"Feature Store registered: {len(numeric)} numeric + {len(categorical)} categorical")

    def load(self):
        """저장된 Feature Store 로드."""
        meta_path = self.store_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Feature Store not found: {self.store_dir}")

        with open(meta_path) as f:
            self.metadata = json.load(f)

        with open(self.store_dir / "feature_stats.json") as f:
            self.feature_stats = json.load(f)

        self.encoder_maps = {}
        encoders_dir = self.store_dir / "encoders"
        if encoders_dir.exists():
            for fp in encoders_dir.glob("*.json"):
                col_name = fp.stem
                with open(fp) as f:
                    self.encoder_maps[col_name] = json.load(f)

    def transform_for_serving(self, df: pd.DataFrame) -> pd.DataFrame:
        """서빙용 피처 변환 (학습과 동일한 인코딩 보장).

        Feature Store에 저장된 인코더를 사용하여
        학습-서빙 간 피처 일관성을 보장한다.
        """
        if not self.metadata:
            self.load()

        df = engineer_gaming_behavior_features(df)

        # 저장된 인코더로 범주형 변환 (Training-Serving Skew 방지)
        for col, mapping in self.encoder_maps.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        return df

    def validate_serving_data(self, df: pd.DataFrame) -> dict:
        """서빙 데이터가 학습 데이터 분포에서 크게 벗어나지 않는지 검증.

        Returns:
            {"valid": bool, "warnings": [...]}
        """
        if not self.feature_stats:
            self.load()

        warnings = []
        for col, stats in self.feature_stats.items():
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if len(vals) == 0:
                continue

            # IQR 범위를 벗어나는 값 체크
            iqr = stats["q75"] - stats["q25"]
            lower = stats["q25"] - 3 * iqr
            upper = stats["q75"] + 3 * iqr

            out_of_range = ((vals < lower) | (vals > upper)).sum()
            if out_of_range > 0:
                warnings.append(
                    f"{col}: {out_of_range} values out of training range "
                    f"[{lower:.2f}, {upper:.2f}]"
                )

        return {"valid": len(warnings) == 0, "warnings": warnings}


def _compute_schema_hash(df: pd.DataFrame, columns: list[str]) -> str:
    """피처 스키마 해시 (컬럼명 + 타입)."""
    import hashlib

    schema_str = "|".join(f"{c}:{df[c].dtype}" for c in columns if c in df.columns)
    return hashlib.md5(schema_str.encode()).hexdigest()[:12]  # noqa: S324


# 전역 Feature Store 인스턴스
feature_store = FeatureStore()
