"""XGBoost/LightGBM → ONNX 변환 및 추론 모듈.

ONNX Runtime으로 추론 속도를 최적화한다.
"""

import time

import joblib
import numpy as np
import onnxruntime as ort

from src.config import MODEL_DIR, ONNX_MODEL_PATH


def convert_model_to_onnx(model_path=None, output_path=None):
    """학습된 모델을 ONNX 형식으로 변환.

    Args:
        model_path: joblib 모델 경로 (기본: models/best_model.joblib)
        output_path: ONNX 출력 경로 (기본: models/best_model.onnx)

    Returns:
        ONNX 모델 경로
    """
    from onnxmltools import convert_lightgbm, convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    if model_path is None:
        model_path = MODEL_DIR / "best_model.joblib"
    if output_path is None:
        output_path = ONNX_MODEL_PATH

    model = joblib.load(model_path)
    model_type = type(model).__name__

    # 피처 수 결정
    feature_path = MODEL_DIR / "feature_names.txt"
    if feature_path.exists():
        n_features = len(feature_path.read_text().strip().split("\n"))
    else:
        n_features = model.n_features_in_

    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    if model_type == "XGBClassifier":
        onnx_model = convert_xgboost(model, initial_types=initial_type)
    elif model_type == "LGBMClassifier":
        onnx_model = convert_lightgbm(model, initial_types=initial_type)
    else:
        raise ValueError(f"Unsupported model type for ONNX conversion: {model_type}")

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


class ONNXModelService:
    """ONNX Runtime 기반 추론 서비스."""

    def __init__(self, model_path=None):
        self._session = None
        self._model_path = model_path or ONNX_MODEL_PATH

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def load(self):
        """ONNX 모델 로드."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self._model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """ONNX 추론 (확률 반환).

        Args:
            features: (N, n_features) 형태의 float32 배열

        Returns:
            (N, 2) 확률 배열
        """
        if not self.is_loaded:
            raise RuntimeError("ONNX model not loaded")

        input_name = self._session.get_inputs()[0].name
        features = features.astype(np.float32)

        result = self._session.run(None, {input_name: features})
        # ONNX 분류 모델은 [labels, probabilities] 반환
        return np.array(result[1])


def benchmark_inference(n_samples: int = 1000, n_warmup: int = 10, n_runs: int = 100):
    """joblib vs ONNX 추론 속도 벤치마크.

    Args:
        n_samples: 테스트 샘플 수
        n_warmup: 워밍업 횟수
        n_runs: 벤치마크 반복 횟수

    Returns:
        {"joblib": {p50, p95, p99, mean}, "onnx": {p50, p95, p99, mean}}
    """
    feature_path = MODEL_DIR / "feature_names.txt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature names not found: {feature_path}")
    n_features = len(feature_path.read_text().strip().split("\n"))
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    results = {}

    # joblib 모델 벤치마크
    joblib_model = joblib.load(MODEL_DIR / "best_model.joblib")
    joblib_times = []
    for _ in range(n_warmup):
        joblib_model.predict_proba(X)
    for _ in range(n_runs):
        start = time.perf_counter()
        joblib_model.predict_proba(X)
        joblib_times.append(time.perf_counter() - start)

    results["joblib"] = _compute_percentiles(joblib_times)

    # ONNX 모델 벤치마크
    if ONNX_MODEL_PATH.exists():
        onnx_svc = ONNXModelService()
        onnx_svc.load()
        onnx_times = []
        for _ in range(n_warmup):
            onnx_svc.predict_proba(X)
        for _ in range(n_runs):
            start = time.perf_counter()
            onnx_svc.predict_proba(X)
            onnx_times.append(time.perf_counter() - start)

        results["onnx"] = _compute_percentiles(onnx_times)
    else:
        results["onnx"] = None

    return results


def _compute_percentiles(times: list[float]) -> dict:
    """시간 리스트에서 p50/p95/p99/mean 계산 (ms 단위)."""
    arr = np.array(times) * 1000  # s → ms
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "samples": len(times),
    }


if __name__ == "__main__":
    print("Converting model to ONNX...")
    convert_model_to_onnx()

    print("\nRunning benchmark...")
    results = benchmark_inference(n_samples=100, n_runs=50)

    for name, stats in results.items():
        if stats is None:
            print(f"\n{name}: not available")
            continue
        print(f"\n{name} ({stats['samples']} runs, 100 samples):")
        print(f"  p50:  {stats['p50_ms']:.2f} ms")
        print(f"  p95:  {stats['p95_ms']:.2f} ms")
        print(f"  p99:  {stats['p99_ms']:.2f} ms")
        print(f"  mean: {stats['mean_ms']:.2f} ms")
