"""API 응답 시간 벤치마크.

단일/배치 예측 API의 p50/p95/p99 응답 시간을 측정한다.

사용법:
    python scripts/benchmark.py --url http://localhost:8000 --n-requests 200
"""

import argparse
import json
import random
import time

import httpx
import numpy as np

API_KEY = "dev-key-gameai-2024"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

GENRES = ["RPG", "FPS", "MOBA", "Sports", "Strategy", "Simulation", "Action", "Adventure"]
LOCATIONS = ["USA", "Europe", "Asia", "South America", "Africa", "Australia", "Other"]


def random_player():
    return {
        "Age": random.randint(10, 65),
        "Gender": random.choice(["Male", "Female"]),
        "Location": random.choice(LOCATIONS),
        "GameGenre": random.choice(GENRES),
        "GameDifficulty": random.choice(["Easy", "Medium", "Hard"]),
        "PlayTimeHours": round(random.uniform(1, 2000), 1),
        "SessionsPerWeek": random.randint(1, 30),
        "AvgSessionDurationMinutes": round(random.uniform(10, 300), 1),
        "PlayerLevel": random.randint(1, 150),
        "AchievementsUnlocked": random.randint(0, 200),
        "InGamePurchases": random.randint(0, 100),
    }


def benchmark_endpoint(client, url, method, payload, n_requests, warmup=5):
    """단일 엔드포인트 벤치마크."""
    # 워밍업
    for _ in range(warmup):
        if method == "POST":
            client.post(url, json=payload(), headers=HEADERS)
        else:
            client.get(url, headers=HEADERS)

    # 벤치마크
    latencies = []
    errors = 0
    for _ in range(n_requests):
        data = payload() if callable(payload) else payload
        start = time.perf_counter()
        try:
            if method == "POST":
                resp = client.post(url, json=data, headers=HEADERS)
            else:
                resp = client.get(url, headers=HEADERS)
            if resp.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
            continue
        latencies.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies)
    return {
        "n_requests": n_requests,
        "errors": errors,
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "throughput_rps": round(n_requests / (sum(latencies) / 1000), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="GameAI API Benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n-requests", type=int, default=200)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    n = args.n_requests

    print(f"Benchmarking {base} with {n} requests per endpoint\n")
    print("=" * 70)

    client = httpx.Client(timeout=30)

    # 1. Health Check
    print("\n[Health Check] GET /health")
    result = benchmark_endpoint(client, f"{base}/health", "GET", lambda: None, n)
    _print_result(result)

    # 2. Single Prediction
    print("\n[Single Prediction] POST /api/v1/predict/single")
    result_single = benchmark_endpoint(
        client, f"{base}/api/v1/predict/single", "POST", random_player, n
    )
    _print_result(result_single)

    # 3. Batch Prediction (10 users)
    print("\n[Batch Prediction x10] POST /api/v1/predict/batch")
    result_batch10 = benchmark_endpoint(
        client, f"{base}/api/v1/predict/batch", "POST",
        lambda: {"players": [random_player() for _ in range(10)]},
        n // 2,
    )
    _print_result(result_batch10)

    # 4. Batch Prediction (100 users)
    print("\n[Batch Prediction x100] POST /api/v1/predict/batch")
    result_batch100 = benchmark_endpoint(
        client, f"{base}/api/v1/predict/batch", "POST",
        lambda: {"players": [random_player() for _ in range(100)]},
        n // 4,
    )
    _print_result(result_batch100)

    # 5. Segment Classify
    print("\n[Segment Classify] POST /api/v1/segment/classify")
    result_segment = benchmark_endpoint(
        client, f"{base}/api/v1/segment/classify", "POST", random_player, n
    )
    _print_result(result_segment)

    client.close()

    # 결과 요약 JSON 저장
    summary = {
        "health_check": result,
        "single_prediction": result_single,
        "batch_10": result_batch10,
        "batch_100": result_batch100,
        "segment_classify": result_segment,
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to benchmark_results.json")


def _print_result(r):
    print(f"  Requests: {r['n_requests']} | Errors: {r['errors']}")
    print(f"  p50: {r['p50_ms']:.2f}ms | p95: {r['p95_ms']:.2f}ms | p99: {r['p99_ms']:.2f}ms")
    print(f"  Mean: {r['mean_ms']:.2f}ms | Throughput: {r['throughput_rps']:.1f} req/s")


if __name__ == "__main__":
    main()
