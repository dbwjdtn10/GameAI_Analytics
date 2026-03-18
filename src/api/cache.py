"""Redis 캐싱 레이어.

동일 입력에 대한 중복 추론을 방지하여 응답 속도를 향상시킨다.
"""

import hashlib
import json

import redis.asyncio as aioredis
import structlog

from src.config import CACHE_TTL_SECONDS, REDIS_URL

logger = structlog.get_logger("gameai.cache")

_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis | None:
    """Redis 클라이언트 반환 (연결 실패 시 None)."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = aioredis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await _redis_client.ping()
        except Exception as e:
            logger.warning("Redis connection failed: %s (cache disabled)", e)
            _redis_client = None
    return _redis_client


def _make_cache_key(prefix: str, data: dict) -> str:
    """입력 데이터로부터 캐시 키 생성."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    hash_val = hashlib.md5(serialized.encode()).hexdigest()  # noqa: S324
    return f"gameai:{prefix}:{hash_val}"


async def get_cached_prediction(player_data: dict) -> dict | None:
    """캐시된 예측 결과 조회."""
    client = await get_redis()
    if client is None:
        return None

    key = _make_cache_key("predict", player_data)
    try:
        cached = await client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning("Cache read error: %s", e)
    return None


async def set_cached_prediction(player_data: dict, result: dict):
    """예측 결과 캐싱."""
    client = await get_redis()
    if client is None:
        return

    key = _make_cache_key("predict", player_data)
    try:
        await client.setex(key, CACHE_TTL_SECONDS, json.dumps(result, default=str))
    except Exception as e:
        logger.warning("Cache write error: %s", e)


async def close_redis():
    """Redis 연결 종료."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
