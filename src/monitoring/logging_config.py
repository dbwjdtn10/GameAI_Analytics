"""구조화된 로깅 설정 (structlog 기반)."""

import logging
import sys
import uuid

import structlog


def setup_logging(log_level: str = "INFO", json_format: bool = True):
    """애플리케이션 전역 로깅 설정.

    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        json_format: True이면 JSON 포맷, False이면 콘솔 포맷
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # uvicorn 로거도 포맷 통일
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(handler)


def generate_request_id() -> str:
    """고유 요청 ID 생성."""
    return str(uuid.uuid4())[:8]


def get_logger(name: str = "gameai"):
    """structlog 로거 반환."""
    return structlog.get_logger(name)
