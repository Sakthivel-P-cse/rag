import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


LOGGER_NAME = "rag_metrics"
logger = logging.getLogger(LOGGER_NAME)


def configure_metrics_logger(level: int = logging.INFO) -> None:
    """Configure the metrics logger to emit one JSON object per line.

    Call this once at process startup (e.g., in CLI entrypoints).
    """
    if logger.handlers:
        # Already configured
        return

    handler = logging.StreamHandler()

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: Dict[str, Any] = {
                "level": record.levelname,
                "message": record.getMessage(),
                "time": int(time.time() * 1000),
            }
            # If the record has an "extra" dict, merge it
            extra = getattr(record, "extra", None)
            if isinstance(extra, dict):
                payload.update(extra)
            return json.dumps(payload, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)


def log_stage(
    stage: str,
    *,
    duration_ms: Optional[float] = None,
    num_items: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured log entry for a pipeline stage.

    Example payload (as JSON):
    {
        "stage": "embedding",
        "duration_ms": 1540,
        "num_chunks": 512
    }
    """
    payload: Dict[str, Any] = {"stage": stage}
    if duration_ms is not None:
        payload["duration_ms"] = float(duration_ms)
    if num_items is not None:
        # Use a generic field name but keep it discoverable
        payload["num_items"] = int(num_items)
    if extra:
        payload.update(extra)

    logger.info("stage_completed", extra=payload)


@contextmanager
def stage_timer(stage: str, *, extra: Optional[Dict[str, Any]] = None):
    """Context manager to time a block and log the result via log_stage.

    Usage:
        with stage_timer("embedding", extra={"batch_size": 64}):
            ...
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        duration_ms = (end - start) * 1000.0
        log_stage(stage, duration_ms=duration_ms, extra=extra)
