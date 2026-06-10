# ============================================================
# shared/utils/logging.py
# Structured logging — every log line carries correlation context
# ============================================================

import logging
import sys
import json
from datetime import datetime
from typing import Any
from shared.utils.correlation import get_correlation_id, get_tenant_id, get_layer


class SIJsonFormatter(logging.Formatter):
    """
    JSON log formatter that automatically injects correlation context.
    Every log line is parseable by Elasticsearch / Loki / any log aggregator.
    """

    LEVEL_MAP = {
        logging.DEBUG:    "DEBUG",
        logging.INFO:     "INFO",
        logging.WARNING:  "WARN",
        logging.ERROR:    "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp":       datetime.utcnow().isoformat() + "Z",
            "level":           self.LEVEL_MAP.get(record.levelno, "INFO"),
            "logger":          record.name,
            "message":         record.getMessage(),
            "correlation_id":  get_correlation_id(),
            "tenant_id":       get_tenant_id(),
            "si_layer":        get_layer(),
            "module":          record.module,
            "function":        record.funcName,
            "line":            record.lineno,
        }

        # Merge extra fields from the log call
        if hasattr(record, "__dict__"):
            for key, val in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "levelname", "levelno",
                    "pathname", "filename", "module", "exc_info",
                    "exc_text", "stack_info", "lineno", "funcName",
                    "created", "msecs", "relativeCreated", "thread",
                    "threadName", "processName", "process", "message",
                    "taskName",
                ):
                    log_data[key] = val

        # Exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """
    Configure root logger for SI. Call once at application startup.

    Args:
        level: Log level string (DEBUG, INFO, WARN, ERROR)
        json_output: If True, use JSON formatter. If False, use human-readable.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if json_output:
        handler.setFormatter(SIJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ["urllib3", "httpx", "asyncio", "kafka", "confluent_kafka"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Use in every SI module."""
    return logging.getLogger(name)