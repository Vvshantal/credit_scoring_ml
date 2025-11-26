"""Structured logging utility for the ML Loan Eligibility Platform."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

from .config import config


def setup_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    output: Optional[str] = None,
) -> None:
    """Setup structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json or plain)
        output: Output destination (stdout or file path)
    """
    level = level or config.logging.level
    log_format = log_format or config.logging.format
    output = output or config.logging.output

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        add_log_level,
        TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging_level = getattr(logging, level.upper())

    # Create handlers
    handlers = []

    if output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
        handlers.append(handler)
    else:
        # Create log directory if it doesn't exist
        log_path = Path(config.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handlers.append(handler)

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        level=logging_level,
        handlers=handlers,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls with arguments and results.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.info(
            "function_called",
            function=func.__name__,
            args=args,
            kwargs=kwargs,
        )
        try:
            result = func(*args, **kwargs)
            logger.info(
                "function_completed",
                function=func.__name__,
                result=str(result)[:100],  # Limit result size
            )
            return result
        except Exception as e:
            logger.error(
                "function_failed",
                function=func.__name__,
                error=str(e),
                exc_info=True,
            )
            raise

    return wrapper


def log_exception(logger: structlog.BoundLogger, exc: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log an exception with context.

    Args:
        logger: Logger instance
        exc: Exception to log
        context: Additional context information
    """
    log_data = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }

    if context:
        log_data.update(context)

    logger.error("exception_occurred", **log_data, exc_info=True)


# Initialize logging on module import
setup_logging()
