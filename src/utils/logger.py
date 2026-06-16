"""
Logging configuration for the project.
Sets up structured logging with both file and console output.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Configure a logger with file and console handlers.

    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console output.

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__, log_level="DEBUG", log_file="logs/app.log")
        >>> logger.info("Application started")
    """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler with UTF-8 encoding (fixes Windows emoji issues)
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    console_handler.stream.reconfigure(encoding='utf-8')  # Force UTF-8 on Windows
    logger.addHandler(console_handler)

    # File handler (if log file specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger for the application
logger = setup_logger(__name__, log_level="INFO", log_file="logs/app.log")
