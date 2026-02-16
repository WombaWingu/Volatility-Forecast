"""
Logging utilities for tracking runs and decisions.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

from volforecast import paths as vpaths


def setup_logger(run_id: str | None = None, date_str: str | None = None) -> logging.Logger:
    """
    Set up logger that writes to artifacts/logs/<DATE>/run.log
    
    Args:
        run_id: Optional run ID to include in log filename
        date_str: Optional date string (default: today)
    
    Returns:
        Configured logger
    """
    if date_str is None:
        date_str = date.today().isoformat()
    
    logs_dir = vpaths.get_logs_dir(date_str)
    log_file = logs_dir / f"run_{run_id or datetime.now().strftime('%H%M%S')}.log"
    
    logger = logging.getLogger("volforecast")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_run_info(
    logger: logging.Logger,
    ticker: str,
    start: str,
    end: str,
    model_used: str | None = None,
    shares: int | None = None,
    **kwargs
):
    """Log run information."""
    logger.info(f"=== Run Info ===")
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Start: {start}")
    logger.info(f"End: {end}")
    if model_used:
        logger.info(f"Model used: {model_used}")
    if shares is not None:
        logger.info(f"Computed shares: {shares}")
    for key, value in kwargs.items():
        logger.info(f"{key}: {value}")
