"""
Centralised debug logger for the Prompt Enhancer app.

Provides log_debug() used throughout the codebase.
"""

import atexit
import logging
from datetime import datetime

DEBUG_LOG_PATH = "debug.log"

_debug_logger = logging.getLogger("prompt_enhancer")
_debug_logger.setLevel(logging.DEBUG)
_debug_handler = logging.FileHandler(DEBUG_LOG_PATH, mode="w")
_debug_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
_debug_logger.addHandler(_debug_handler)
_debug_logger.info("=== Prompt Enhancer Debug Log Started at %s ===", datetime.now())
_debug_logger.info("Application initialized\n")


def log_debug(message: str):
    """Write a debug message to the log file"""
    _debug_logger.debug(message)


def shutdown_debug_log():
    """Log shutdown message"""
    _debug_logger.info("=== Application shutting down ===\n")


atexit.register(shutdown_debug_log)
