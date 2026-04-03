"""Tests for src.utils.logger."""

import logging

from src.utils.logger import setup_logger


def test_setup_logger_returns_logger() -> None:
    logger = setup_logger("test.returns")
    assert isinstance(logger, logging.Logger)


def test_setup_logger_has_handler() -> None:
    logger = setup_logger("test.handler")
    assert len(logger.handlers) >= 1


def test_setup_logger_level() -> None:
    logger = setup_logger("test.level", level="DEBUG")
    assert logger.level == logging.DEBUG


def test_setup_logger_no_duplicate_handlers() -> None:
    name = "test.duplicates"
    logger1 = setup_logger(name)
    handler_count = len(logger1.handlers)
    logger2 = setup_logger(name)
    assert len(logger2.handlers) == handler_count
    assert logger1 is logger2
