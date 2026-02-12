import math

import pytest
from pyairtable.testing import fake_record

import sieve.utils as utils


def test_sanitize_text_basic():
    # Test basic functionality
    text = "Hello World"
    result = utils.sanitize_text(text)
    assert result == "Hello World"


def test_sanitize_text_empty_string():
    # Test with empty string
    result = utils.sanitize_text("")
    assert result == ""


def test_sanitize_text_none():
    # Test with None input
    result = utils.sanitize_text(None)
    assert result == ""


def test_sanitize_text_unicode_quotes():
    # Test replacement of unicode quotes
    text = "He said \u2018hello\u2019 and she replied \u201cgoodbye\u201d"
    result = utils.sanitize_text(text)
    assert result == "He said 'hello' and she replied \"goodbye\""


def test_sanitize_text_unicode_dashes():
    # Test replacement of unicode dashes
    text = "This is an en–dash and an em—dash"
    result = utils.sanitize_text(text)
    assert result == "This is an en-dash and an em-dash"


def test_sanitize_text_ellipsis():
    # Test replacement of ellipsis
    text = "Wait for it… here it is"
    result = utils.sanitize_text(text)
    assert result == "Wait for it... here it is"


def test_sanitize_text_newlines_and_carriage_returns():
    # Test removal of newlines and carriage returns
    text = "Line 1\nLine 2\r\nLine 3\rLine 4"
    result = utils.sanitize_text(text)
    assert result == "Line 1 Line 2 Line 3Line 4"


def test_sanitize_text_multiple_whitespace():
    # Test normalization of multiple whitespace
    text = "Too     many    spaces"
    result = utils.sanitize_text(text)
    assert result == "Too many spaces"


def test_sanitize_text_unicode_normalization():
    # Test unicode normalization (NFKD)
    text = "café"  # é as a single character
    result = utils.sanitize_text(text)
    assert result == "cafe"  # Should remove accents


def test_sanitize_text_non_ascii_removal():
    # Test removal of non-ASCII characters
    text = "Hello 世界"
    result = utils.sanitize_text(text)
    assert result == "Hello"


def test_sanitize_text_comprehensive():
    # Test comprehensive functionality with multiple issues
    text = "  \u2018Hello\u2019  world…\n\tThis is a \u201ctest\u201d—with many\r\n   issues  "  # noqa: E501
    result = utils.sanitize_text(text)
    assert result == "'Hello' world... This is a \"test\"-with many issues"


def test_sanitize_text_leading_trailing_whitespace():
    # Test removal of leading and trailing whitespace
    text = "   Hello World   "
    result = utils.sanitize_text(text)
    assert result == "Hello World"


def test_setup_logger_creates_logger():
    import logging
    import os

    log_file = "test_sieve.log"

    # Clean up if file exists
    if os.path.exists(log_file):
        os.remove(log_file)

    try:
        logger = utils.setup_logger("test_logger", log_file)

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

        # Test that it has a handler
        assert len(logger.handlers) > 0

        # Test that it can log
        logger.info("Test message")
        assert os.path.exists(log_file)

    finally:
        # Clean up
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        if os.path.exists(log_file):
            os.remove(log_file)


def test_setup_logger_reuses_existing_logger():
    import logging
    import os

    log_file = "test_sieve2.log"

    # Clean up if file exists
    if os.path.exists(log_file):
        os.remove(log_file)

    try:
        # Create logger first time
        logger1 = utils.setup_logger("test_logger2", log_file)
        handler_count1 = len(logger1.handlers)

        # Call again with same name - should not add more handlers
        logger2 = utils.setup_logger("test_logger2", log_file)
        handler_count2 = len(logger2.handlers)

        # Should be the same logger instance
        assert logger1 is logger2
        # Should not have added more handlers
        assert handler_count1 == handler_count2

    finally:
        # Clean up
        logger = logging.getLogger("test_logger2")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        if os.path.exists(log_file):
            os.remove(log_file)


def test_setup_logger_rotating_handler():
    import logging
    import os
    from logging.handlers import RotatingFileHandler

    log_file = "test_sieve3.log"

    # Clean up if file exists
    if os.path.exists(log_file):
        os.remove(log_file)

    try:
        logger = utils.setup_logger("test_logger3", log_file)

        # Check that it uses RotatingFileHandler
        assert len(logger.handlers) > 0
        handler = logger.handlers[0]
        assert isinstance(handler, RotatingFileHandler)

        # Check max bytes and backup count
        assert handler.maxBytes == 5 * 1024 * 1024  # 5MB
        assert handler.backupCount == 3

    finally:
        # Clean up
        logger = logging.getLogger("test_logger3")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        if os.path.exists(log_file):
            os.remove(log_file)


def test_create_stats_table_basic():
    from datetime import datetime, timedelta

    start_time = datetime.now() - timedelta(hours=1, minutes=30)

    stats = {
        "start_time": start_time,
        "status": "[green]Idle[/green]",
        "platforms": "Asana, Rayyan, OpenAI",
        "last_check": {"asana": "12:00:00", "rayyan": "12:01:00", "openai": "12:02:00"},
        "last_sync": {
            "asana": "2024-01-01 12:00:00",
            "rayyan": "2024-01-01 12:01:00",
            "openai": "2024-01-01 12:02:00",
        },
        "total_syncs": {"asana": 5, "rayyan": 3, "openai": 2},
        "total_polls": {"asana": 10, "rayyan": 8, "openai": 6},
        "pending_batches": {
            "abstract_screen": 2,
            "fulltext_screen": 1,
            "extraction": 0,
        },
    }

    table = utils.create_stats_table(stats)

    assert table is not None
    assert table.title == "Bigger Picker Status"
    assert len(table.columns) == 2


def test_create_stats_table_uptime_calculation():
    from datetime import datetime, timedelta

    # Test uptime formatting
    start_time = datetime.now() - timedelta(hours=2, minutes=15, seconds=30)

    stats = {
        "start_time": start_time,
        "status": "[green]Running[/green]",
        "platforms": "Test",
        "last_check": {"asana": "N/A", "rayyan": "N/A", "openai": "N/A"},
        "last_sync": {"asana": "N/A", "rayyan": "N/A", "openai": "N/A"},
        "total_syncs": {"asana": 0, "rayyan": 0, "openai": 0},
        "total_polls": {"asana": 0, "rayyan": 0, "openai": 0},
        "pending_batches": {
            "abstract_screen": 0,
            "fulltext_screen": 0,
            "extraction": 0,
        },
    }

    table = utils.create_stats_table(stats)

    # Should create table without errors
    assert table is not None


def test_create_stats_table_with_pending_batches():
    from datetime import datetime

    stats = {
        "start_time": datetime.now(),
        "status": "[yellow]Processing batches[/yellow]",
        "platforms": "All",
        "last_check": {"asana": "12:00:00", "rayyan": "12:00:00", "openai": "12:00:00"},
        "last_sync": {
            "asana": "2024-01-01 12:00:00",
            "rayyan": "2024-01-01 12:00:00",
            "openai": "2024-01-01 12:00:00",
        },
        "total_syncs": {"asana": 1, "rayyan": 1, "openai": 1},
        "total_polls": {"asana": 1, "rayyan": 1, "openai": 1},
        "pending_batches": {
            "abstract_screen": 10,
            "fulltext_screen": 5,
            "extraction": 3,
        },
    }

    table = utils.create_stats_table(stats)

    assert table is not None
    # Should have rows for all the stats including pending batches
    assert len(table.rows) > 0
