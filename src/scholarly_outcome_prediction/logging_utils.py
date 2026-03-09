"""Logging setup with optional Rich console."""

import logging
import sys

try:
    from rich.logging import RichHandler
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(
    level: int | str = logging.INFO,
    log_format: str | None = None,
) -> None:
    """Configure root logger; use Rich handler if available."""
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return
    if log_format is None:
        log_format = "%(message)s" if RICH_AVAILABLE else "%(levelname)s %(name)s: %(message)s"
    if RICH_AVAILABLE:
        handler: logging.Handler = RichHandler(
            console=Console(stderr=True),
            show_time=False,
            show_path=False,
        )
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(log_format))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
