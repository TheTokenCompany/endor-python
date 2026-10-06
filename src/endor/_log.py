"""The ``endor`` logger. The SDK never configures handlers; set ``ENDOR_LOG_LEVEL`` for a quick default."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from ._constants import LOG_LEVEL_ENV, LOGGER_NAME, SECRET_HEADERS

logger = logging.getLogger(LOGGER_NAME)

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "off": logging.CRITICAL + 1,
}


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """A copy of ``headers`` with credentials replaced by ``***``. Used for every logged header set."""
    return {k: "***" if k.lower() in SECRET_HEADERS else v for k, v in headers.items()}


def _setup() -> None:
    logger.addHandler(logging.NullHandler())
    level = (os.environ.get(LOG_LEVEL_ENV) or "").strip().lower()
    if level in _LEVELS:
        logger.setLevel(_LEVELS[level])


_setup()
