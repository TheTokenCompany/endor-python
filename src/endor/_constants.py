"""Internal constants: defaults, environment variables, header names and limits."""

from __future__ import annotations

DEFAULT_BASE_URL = "https://api.endor.thetokencompany.com"
DEFAULT_TIMEOUT = 30.0  # seconds per HTTP operation (a future poll adds its server-side wait on top)

API_KEY_ENV = "ENDOR_API_KEY"
BASE_URL_ENV = "ENDOR_BASE_URL"
DEFAULT_MODEL_ENV = "ENDOR_DEFAULT_MODEL"
LOG_LEVEL_ENV = "ENDOR_LOG_LEVEL"

SDK_NAME = "endor-python"
LOGGER_NAME = "endor"

# Request headers the SDK sends on every call (see "What the SDK sends" in the README).
HEADER_SDK = "X-Endor-SDK"
HEADER_SDK_VERSION = "X-Endor-SDK-Version"
HEADER_RUNTIME = "X-Endor-Runtime"
HEADER_SDK_INTERFACE = "X-Endor-SDK-Interface"
HEADER_SDK_METHOD = "X-Endor-SDK-Method"
HEADER_SDK_RECIPE = "X-Endor-SDK-Recipe"
HEADER_RETRY_COUNT = "X-Endor-Retry-Count"
HEADER_CLIENT_REQUEST_ID = "X-Endor-Client-Request-Id"
HEADER_IDEMPOTENCY_KEY = "Idempotency-Key"
HEADER_REQUEST_ID = "x-request-id"
HEADER_RETRY_AFTER = "retry-after"
HEADER_RETRY_AFTER_MS = "retry-after-ms"

# Server limits the SDK enforces or works around.
MAX_DATUMS_PER_CALL = 1024
MAX_UPLOAD_ROWS = 50_000
MAX_QUESTIONS_PER_REQUEST = 64
MAX_FUTURES_PER_RETRIEVE = 256
FUTURE_POLL_WAIT_S = 25.0  # server-side long poll per request (server clamps at 30)
MAX_ERROR_BODY_LENGTH = 200
LIST_PAGE_SIZE = 100  # page size when a list call pages through every item

# Model names saved from the SDK: the API's name pattern, and never "base" (``<project>/base`` is the project's base
# model).
SDK_MODEL_NAME = r"^[a-z0-9][a-z0-9._-]{0,62}$"
BASE_MODEL_NAME = "base"

# Error codes that are never retried, whatever their status: waiting a few seconds doesn't fix them.
NON_RETRYABLE_CODES = frozenset({"limit_reached", "insufficient_balance"})

SECRET_HEADERS = frozenset({"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie", "set-cookie"})
