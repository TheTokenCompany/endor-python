"""Internal constants: defaults, environment variables, header names and limits."""

from __future__ import annotations

DEFAULT_BASE_URL = "https://api.endor.thetokencompany.com"
DEFAULT_TIMEOUT = 30.0  # seconds per HTTP operation (a future poll adds its server-side wait on top)
# A decision on a model scaled to zero waits while it starts (a few minutes). The API waits up to 300 s, then answers
# 503 warming_up with Retry-After, and the retry is served by the server that started. So a decision waits longer than
# the API, and its retries get a budget that fits that wait and one retry.
DECISION_TIMEOUT = 330.0
DECISION_RETRY_BUDGET = 660.0

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

# Model names saved from the SDK: the API's name pattern, and never "base" (reserved by the API, as are the base model
# ids: ``<project>/<base id>`` calls a base model).
SDK_MODEL_NAME = r"^[a-z0-9][a-z0-9._-]{0,62}$"
BASE_MODEL_NAME = "base"

# Error codes that are never retried, whatever their status: waiting a few seconds doesn't fix them.
NON_RETRYABLE_CODES = frozenset({"limit_reached", "insufficient_balance"})

SECRET_HEADERS = frozenset({"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie", "set-cookie"})
