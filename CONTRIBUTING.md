# Contributing

## Setup

```bash
uv venv && uv pip install -e ".[dev]"      # or: pip install -e ".[dev]"
```

## Checks

```bash
ruff format src tests && ruff check src tests
mypy src/endor
pytest
```

The test suite runs against an in-memory fake of the API (`tests/fake_api.py`), so it needs no credentials and
finishes in a couple of seconds. To run the integration tests against a real deployment:

```bash
ENDOR_TEST_BASE_URL=https://... ENDOR_TEST_API_KEY=edk_... pytest tests/test_integration.py
```

## Releasing

1. Bump `__version__` in `src/endor/_version.py` and move the `Unreleased` notes in `CHANGELOG.md` under the new
   version.
2. Commit, then tag `vX.Y.Z` and push the tag. The `Publish to PyPI` workflow checks that the tag matches the
   version, runs the checks, builds the package and uploads it.

## Conventions

- Python 3.10 and up. Public functions and classes have docstrings; `mypy --strict` must pass on `src/endor`.
- Every request goes through `endor._http.Transport`, which adds the headers in `docs/REQUEST_HEADERS.md`. When you
  add a public method, pass its name as `method_name` so the header names it.
- Don't add anything to the SDK that has no API endpoint behind it.
