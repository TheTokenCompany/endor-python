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
finishes in a few seconds. Most tests check what the fake saw or make it fail, so the suite as a whole doesn't run
against a real API. `tests/test_integration.py` does, and is skipped unless you point it at one:

```bash
ENDOR_TEST_BASE_URL=http://localhost:8000 ENDOR_TEST_API_KEY=edk_... pytest tests/test_integration.py
```

CI runs the checks on Python 3.10 to 3.14.

Every test must close every run it opens (`with client.runs.create(...) as run:`): the suite fails a test that leaves
one open. Open runs count toward an org's limit, in the fake (4) as in the API (25 by default;
`client.whoami().limits["max_active_runs"]` has an org's own).

## Releasing

1. Bump `__version__` in `src/endor/_version.py` and move the `Unreleased` notes in `CHANGELOG.md` under the new
   version.
2. Commit, then tag `vX.Y.Z` and push the tag. The `Publish to PyPI` workflow checks that the tag matches the
   version, runs the checks, builds the package and uploads it.

## Conventions

- Python 3.10 and up. Public functions and classes have docstrings; `mypy --strict` must pass on `src/endor`.
- Every request goes through `endor._http.Transport`, which adds the headers listed under "What the SDK sends" in the README. When you
  add a public method, pass its name as `method_name` so the header names it.
- Don't add anything to the SDK that has no API endpoint behind it.
