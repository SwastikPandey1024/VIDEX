"""Shared test fixtures and configuration.

Fixtures available to all tests without import:
- ``settings``: Clean Settings instance with test overrides.
- ``test_app``: FastAPI test application.
- ``client``: Synchronous TestClient for the test app.
"""

import pytest
from fastapi.testclient import TestClient

from videx.api.main import create_app
from videx.config import Settings, get_settings


@pytest.fixture()
def settings() -> Settings:
    """Return a Settings instance suitable for tests.

    Uses environment variable overrides to avoid reading a real .env file.
    """
    return Settings(
        app_name="VIDEX-test",
        version="0.1.0",
        debug=True,
        database_url="",  # no DB in Phase 0
    )


@pytest.fixture()
def test_app(settings: Settings):  # type: ignore[no-untyped-def]
    """Return a FastAPI application configured for testing."""
    get_settings.cache_clear()
    application = create_app(settings=settings)
    yield application
    get_settings.cache_clear()


@pytest.fixture()
def client(test_app) -> TestClient:  # type: ignore[no-untyped-def]
    """Return a synchronous TestClient bound to the test application."""
    return TestClient(test_app)
