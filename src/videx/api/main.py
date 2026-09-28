"""VIDEX FastAPI application factory.

The application is created via ``create_app()`` to allow:
- Dependency injection of settings in tests.
- Future middleware, lifespan events, and router registration.

The module-level ``app`` is the ASGI entrypoint used by uvicorn::

    uvicorn videx.api.main:app --reload
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from videx.api.routers import health
from videx.config import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events.

    Place startup and shutdown logic here (e.g., DB pool creation,
    provider initialisation). Currently a no-op placeholder.
    """
    # ── Startup ────────────────────────────────────────────────
    # Phase 1+: initialise database connection pool, load model providers.
    yield
    # ── Shutdown ───────────────────────────────────────────────
    # Phase 1+: close DB connections, release GPU memory.


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct and configure the FastAPI application.

    Args:
        settings: Override settings (useful in tests). If ``None``,
                  the cached production settings are used.

    Returns:
        Configured FastAPI application instance.
    """
    resolved_settings = settings or get_settings()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.version,
        description="Video Intelligence — perception, tracking, OCR, ASR, and event detection.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Store settings on app state ───────────────────────────
    app.state.settings = resolved_settings

    # ── Routers ───────────────────────────────────────────────
    app.include_router(health.router)
    # Phase 1+:
    # app.include_router(videos.router, prefix="/api/v1")
    # app.include_router(evidence.router, prefix="/api/v1")
    # app.include_router(events.router, prefix="/api/v1")

    return app


# Module-level ASGI application — used by uvicorn and test clients.
app: FastAPI = create_app()
