"""GET /health — liveness check endpoint."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    service: str
    status: str
    version: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Returns service name, status, and version. Used by load balancers and CI.",
)
async def health_check(request: Request) -> HealthResponse:
    """Return service health status.

    Always returns HTTP 200 with ``status: ok`` while the service is running.
    """
    settings = request.app.state.settings
    return HealthResponse(
        service=settings.app_name,
        status="ok",
        version=settings.version,
    )
