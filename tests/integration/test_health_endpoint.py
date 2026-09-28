"""Integration tests for the FastAPI health endpoint.

Uses FastAPI's TestClient (synchronous) to verify:
- GET /health returns HTTP 200.
- Response body matches HealthResponse schema.
- Service name and version come from application settings.
- OpenAPI docs endpoint is reachable.
"""

from fastapi.testclient import TestClient

from videx.config import Settings


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "service" in data
        assert "status" in data
        assert "version" in data

    def test_health_status_is_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_service_matches_settings(
        self,
        client: TestClient,
        settings: Settings,
    ) -> None:
        response = client.get("/health")
        assert response.json()["service"] == settings.app_name

    def test_health_version_matches_settings(
        self,
        client: TestClient,
        settings: Settings,
    ) -> None:
        response = client.get("/health")
        assert response.json()["version"] == settings.version

    def test_health_content_type_is_json(self, client: TestClient) -> None:
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_docs_available(self, client: TestClient) -> None:
        """OpenAPI docs should be reachable (not 404)."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client: TestClient) -> None:
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_unknown_route_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/does-not-exist")
        assert response.status_code == 404


class TestHealthWithCustomSettings:
    """Verify the health endpoint reflects injected settings."""

    def test_custom_app_name(self) -> None:
        from videx.api.main import create_app

        custom_settings = Settings(app_name="VIDEX-custom", version="9.9.9")
        app = create_app(settings=custom_settings)
        with TestClient(app) as c:
            response = c.get("/health")
            data = response.json()
            assert data["service"] == "VIDEX-custom"
            assert data["version"] == "9.9.9"
