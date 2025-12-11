"""Health check MCP tools for session-mgmt-mcp.

Provides health status endpoints compatible with Docker and Kubernetes
orchestration systems.

Phase 10.1: Production Hardening - Health Check Tools
"""

from __future__ import annotations

import time
import typing as t

from mcp_common.health import HealthCheckResponse

# Server start time for uptime calculation
_SERVER_START_TIME = time.time()


async def get_health_status(ready: bool = False) -> dict[str, t.Any]:
    """Get comprehensive health status of the session management server.

    Args:
        ready: If True, use readiness check logic (stricter, for K8s readiness probes)
               If False, use liveness check logic (looser, for K8s liveness probes)

    Returns:
        Dictionary with health status suitable for JSON serialization

    Example Response:
        {
            "status": "healthy",
            "timestamp": "2025-10-28T12:00:00Z",
            "version": "1.0.0",
            "uptime_seconds": 3600.5,
            "components": [
                {
                    "name": "database",
                    "status": "healthy",
                    "message": "Database operational",
                    "latency_ms": 12.5
                },
                ...
            ]
        }

    Usage:
        # Kubernetes liveness probe (checks if server should be restarted)
        await get_health_status(ready=False)

        # Kubernetes readiness probe (checks if server should receive traffic)
        await get_health_status(ready=True)

        # Docker health check
        await get_health_status()

    """
    from session_buddy.health_checks import get_all_health_checks

    # Get server version
    try:
        from session_buddy import __version__

        version = __version__
    except (ImportError, AttributeError):
        version = "unknown"

    # Run all health checks
    components = await get_all_health_checks()

    # Create health response
    response = HealthCheckResponse.create(
        components=components,  # type: ignore[arg-type]  # ComponentHealth structurally compatible
        version=version,
        start_time=_SERVER_START_TIME,
        metadata={"check_type": "readiness" if ready else "liveness"},
    )

    # For readiness checks, be strict (only HEALTHY passes)
    # For liveness checks, be loose (only UNHEALTHY fails)
    if ready and not response.is_healthy():
        # Readiness check failed - server not ready for traffic
        result = response.to_dict()
        result["ready"] = False
        return result

    if not ready and not response.is_ready():
        # Liveness check failed - server should be restarted
        result = response.to_dict()
        result["alive"] = False
        return result

    # Checks passed
    result = response.to_dict()
    if ready:
        result["ready"] = True
    else:
        result["alive"] = True

    return result


__all__ = ["get_health_status"]
