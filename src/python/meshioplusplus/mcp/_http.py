"""HTTP front-end for the MCP server: the companion process the browser
dataset manager talks to (``meshioplusplus-mcp --http``, doc/dashboard.md).

This is the **only** module importing Starlette/uvicorn (the ``[dashboard]``
extra). It serves two things from one process:

* a small JSON API for the dataset-manager page — ``GET /api/health``,
  ``POST /api/tools/<name>`` (every registry tool, dispatched through
  :func:`._tools.call_tool` so the sandbox and the strict-JSON sanitizer are
  the ones MCP already owns) and ``GET /api/files`` (a sandboxed binary
  download for checkpoints and prediction files);
* MCP over HTTP at ``/mcp`` for agents (FastMCP's streamable-HTTP app when
  the installed SDK has it, its SSE app otherwise).

The FastMCP app is the **root** ASGI application and our routes are added to
it, not the other way round: its own lifespan starts the streamable session
manager, and mounting it under a parent Starlette would silently skip that
lifespan. Both middlewares below are plain ASGI (not
``BaseHTTPMiddleware``) so the MCP endpoint's streaming responses pass
through untouched.

Auth is a bearer token, generated per start unless ``--token`` fixes it,
required on ``/api/*`` and ``/mcp*`` (``/api/files`` also accepts
``?token=`` because an ``<a download>`` cannot set a header). The server
binds the loopback interface by default. CORS admits the loopback dev
origins and the hosted docs site, so the GitHub-Pages copy of the page can
call a local server; Chrome's Private Network Access preflight is answered.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import secrets
from typing import Iterable, Optional, Sequence

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..__about__ import __version__
from . import _tools

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
#: Origins admitted by default besides any loopback origin on any port.
DEFAULT_ALLOWED_ORIGINS: Sequence[str] = ("https://loumalouomega.github.io",)
_LOOPBACK_ORIGIN = r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"
_PROTECTED_PREFIXES = ("/api/", "/mcp")


def new_token() -> str:
    """A fresh bearer token (24 random bytes, URL-safe)."""
    return secrets.token_urlsafe(24)


def default_runs_dir(root: Optional[str]) -> str:
    """Where training runs land by default: ``<root or cwd>/runs``."""
    return os.path.join(root or os.getcwd(), "runs")


def origin_regex(allowed_origins: Iterable[str]) -> str:
    """One regex admitting every loopback origin plus the explicit ones."""
    parts = [_LOOPBACK_ORIGIN]
    for origin in allowed_origins:
        parts.append(re.escape(origin.rstrip("/")))
    return "^(" + "|".join(parts) + ")$"


def _error(status: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        {"error": message, "error_type": error_type}, status_code=status
    )


class _TokenMiddleware:
    """Require ``Authorization: Bearer <token>`` on the protected prefixes.

    Preflights (OPTIONS) pass, since a browser sends them without headers;
    ``/api/files`` additionally accepts ``?token=`` for download links. Sits
    *inside* the CORS layer so a 401 still carries CORS headers and the
    browser reports it as a 401 rather than an opaque network error.
    """

    def __init__(self, app: ASGIApp, token: Optional[str]) -> None:
        self.app = app
        self.token = token

    def _authorized(self, scope: Scope) -> bool:
        if self.token is None:
            return True
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope["headers"]
        }
        auth = headers.get("authorization", "")
        if auth.startswith("Bearer ") and secrets.compare_digest(auth[7:], self.token):
            return True
        if scope["path"] == "/api/files":
            query = scope.get("query_string", b"").decode("latin-1")
            for pair in query.split("&"):
                key, _, value = pair.partition("=")
                if key == "token" and secrets.compare_digest(value, self.token):
                    return True
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] != "OPTIONS":
            path = scope["path"]
            if any(
                path.startswith(p) for p in _PROTECTED_PREFIXES
            ) and not self._authorized(scope):
                response = _error(
                    401, "missing or invalid bearer token", "PermissionError"
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class _PrivateNetworkMiddleware:
    """Answer Chrome's Private Network Access preflight: a public (https)
    page calling a loopback server must be told the server consents."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "OPTIONS":
            await self.app(scope, receive, send)
            return
        wants = any(
            k == b"access-control-request-private-network"
            and v.strip().lower() == b"true"
            for k, v in scope["headers"]
        )

        async def send_with_pna(message):
            if wants and message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"access-control-allow-private-network", b"true"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_pna)


def build_app(
    server,
    *,
    token: Optional[str],
    allowed_origins: Iterable[str] = DEFAULT_ALLOWED_ORIGINS,
    root: Optional[str] = None,
    runs_dir: Optional[str] = None,
    watch_jobs: bool = False,
) -> Starlette:
    """The ASGI app: FastMCP's HTTP app plus the ``/api`` routes and auth.

    ``watch_jobs`` starts the job watcher, which polls live training jobs so
    a terminal transition -- and its webhook -- is noticed with nobody
    asking; only worth running when a webhook is configured, since every
    request already re-derives the jobs it touches.
    """
    if hasattr(server, "streamable_http_app"):
        app = server.streamable_http_app()
        transport = "streamable-http"
    else:  # pragma: no cover - older SDKs in the >=1.2 range
        app = server.sse_app()
        transport = "sse"
    if runs_dir:
        _tools.set_runs_dir(runs_dir)
    runs = _tools.get_runs_dir()
    if watch_jobs:
        from ._jobs import Watcher

        app.state.job_watcher = Watcher(_tools._jobs_manager()).start()
    app.state.meshioplusplus = {
        "token": token,
        "transport": transport,
        "root": root,
        "runs_dir": runs,
    }

    async def health(request: Request) -> JSONResponse:
        info = request.app.state.meshioplusplus
        return JSONResponse(
            {
                "version": __version__,
                "root": info["root"],
                "runs_dir": info["runs_dir"],
                "tools": list(_tools.TOOL_REGISTRY),
                "mcp": "/mcp",
                "transport": info["transport"],
                "auth": "token" if info["token"] else "none",
            }
        )

    async def call(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        if name not in _tools.TOOL_REGISTRY:
            return _error(404, f"meshio++: mcp: unknown tool '{name}'", "KeyError")
        body = await request.body()
        try:
            kwargs = json.loads(body) if body.strip() else {}
        except ValueError as e:
            return _error(
                400, f"meshio++: mcp: the request body is not JSON: {e}", "ValueError"
            )
        if not isinstance(kwargs, dict):
            return _error(
                400,
                "meshio++: mcp: the request body must be a JSON object",
                "TypeError",
            )
        # Tool failures come back as {"error", "error_type"} payloads with 200,
        # exactly as over MCP -- the client acts on them, never on a status.
        report = await run_in_threadpool(_tools.call_tool, name, kwargs)
        return JSONResponse(report)

    async def files(request: Request):
        path = request.query_params.get("path")
        if not path:
            return _error(400, "meshio++: mcp: 'path' is required", "ValueError")
        try:
            resolved = _tools._resolve(path, must_exist=True)
        except ValueError as e:
            return _error(404, str(e), "ValueError")
        # `must_exist` also admits a directory, because two mesh formats ARE
        # directories -- but this endpoint serves a download, so a directory
        # is a 404 here rather than a 500 out of FileResponse.
        if not os.path.isfile(resolved):
            return _error(404, f"meshio++: mcp: not a file: '{resolved}'", "ValueError")
        return FileResponse(resolved, filename=os.path.basename(resolved))

    app.add_route("/api/health", health, methods=["GET"])
    app.add_route("/api/tools/{name}", call, methods=["POST"])
    app.add_route("/api/files", files, methods=["GET"])
    # add_middleware inserts OUTERMOST: token (inner) -> CORS -> PNA (outer).
    app.add_middleware(_TokenMiddleware, token=token)
    cors_kwargs = {}
    # Starlette >= 0.46 validates Chrome's Private Network Access preflight
    # itself and REJECTS it (400 "Disallowed CORS private-network") unless
    # told to allow it, and then emits the allow header itself; the shim is
    # installed only on older releases, or the header would appear twice.
    native_pna = (
        "allow_private_network" in inspect.signature(CORSMiddleware.__init__).parameters
    )
    if native_pna:
        cors_kwargs["allow_private_network"] = True
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=origin_regex(allowed_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        allow_credentials=False,
        **cors_kwargs,
    )
    if not native_pna:
        app.add_middleware(_PrivateNetworkMiddleware)
    return app


def serve(
    app: Starlette, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> None:
    """Run the app under uvicorn, printing how the dataset manager connects."""
    import uvicorn

    info = app.state.meshioplusplus
    url = f"http://{host}:{port}"
    print(f"meshio++ companion process at {url}")
    print(
        f"  dataset manager: connect to {url}"
        + (f" with token {info['token']}" if info["token"] else " (no token)")
    )
    print(f"  MCP over HTTP ({info['transport']}): {url}/mcp")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "  WARNING: bound to a non-loopback interface; anyone reaching it with the token can read and write files under the root"
        )
    uvicorn.run(app, host=host, port=port, log_level="warning")
