from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope


class SPAStaticFiles(StaticFiles):
    """Serve static assets and fall back to index.html for client-side routes."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            is_api_path = path == "api" or path.startswith("api/")
            looks_like_asset = bool(Path(path).suffix)
            if exc.status_code != 404 or is_api_path or looks_like_asset:
                raise
            return await super().get_response("index.html", scope)


def mount_frontend(app: FastAPI, directory: str | Path | None) -> bool:
    """Mount a built frontend when present and report whether it was mounted."""
    if directory is None:
        return False
    frontend_directory = Path(directory)
    if not (frontend_directory / "index.html").is_file():
        return False
    app.mount(
        "/",
        SPAStaticFiles(directory=frontend_directory, html=True),
        name="frontend",
    )
    return True
