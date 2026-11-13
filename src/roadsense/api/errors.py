"""The API's error shape: ``{"error": {"code": ..., "message": ...}}``.

Every failure, including FastAPI's own validation errors, unknown paths and
unexpected exceptions, is answered in this shape.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)

# Codes for errors raised by Starlette itself (unknown path, wrong method, ...).
_HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "bad_request",
    413: "too_large",
    422: "invalid_input",
}


class ApiError(Exception):
    """An error with an HTTP status and a contract ``code``."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _api_error(request: Request, exc: ApiError) -> JSONResponse:
    return error_response(exc.status, exc.code, exc.message)


def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _HTTP_CODES.get(exc.status_code, "internal" if exc.status_code >= 500 else "bad_request")
    message = {404: "Not found."}.get(exc.status_code) or str(exc.detail)
    return error_response(exc.status_code, code, message)


def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"][1:])
        parts.append(f"{where}: {err['msg']}" if where else str(err["msg"]))
    return error_response(422, "invalid_input", "Invalid request: " + "; ".join(parts))


def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return error_response(500, "internal", "An internal error occurred. See the server log.")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected)
