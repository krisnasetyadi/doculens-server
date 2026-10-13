"""Central handling of unexpected errors (MS-657).

Any exception a route does not turn into an HTTPException itself ends here:
the full traceback is logged with a short error_id, and the client gets a
generic 500 carrying that id, never the exception text (which can hold SQL,
connection strings or file paths). HTTPExceptions are FastAPI's own and pass
through unchanged.
"""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    error_id = uuid.uuid4().hex[:12]
    logger.error("Unhandled error %s on %s %s: %s", error_id, request.method, request.url.path, exc,
                 exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error_id": error_id})


def register(app: FastAPI) -> None:
    app.add_exception_handler(Exception, handle_unexpected)
