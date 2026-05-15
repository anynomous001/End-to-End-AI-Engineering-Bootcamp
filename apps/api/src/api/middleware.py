import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)



class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to generate a request ID for each incoming request and store it in the request state.
    """

    async def dispatch(self, request: Request, call_next):

        #Generate Request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        logger.info(f"Request started processing {request.method} {request.url.path} with request id: {request_id}")

        #Log incoming request
        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        logger.info(f"Request completed with status code: {response.status_code} {request.url.path} request_id: {request_id}")
        return response
