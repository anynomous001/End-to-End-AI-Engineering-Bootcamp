from fastapi import FastAPI, Request
from pydantic import BaseModel

from api.endpoints import api_router
from api.middleware import RequestIDMiddleware
from fastapi.middleware.cors import CORSMiddleware

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)




app = FastAPI()

app.add_middleware(RequestIDMiddleware)
app.include_router(api_router, prefix="/api/v1")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)


