from pydantic import BaseModel, Field
from typing import Optional, List


class RAGRequest(BaseModel):
    query: str


class RAGUsedContext(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the product image")
    price: Optional[str] = Field(None, description="Price of the product")
    description: str = Field(..., description="Description of the product")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="Request ID")
    answer: str = Field(..., description="Answer to the question")
    used_context: list[RAGUsedContext] = Field(..., description="List of items used to answer the question")
