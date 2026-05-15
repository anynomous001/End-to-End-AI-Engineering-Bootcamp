from pydantic import BaseModel

class RAGRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    request_id: str
    answer: str
