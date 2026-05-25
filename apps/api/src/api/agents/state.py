from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated
from operator import add

class RAGUsedContext(BaseModel):
    id: str = Field(
        description="The ID of the item used to answer the question"
    )
    description: str = Field(
        description="Short description of the item used to answer the question"
    )

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []
