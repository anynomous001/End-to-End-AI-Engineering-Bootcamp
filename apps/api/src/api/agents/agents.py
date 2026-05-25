from api.agents.state import State, ToolCall, RAGUsedContext
from api.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from openai import OpenAI
import instructor
from jinja2 import Template
from langsmith import traceable
from langchain_core.messages import convert_to_openai_messages



class AgentResponse(BaseModel):
    answer: str = Field(
        description="The answer to the question based on your current knowledge and the tool results. "
                    "Should contain detailed product specs in bullet points."
    )
    tool_calls: List[ToolCall] = Field(
        description="The list of tool calls to make if more information is needed. Should be empty if final_answer is True."
    )
    final_answer: bool = Field(
        description="True if you have all the information needed to answer the question, False otherwise."
    )
    references: List[RAGUsedContext] = Field(
        description="List of references from the retrieved context used to compile the answer."
    )

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o-mini"}
)
def agent_node(state: State) -> dict:
    template = prompt_template_config("api/agents/prompts/qa_agent.yaml", "qa_agent")
    prompt = template.render(
        available_tools=state.available_tools
    )
    # Fixed typo: 'state.eessages' -> 'state.messages'
    messages = state.messages
    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))
    client = instructor.from_openai(OpenAI())
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )
    ai_message = format_ai_message(response)
    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references
    }



## Intent Router Node

class IntentRouterResponse(BaseModel):
    question_relevant: bool = Field(
        description="True if the question is relevant to products, product discovery, or shopping. False otherwise."
    )
    answer: str = Field(
        description="Brief explanation of why the question is not relevant (only set if question_relevant is False, otherwise leave empty '')."
    )


@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o-mini"} # Fixed 'gpt-4.1-mini' -> 'gpt-4o-mini'
)
def intent_router_node(state: State):
    template = prompt_template_config("api/agents/prompts/intent_router.yaml", "intent_router")

    first_message = state.messages[0]
    query = first_message.content if hasattr(first_message, "content") else first_message.get("content", "")


    prompt = template.render(
        query=query
    )
    client = instructor.from_openai(OpenAI())
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini", # Fixed 'gpt-4.1-mini' -> 'gpt-4o-mini'
        response_model=IntentRouterResponse,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
    )
    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer
    }