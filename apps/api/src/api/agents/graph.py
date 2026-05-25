from langgraph.checkpoint.postgres import PostgresSaver
import logging
import numpy as np
from typing import List, Dict, Any
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from qdrant_client.models import Filter, FieldCondition, MatchValue

from api.core.config import config
from api.agents.state import State, ToolCall, RAGUsedContext
from api.agents.agents import agent_node, intent_router_node
from api.agents.tools import get_formatted_context
from api.agents.utils.utils import get_tool_descriptions
from api.agents.retrieval_generation import qdrant_client

logger = logging.getLogger(__name__)


# Conditional Edge Functions

def intent_router_conditional_edges(state: State) -> str:
    return "agent_node" if state.question_relevant else "end"


def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"


# Workflow Graph Compilation

workflow = StateGraph(State)

tools = [get_formatted_context]
tools_node = ToolNode(tools)
tools_description = get_tool_descriptions(tools)

# Nodes
workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tools_node)
workflow.add_node("intent_router_node", intent_router_node)

# Edges
workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END
    }
)

workflow.add_edge("tool_node", "agent_node")

graph = workflow.compile()


# Agent Execution Functions

@traceable(
    name="run_agent",
    run_type="chain"
)
def run_agent(question: str, thread_id: str) -> dict:
    initial_state = {
        "messages": [{
            "role": "user", 
            "content": question
        }],
        "iteration": 0,
        "available_tools": tools_description
    }
    graph_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    with PostgresSaver.from_conn_string(config.POSTGRES_URL) as checkpointer:
        graph_compiled = workflow.compile(checkpointer=checkpointer)
        result = graph_compiled.invoke(initial_state, graph_config)
    return result


@traceable(
    name="rag_agent_wrapper",
    run_type="chain"
)
def rag_agent_wrapper(question: str, thread_id: str, top_k: int = 5) -> dict:
    result = run_agent(question, thread_id)
    
    used_context = []
    dummy_vector = np.zeros(1536).tolist()
    
    for item in result.get("references", []):
        point_results = qdrant_client.query_points(
            collection_name="Products-collection-01-hybrid-search",
            query=dummy_vector,
            using="text-embedding-3-small",
            limit=1,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        )
        
        if point_results.points:
            payload = point_results.points[0].payload
            image_url = payload.get("image")
            price = payload.get("price")
            if image_url:
                used_context.append({
                    "image_url": image_url,
                    "price": str(price) if price is not None else None,
                    "description": item.description
                })
            
    return {
        "answer": result.get("answer", ""),
        "used_context": used_context,
    }
