import openai
from langsmith import traceable
from langsmith import get_current_run_tree


import os
from qdrant_client import QdrantClient
from openai import OpenAI

from api.core.config import config
import logging
from langsmith import traceable, get_current_run_tree
from pydantic import BaseModel, Field
import instructor
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue, Document, Prefetch, FusionQuery
from api.utils.prompt_management import build_prompt_jinja


logger = logging.getLogger(__name__)



class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

import socket

def get_qdrant_host():
    if "QDRANT_HOST" in os.environ:
        return os.environ["QDRANT_HOST"]
    try:
        socket.gethostbyname("qdrant")
        return "qdrant"
    except socket.error:
        return "localhost"

qdrant_client = QdrantClient(host=get_qdrant_host(), port=6333)
client = instructor.from_openai(OpenAI())





@traceable(
    name="get_embedding",
    run_type="embedding",
    metadata={
        "ls_model": "openai/text-embedding-3-small",
        "ls_provider": "openai",
        "ls_model_type": "embedding"
        }
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
            'total_tokens': response.usage.total_tokens
        }
    return response.data[0].embedding

@traceable(
    name="retrieve_products_data",
    run_type="retriever",
    metadata={
        "ls_provider": "qdrant",

    }
)
def retrieve_products_data(query: str, limit: int = 3):
    """
    Search for products based on a natural language query.
    """
    # 1. Convert the user's text query into a vector embedding
    query_vector = get_embedding(query)
    
    # 2. Search the Qdrant database for the closest matching vectors
    search_results = qdrant_client.query_points(
        collection_name="Products-collection-01-hybrid-search",
        prefetch=[
            Prefetch(
                query=query_vector,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="bm25",
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=limit
    )


    retrieve_context = []
    average_rating = []
    similarity_score = []
    context_id = []

    for result in search_results.points:
        # Extract fields from the payload and the Qdrant result object
        retrieve_context.append(result.payload["description"])
        average_rating.append(result.payload["average_rating"])
        similarity_score.append(result.score)
        context_id.append(result.payload["parent_asin"])
        
    return {
        "retrieved_context": retrieve_context,
        "retrieved_context_ratings": average_rating,
        "similarity_score": similarity_score,
        "retrieved_context_ids": context_id
    }

def process_context(context):
    formatted_context = ""
    
    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
        
    return formatted_context

def get_formatted_context(query: str, top_k: int = 5) -> str:
    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk.
    """
    # Annotate the variable type correctly:
    context: dict[str, list] = retrieve_products_data(query, limit=top_k)
    formatted_context = process_context(context)
    
    return formatted_context
