import os
from qdrant_client import QdrantClient
from openai import OpenAI
from groq import Groq
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



# Initialize clients
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY
)
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
groq_client = Groq(api_key=config.GROQ_API_KEY)
client = instructor.from_openai(OpenAI())




@traceable(
    name="get_embedding",
    run_type="embedding",
    metadata={
        "ls_model": "openai/text-embedding-3-small",
        "ls_provider": "openrouter",
        "ls_model_type": "embedding"
        }
)
def get_embedding(text: str, model="openai/text-embedding-3-small"):
    response = openrouter_client.embeddings.create(
        input=text,
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
    name="retrieve_products",
    run_type="retriever",
    metadata={
        "ls_provider": "qdrant",

    }
)
def retrieve_products(query: str, limit: int = 3):
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



@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_context, question):
    return build_prompt_jinja(preprocessed_context, question)



@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={
        "ls_model": "meta-llama/llama-3.1-8b-instant",
        "ls_provider": "groq",
        "ls_model_type": "llm"
        }
)


def generate_answer(prompt):
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_model=RAGGenerationResponse
    )

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata['usage_metadata'] = {
            'prompt_tokens': raw_response.usage.prompt_tokens,
            'completion_tokens': raw_response.usage.completion_tokens,
            'total_tokens': raw_response.usage.total_tokens
        }

    return response




@traceable(
    name="rag_pipeline",
    run_type="llm",
)
def rag_pipeline(question: str, top_k: int = 5) -> str:
    try:
        retrieved_context = retrieve_products(question, limit=top_k)
        preprocessed_context = process_context(retrieved_context)
        prompt = build_prompt(preprocessed_context, question)
        answer = generate_answer(prompt)

        final_result={
            "original_answer":answer,
            "answer":answer.answer,
            "references":answer.references,
            "question":question,
            "retrieved_context":retrieved_context["retrieved_context"],
            "average_rating":retrieved_context["retrieved_context_ratings"],
            "similarity_score": retrieved_context["similarity_score"],
            "context_id":retrieved_context["retrieved_context_ids"]
        }


        return final_result

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        return {
            "answer": "Sorry, there was an error processing your request.",
            "question": question,
            "retrieved_context": [],
            "average_rating": [],
            "similarity_score": [],
            "context_id": []
        }

def rag_pipeline_wrapper(question, top_k=5):
    
 

    result = rag_pipeline(question, top_k=top_k)
    
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
        "answer": result["answer"],
        "used_context": used_context,
    }

