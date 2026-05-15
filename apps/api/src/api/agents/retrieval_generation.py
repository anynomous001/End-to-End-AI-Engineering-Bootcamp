import os
from qdrant_client import QdrantClient
from openai import OpenAI
from groq import Groq
from api.core.config import config
import logging
from langsmith import traceable, get_current_run_tree

logger = logging.getLogger(__name__)

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
        collection_name="products",
        query=query_vector,
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
    prompt = f"""
    You are a shopping assistant that can answer questions about the products in stock.

    You will be given a question and a list of context.

    Instructions:
    - You need to answer the question based on the provided context only.
    - Never use word context and refer to it as the available products.

    Context:
    {preprocessed_context}

    Question:
    {question}
    """
    return prompt

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
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )


    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }

    return response.choices[0].message.content




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
            "answer":answer,
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
