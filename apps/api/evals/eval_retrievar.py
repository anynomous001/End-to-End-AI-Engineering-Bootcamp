from ragas.metrics import IDBasedContextRecall
from ragas.metrics import IDBasedContextPrecision
from ragas import SingleTurnSample
import os
import openai

from langsmith import Client
from qdrant_client import QdrantClient

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from api.agents.retrieval_generation import rag_pipeline



langsmith_client = Client()


qdrant_client = QdrantClient(host="localhost", port=6333)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
      api_key=os.environ.get("OPENROUTER_API_KEY"),
)


ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


async def ragas_faithfulness(run, example):
    
    sample = SingleTurnSample(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"]
    )
    scorer = Faithfulness(llm=ragas_llm)
    
    return await scorer.single_turn_ascore(sample)

async def ragas_responce_relevancy(run, example):
    
    sample = SingleTurnSample(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"]
    )
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    
    return await scorer.single_turn_ascore(sample)


async def ragas_context_precision_id_based(run, example):
    
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["context_id"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )
    scorer = IDBasedContextPrecision()
    
    return await scorer.single_turn_ascore(sample)

async def ragas_context_recall_id_based(run, example):
    
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["context_id"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )
    scorer = IDBasedContextRecall()
    
    return await scorer.single_turn_ascore(sample)

from langsmith.evaluation import aevaluate
import asyncio

async def main():
    async def async_rag_pipeline(x):
        return rag_pipeline(x["question"], top_k=5)

    results = await aevaluate(
        async_rag_pipeline,
        data="rag-evaluation-dataset",
        evaluators=[
            ragas_faithfulness,
            ragas_responce_relevancy,
            ragas_context_precision_id_based,
            ragas_context_recall_id_based
        ],
        experiment_prefix="retriever",
        max_concurrency=10
    )

if __name__ == "__main__":
    asyncio.run(main())