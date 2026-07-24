# pipeline.py
# RAG pipeline for legal memo generation
# Supports per-collection querying, multiple models, quantization levels, and prompt variants

import argparse
import os
import time
from typing import List

import mlflow
import torch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from pydantic import Field
from qdrant_client import QdrantClient
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# model registry
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "saul": "Equall/Saul-Instruct-v1",
}

# federal collections
FEDERAL_COLLECTIONS = [
    "caselaw_f_supp",
    "caselaw_f_supp_2d",
    "caselaw_f_supp_3d",
    "caselaw_f2d",
    "caselaw_f3d",
    "caselaw_f_appx",
    "caselaw_us",
    "caselaw_us_app_dc",
    "caselaw_mj",
    "caselaw_frd",
    "caselaw_cust_ct",
    "caselaw_vet_app",
    "caselaw_tc",
]


# quantization configs
def get_quant_config(precision):
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None


def load_model(model_key, precision):
    model_id = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quant_config = get_quant_config(precision)

    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=pipe)


class FederalRetriever(BaseRetriever):
    """Queries all federal collections and merges results by relevance."""

    embeddings: HuggingFaceEmbeddings = Field(...)
    client: QdrantClient = Field(...)
    k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = []
        for collection in FEDERAL_COLLECTIONS:
            try:
                vs = QdrantVectorStore(
                    client=self.client,
                    collection_name=collection,
                    embedding=self.embeddings,
                )
                docs = vs.similarity_search_with_score(query, k=self.k)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Skipping {collection}: {e}")

        # sort all results by score descending, return top k
        all_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in all_docs[: self.k]]


def load_retriever(collection, k=5):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cuda"},
    )
    client = QdrantClient(url="http://localhost:6333", timeout=60)

    if collection == "federal":
        return FederalRetriever(embeddings=embeddings, client=client, k=k)
    else:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
        )
        return vector_store.as_retriever(search_kwargs={"k": k})


def load_prompt(variant):
    with open(f"prompts/memo_{variant}.txt", "r") as f:
        template = f.read()
    return PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def run_pipeline(model_key, precision, collection, prompt_variant, query):
    print(f"Loading model {model_key} at {precision}...")
    llm = load_model(model_key, precision)
    retriever = load_retriever(collection)
    prompt = load_prompt(prompt_variant)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": model_key,
                "precision": precision,
                "collection": collection,
                "prompt_variant": prompt_variant,
                "query": query,
            }
        )

        start = time.time()
        result = chain.invoke(query)
        latency = time.time() - start

        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated() / 1024**3
            mlflow.log_metric("peak_vram_gb", vram_used)
            torch.cuda.reset_peak_memory_stats()

        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_text(result, "output.txt")

        print(f"\n--- Output ---\n{result}")
        print(f"\nLatency: {latency:.2f}s")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--precision", choices=["fp16", "int8", "int4"], required=True)
    parser.add_argument(
        "--collection",
        required=True,
        help="collection name or 'federal' for all federal collections",
    )
    parser.add_argument(
        "--prompt", choices=["unconstrained", "constrained"], required=True
    )
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    run_pipeline(args.model, args.precision, args.collection, args.prompt, args.query)
