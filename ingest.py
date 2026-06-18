# Ingestion script to draw down CAP dataset
# Using Qdrant for vector database
# Langchain for RAG
# Using BAAI/bge-base-en-v1.5 as embedding model

import json

# Imports
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantClient, QdrantVectorStore
from qdrant_client import Distance, VectorParams

# Global vars
TARGET_N = 1_000
SEED = 42
BATCH_SIZE = 256

# embedding model draw down
# use gpu for efficient embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"},
)

# qdrant client setup
client = QdrantClient(url="http://localhost:6333")

# text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

# stream
ds = load_dataset(
    "common-pile/caselaw_access_project_filtered", split="train", streaming=True
)
ds = ds.shuffle(seed=SEED, buffer_size=10_000)

# dynamic per-reporter state
vector_stores = {}
batches = {}
case_ids = []
collected = 0


def get_reporter(example_id):
    prefix = example_id.split("/")[0]
    parts = prefix.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else prefix


# collections cannot have hypens
def sanitize_collection_name(reporter):
    return "caselaw_" + reporter.replace("-", "_")


def ensure_collection(reporter):
    if reporter not in vector_stores:
        collection_name = sanitize_collection_name(reporter)
        if not client.collection_exits(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=768, distance=Distance.COSINE),
            )
        vector_stores[reporter] = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        batches[reporter] = {"texts": [], "metadatas": []}


for example in ds:
    reporter = get_reporter(example["id"])
    ensure_collection(reporter)

    chunks = splitter.split_text(example["text"])
    for chunk in chunks:
        batches[reporter]["texts"].append(chunk)
        batches[reporter]["metadatas"].append(
            {
                "case_id": example["id"],
                "reporter": reporter,
                "author": example["metadata"].get("author", ""),
            }
        )

    case_ids.append(example["id"])
    collected += 1

    # flush batches
    for r, batch in batches.items():
        if len(batch["texts"]) >= BATCH_SIZE:
            vector_stores[r].add_texts(
                texts=batch["texts"],
                metadatas=batch["metadatas"],
            )
            batch["texts"], batch["metadatas"] = [], []

    if collected % 100 == 0:
        print(f"Cases collected: {
            collected}/{TARGET_N} | Collections: {len(vector_stores)}")

    if collected >= TARGET_N:
        break

# flush remainders
for r, batch in batches.items():
    vector_stores[r].add_texts(
        texts=batch["texts"],
        metadatas=batch["metadatas"],
    )

# save reproducibility manifests
with open("corpus_case_ids.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "target_n": TARGET_N,
            "reporters": list(vector_stores.keys()),
            "case_ids": case_ids,
        },
        f,
    )

print(f"Done. {collected} cases ingested across {
      len(vector_stores)} collections.")
