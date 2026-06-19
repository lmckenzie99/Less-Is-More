# Sanity check retrieval on 10k ingestion

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cuda"},
        )

client = QdrantClient(url="http://localhost:6333")

# pick a collection to test
vector_store = QdrantVectorStore(
        client=client,
        collection_name="caselaw_tc",
        embedding=embeddings,
        )

query = "tax evasion"
results = vector_store.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Case ID: {doc.metadata['case_id']}")
    print(f"Reporter: {doc.metadata['reporter']}")
    print(f"Text: {doc.page_content[:300]}")
