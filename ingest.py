#Ingestion script to draw down CAP dataset
#Using Qdrant for vector database
#Langchain for RAG
#Using BAAI/bge-base-en-v1.5 as embedding model

#Imports
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import QdrantClient
from qdrant_client import Distance, VectorParams
import json

#Global vars 
TARGET_N = 1_000
SEED = 42
BATCH_SIZE = 256

#embedding model draw down
#use gpu for efficient embedding
embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cuda"},
        )

#qdrant client setup
client = QdrantClient(url="http://localhost:6333")

#text splitter
splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        )

#stream
ds = load_dataset("common-pile/caselaw_access_project_filtered", split="train", streaming=True)
ds = ds.shuffle(seed=SEED, buffer_size=10_000)

#dynamic per-reporter
