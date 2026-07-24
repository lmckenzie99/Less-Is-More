import re
import sys
import os
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

MEMO = sys.stdin.read()
COLLECTION = os.environ.get("CITE_COLLECTION", "caselaw_f2d")

# strip common preamble phrases before matching
cleaned = re.sub(
    r'(This is outlined in the case of|The case of|as defined in|outlined in the case of|'
    r'in the case of|as set forth in|as stated in|as noted in|as held in|as discussed in)\s+',
    '',
    MEMO
)

# matches: Name v. Name, vol reporter page [(year)]
# handles LLC., multi-word names, multiple page refs
citation_pattern = re.compile(
    r'([A-Z][A-Za-z0-9\s\'\-]+?(?:,\s*(?:LLC|Inc|Ltd|Corp)\.?)?)\s+v\.\s+'
    r'([A-Z][A-Za-z0-9\s\'\-,\.]+?),\s*'
    r'\d+\s+[A-Za-z][A-Za-z0-9\s\.]*\d+'
    r'(?:,\s*\d+)*'
    r'(?:\s*\(\d{4}\))?'
)

citations = []
seen = set()
for m in citation_pattern.finditer(cleaned):
    full = m.group(0).strip()
    key = re.sub(r'\s+', ' ', full[:50]).lower()
    if key not in seen:
        seen.add(key)
        citations.append(full)

print(f"Collection: {COLLECTION}")
print(f"Citations found: {len(citations)}")
for c in citations:
    print(f"  {c}")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"},
)

client = QdrantClient(url="http://localhost:6333", timeout=60)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION,
    embedding=embeddings,
)

print(f"\nSearching {COLLECTION} for each citation...\n")
hallucinated = []
verified = []
not_in_corpus = []

for citation in citations:
    plaintiff_match = re.match(r'([A-Z][A-Za-z]+)', citation)
    defendant_match = re.search(r'v\.\s+([A-Z][A-Za-z]+)', citation)
    plaintiff_name = plaintiff_match.group(1) if plaintiff_match else ""
    defendant_name = defendant_match.group(1) if defendant_match else ""

    results = vector_store.similarity_search(citation.strip(), k=10)

    print(f"--- {citation[:80]} ---")
    if results:
        genuine_match = any(
            defendant_name in r.page_content or plaintiff_name in r.page_content
            for r in results
        )
        if genuine_match:
            verified.append(citation)
            print(f"  VERIFIED: {results[0].metadata['case_id']}")
            print(f"  {results[0].page_content[:150]}")
        else:
            not_in_corpus.append(citation)
            print(f"  NOT IN CORPUS: neither '{plaintiff_name}' nor '{defendant_name}' found in top 10 results")
            print(f"  Top match: {results[0].metadata['case_id']}")
    else:
        not_in_corpus.append(citation)
        print(f"  NOT IN CORPUS: no results returned")
    print()

print(f"Summary: {len(verified)} verified, {len(not_in_corpus)} not in corpus, {len(hallucinated)} hallucinated")
if not_in_corpus:
    print("Not in corpus (verify externally):")
    for c in not_in_corpus:
        print(f"  {c[:80]}")
if hallucinated:
    print("Hallucinated:")
    for h in hallucinated:
        print(f"  {h[:80]}")
