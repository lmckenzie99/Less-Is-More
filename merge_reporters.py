from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import uuid

client = QdrantClient(url="http://localhost:6333")

NEV_SPLITS = [
    "caselaw_nev_109_2",
    "caselaw_nev_110_2",
    "caselaw_nev_111_2",
    "caselaw_nev_112_2",
    "caselaw_nev_113_2",
    "caselaw_nev_114_2",
    "caselaw_nev_116_2",
    "caselaw_nev_118_2",
    "caselaw_nev_120_2",
    "caselaw_nev_121_2",
    "caselaw_nev_122_2",
    "caselaw_nev_124_2",
]

TARGET = "caselaw_nev"

for collection in NEV_SPLITS:
    print(f"Merging {collection} into {TARGET}...")
    offset = None
    total = 0

    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )

        if not results:
            break

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=p.vector,
                payload=p.payload,
            )
            for p in results
        ]

        client.upsert(collection_name=TARGET, points=points)
        total += len(points)

        if next_offset is None:
            break
        offset = next_offset

    client.delete_collection(collection)
    print(f"Done. Moved {total} points, deleted {collection}.")

print("All Nevada splits merged.")
