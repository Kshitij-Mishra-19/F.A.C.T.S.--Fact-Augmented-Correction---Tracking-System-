from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="halluguard_knowledge",
    metadata={"hnsw:space": "cosine"}
)


def retrieve_evidence(claim, top_k=3):
    """
    Retrieve relevant evidence for one factual claim.
    """

    if not isinstance(claim, str) or not claim.strip():
        return []

    if collection.count() == 0:
        return []

    query_embedding = model.encode(
        claim,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    evidence = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        evidence.append({
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": round(float(distance), 4),
            "similarity": round(
                max(0.0, min(1.0, 1.0 - float(distance))),
                4
            )
        })

    return evidence


if __name__ == "__main__":

    test_claim = test_claim = "Saturn has a prominent ring system."

    results = retrieve_evidence(test_claim)

    print("\nRETRIEVED EVIDENCE\n")

    for item in results:

        print("Source:", item["source"])
        print("Evidence:", item["text"])
        print("Similarity:", item["similarity"])
        print("-" * 60)