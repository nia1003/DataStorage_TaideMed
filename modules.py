import re
import json
from datetime import datetime
import os
from typing import List, Dict

from mistralai import Mistral

from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, Range


from typing import List, Dict, Any

from typing import List, Dict


# chunk
def chunk(text: str, method="char", size=20):
    """
    切分文字為 chunk。
    method: 'char', 'sentence', 'word'
    size: chunk 長度（字數或詞數）
    """
    if method == "char":
        return [text[i : i + size] for i in range(0, len(text), size)]

    elif method == "sentence":
        # 分句（支援中英文標點）
        sentences = re.split(r"(?<=[。！？!?])", text)
        return [s.strip() for s in sentences if s.strip()]

    elif method == "word":
        words = text.split()
        return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]

    else:
        raise ValueError("Unknown method. Use 'char', 'sentence', or 'word'.")


def json_chunks_with_metadata(
    json_path: str,
    method: str = "char",
    size: int = 20,
    text_key: str = "text",
    feedback_key: str = "feedback_rate",
    source_key: str = "source",
) -> List[Dict[str, Any]]:

    items: List[Dict[str, Any]] = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for obj in data:
            txt = obj.get(text_key, "")
            rate = obj.get(feedback_key)
            src = obj.get(source_key)
            for c in chunk(txt, method=method, size=size):
                items.append({"text": c, "feedback_rate": rate, "source": src})
    return items


from qdrant_client import QdrantClient, models


def get_or_create_collection(
    api_key: str,
    url: str,
    collection_name: str,
    dimension: int,
    metric: str = "cosine",
    indexed_fields: Dict[str, str] = None,  # e.g. {"feedback_rate": "float"}
) -> QdrantClient:

    client = QdrantClient(url=url, api_key=api_key)

    # 1) Create collection if missing
    existing = [col.name for col in client.get_collections().collections]
    if collection_name not in existing:
        print(f"🔨  Creating collection '{collection_name}' …")
        vectors_config = models.VectorParams(
            size=dimension,
            distance={
                "cosine": models.Distance.COSINE,
                "euclid": models.Distance.EUCLID,
                "dot": models.Distance.DOT,
            }[metric],
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )
        print("Collection is being initialized")
    else:
        print(f"Collection '{collection_name}' already exists.")

    # 2) Ensure payload indexes for any numeric metadata fields
    if indexed_fields:
        for field_name, field_type in indexed_fields.items():
            print(f"Ensuring payload index on '{field_name}' ({field_type}) …")
            client.create_payload_index(
                collection_name=collection_name,
                field_name="feedback_rate",
                field_schema="float",
                wait=True,
            )
        print("All specified payload indexes are ensured.")

    return client


def embed_text(inputs: List[str], api_key: str) -> List[List[float]]:

    # print("mistral api key", api_key)
    model = "mistral-embed"
    client = Mistral(api_key=api_key)

    resp = client.embeddings.create(model=model, inputs=inputs)
    return [item.embedding for item in resp.data]


def upsert_text_embeddings(
    client: QdrantClient,
    collection_name: str,
    items: List[Dict[str, Any]],
    vectors: List[List[float]],
) -> Dict[str, Any]:

    assert len(items) == len(vectors), "must match"
    ts = datetime.utcnow().isoformat()
    points = []
    for i, (item, vec) in enumerate(zip(items, vectors), start=1):
        points.append(
            models.PointStruct(
                id=i,
                vector=vec,
                payload={
                    "text": item["text"],
                    "feedback_rate": float(item["feedback_rate"]),
                    "source": item["source"],
                    "created": ts,
                },
            )
        )
    client.upsert(collection_name=collection_name, wait=True, points=points)
    return {"status": "ok", "upserted": len(points)}


from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range


def search_similar_texts(
    client: QdrantClient,
    collection_name: str,
    query: str,
    embed_fn,
    top_k: int = 5,
    feedback_rate: int = 4,
    verbose: bool = True,
    api_key: str = None,
) -> List[Dict]:
    query_vec = embed_fn([query], api_key=api_key)[0]

    payload_filter = Filter(
        must=[FieldCondition(key="feedback_rate", range=Range(gt=feedback_rate))]
    )
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
        query_filter=payload_filter,
    )

    if verbose:
        print(
            f"\nSearch results for query: '{query}' (feedback_rate > {feedback_rate})"
        )
        if not results:
            print("  No results matching the feedback_rate filter.")
        for m in results:
            txt = m.payload.get("text", "").replace("\n", " ")[:80]
            rate = m.payload.get("feedback_rate")
            print(f"  id={m.id:<8} score={m.score:.3f} feedback_rate={rate}")
            print(f"    snippet: {txt}")

    return results
