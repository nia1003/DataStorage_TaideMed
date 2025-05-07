import re
import json
from datetime import datetime
import os
from typing import List, Dict

from mistralai import Mistral

from qdrant_client import QdrantClient, models


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


def json_chunk(json_path, method="char", size=20, text_key="text"):
    """
    從 JSON 或 JSONL 檔案載入資料，針對每行的 text_key 欄位進行 chunk。
    傳回 list of list of chunks。
    """
    results = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # read entire file as JSON array
        for obj in data:
            text = obj.get(text_key, "")
            chunks = chunk(text, method=method, size=size)
            results.append(chunks)
    return results


def get_or_create_collection(
    api_key: str,
    url: str,
    collection_name: str,
    dimension: int,
    metric: str = "cosine",  #
) -> QdrantClient:

    client = QdrantClient(url=url, api_key=api_key)

    collections = [col.name for col in client.get_collections().collections]
    if collection_name not in collections:
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

    return client


def embed_text(inputs: List[str], api_key: str) -> List[List[float]]:

    print("mistral api key", api_key)
    model = "mistral-embed"
    client = Mistral(api_key=api_key)

    resp = client.embeddings.create(model=model, inputs=inputs)
    return [item.embedding for item in resp.data]


def upsert_text_embeddings(
    client: QdrantClient,
    collection_name: str,
    texts: List[str],
    vectors: List[List[float]],
    source: str = "example_script",
) -> Dict:

    assert len(texts) == len(vectors), "Text and vector count must match."

    timestamp = datetime.utcnow().isoformat()
    points = []

    for i, (txt, vec) in enumerate(zip(texts, vectors), start=1):
        points.append(
            models.PointStruct(
                id=i,
                vector=vec,
                payload={
                    "source": source,
                    "text": txt,
                    "created": timestamp,
                },
            )
        )

    client.upsert(collection_name=collection_name, wait=True, points=points)
    return {"status": "ok", "upserted": len(points)}


def search_similar_texts(
    client: QdrantClient,
    collection_name: str,
    query: str,
    embed_fn,
    top_k: int = 5,
    verbose: bool = True,
    api_key: str = None
) -> List[Dict]:

    query_vec = embed_fn([query],api_key=api_key)[0]

    result = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )

    if verbose:
        print(f"\nSearch results for query: '{query}'")
        for m in result:
            score = m.score
            text = m.payload.get("text", "")[:80]
            print(f"  id={m.id:<8}  score={score:.3f}\n    text_snippet: {text}")

    return result
