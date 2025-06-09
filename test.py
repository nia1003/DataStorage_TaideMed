from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from modules import (
    chunk,
    json_chunks_with_metadata,
    get_or_create_collection,
    embed_text,
    upsert_text_embeddings,
    search_similar_texts,
)

import os
from pathlib import Path


COLLECTION_NAME = "data_storage_demo_collection_test_2"
JSON_FILE = "data.json"
CHUNK_METHOD = "sentence"
CHUNK_SIZE = 1

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_KEY")

print("Loading & chunking data …")
items = json_chunks_with_metadata(
    json_path=JSON_FILE,
    method=CHUNK_METHOD,
    size=CHUNK_SIZE,
)
texts = [item["text"] for item in items]
print(f"Total chunks to embed: {len(texts)}")

client = get_or_create_collection(
    api_key=QDRANT_KEY,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    dimension=1024,
    metric="cosine",
    indexed_fields={"feedback_rate": "float"},
)

print("Generating embeddings via Mistral API …")
vectors = embed_text(texts, api_key=MISTRAL_KEY)
print(f"Got {len(vectors)} embeddings.")

print("Upserting to Qdrant …")
resp = upsert_text_embeddings(
    client=client,
    collection_name=COLLECTION_NAME,
    items=items,
    vectors=vectors,
)
print(f"Upsert complete: {resp}")

print("\nRunning sample query …")
query = "How can I control LLM personality?"
matches = search_similar_texts(
    client=client,
    collection_name=COLLECTION_NAME,
    query=query,
    feedback_rate=4,
    embed_fn=embed_text,
    api_key=MISTRAL_KEY,
)

for m in matches:
    payload = m.payload
    print(
        f"- \"{payload.get('text')}\"\n"
        f"    score: {m.score:.3f}  "
        f"feedback_rate: {payload.get('feedback_rate')}  "
        f"source: {payload.get('source')}\n"
    )
