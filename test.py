from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from modules import (
    chunk,
    json_chunk,
    get_or_create_collection,
    embed_text,
    upsert_text_embeddings,
    search_similar_texts,
)

import os
from pathlib import Path


COLLECTION_NAME = "data_storage_demo_collection"
JSON_FILE = "data.json"
CHUNK_METHOD = "sentence"
CHUNK_SIZE = 1

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_KEY")

print("Loading & chunking data …")
chunked_data = json_chunk(json_path=JSON_FILE, method=CHUNK_METHOD, size=CHUNK_SIZE)
flat_chunks = [c for line in chunked_data for c in line]
print(f"Total chunks to embed: {len(flat_chunks)}")

client = get_or_create_collection(
    api_key=QDRANT_KEY,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    dimension=1024,
    metric="cosine",
)


print("Generating embeddings via Mistral API …")
vectors = embed_text(flat_chunks,api_key=MISTRAL_KEY)
print(f"Got {len(vectors)} embeddings.")


print("Upserting to Qdrant …")
resp = upsert_text_embeddings(
    client=client,
    collection_name=COLLECTION_NAME,
    texts=flat_chunks,
    vectors=vectors,
    source="demo_pipeline",
)
print(f"Upsert complete. {resp}")


print("\nRunning sample query …")
query = "How can I control LLM personality?"
matches = search_similar_texts(
    client=client,
    collection_name=COLLECTION_NAME,
    query=query,
    embed_fn=embed_text,
    api_key=MISTRAL_KEY
)
