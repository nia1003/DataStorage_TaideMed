# query
from typing import List, Dict
from pinecone import Index


import os
from pathlib import Path


INDEX_NAME = "demo-index"
JSON_FILE = "data.json"   # Make sure this exists with {"text": "..."} per line
CHUNK_METHOD = "sentence"
CHUNK_SIZE = 1  # sentence-level, size ignored here but retained for consistency

# ----------------------------
# 1. Load and chunk data
# ----------------------------
print(" Loading and chunking data...")
chunked_data = json_chunk(json_path=JSON_FILE, method=CHUNK_METHOD, size=CHUNK_SIZE)
# Flatten the chunked data into a single list of text chunks
flat_chunks = [chunk for chunks in chunked_data for chunk in chunks]

print(f"Total chunks to embed: {len(flat_chunks)}")


pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
index = get_or_create_index(pc, index_name=INDEX_NAME)

# ----------------------------
# 3. Embed all chunks via Mistral API
# ----------------------------
print("Generating embeddings via Mistral API...")
vectors = embed_text(flat_chunks)
print(f"Got {len(vectors)} embeddings.")

# ----------------------------
# 4. Upsert to Pinecone
# ----------------------------
print("Upserting to Pinecone...")
resp = upsert_text_embeddings(
    index=index,
    texts=flat_chunks,
    vectors=vectors,
    source="demo_pipeline"
)
print(f"Upsert complete. Response: {resp}")

# ----------------------------
# 5. Run a sample search
# ----------------------------
print("\n Running sample query...")
query = "How can I control LLM personality?"
matches = search_similar_texts(
    index=index,
    query=query,
    embed_fn=embed_text,
)
