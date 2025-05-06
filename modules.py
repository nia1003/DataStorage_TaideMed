import re
import json

#chunk
def chunk(text: str, method="char", size=20):
    """
    切分文字為 chunk。
    method: 'char', 'sentence', 'word'
    size: chunk 長度（字數或詞數）
    """
    if method == "char":
        return [text[i:i+size] for i in range(0, len(text), size)]

    elif method == "sentence":
        # 分句（支援中英文標點）
        sentences = re.split(r'(?<=[。！？!?])', text)
        return [s.strip() for s in sentences if s.strip()]

    elif method == "word":
        words = text.split()
        return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

    else:
        raise ValueError("Unknown method. Use 'char', 'sentence', or 'word'.")


def json_chunk(json_path, method="char", size=20, text_key="text"):
    """
    從 JSON 或 JSONL 檔案載入資料，針對每行的 text_key 欄位進行 chunk。
    傳回 list of list of chunks。
    """
    results = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get(text_key, "")
            chunks = chunk(text, method=method, size=size)
            results.append(chunks)
    return results



# create index
from datetime import datetime
import os
from typing import List

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec, PineconeIndexSpec, Index


os.environ["PINECONE_KEY"] = "pcsk_5bsuWC_U64EWzFTtNyy5TpdebnbLXEJfvwaPwZAEoAYnTSJrpw5DYzSHEhwpFa9nXsPkrg"


pc = Pinecone(api_key=os.environ["PINECONE_KEY"])


def get_or_create_index(pc: Pinecone,
                        index_name: str,
                        dimension: int = 1536,
                        metric: str = "cosine") -> Index:

    if index_name not in pc.list_indexes():
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric
        )
    else:
        print(f"Index '{index_name}' already exists.")

    return pc.Index(index_name)



#embedding
import os
from mistralai import Mistral
from typing import List

def embed_text(inputs: List[str]) -> List[List[float]]:

    api_key = os.environ["TOGETHER_API_KEY"]
    model = "mistral-embed"

    client = Mistral(api_key=api_key)

    embeddings_batch_response = client.embeddings.create(
        model=model,
        inputs=inputs,
    )

    return [item.embedding for item in embeddings_batch_response.data]



# upsert
from datetime import datetime
from typing import List
from pinecone import Index

def upsert_text_embeddings(index: Index,
                           texts: List[str],
                           vectors: List[List[float]],
                           namespace: str = "default",
                           source: str = "example_script") -> dict:

    assert len(texts) == len(vectors), "Text and vector count must match."

    timestamp = datetime.utcnow().isoformat()
    records = []

    for i, (txt, vec) in enumerate(zip(texts, vectors), start=1):
        records.append({
            "id": f"doc-{i}",
            "values": vec,
            "metadata": {
                "source": source,
                "text": txt,
                "created": timestamp
            }
        })

    # Upsert to Pinecone
    response = index.upsert(vectors=records, namespace=namespace)
    return response



# query
from typing import List, Dict
from pinecone import Index

def search_similar_texts(index: Index,
                         query: str,
                         embed_fn,
                         namespace: str = "default",
                         top_k: int = 5,
                         verbose: bool = True) -> List[Dict]:

    query_vector = embed_fn([query])[0]

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    matches = result.get("matches", [])

    if verbose:
        print(f"\n Search results for query: '{query}'")
        for match in matches:
            print(f"  id={match['id']:<6}  score={match['score']:.3f}")
            print(f"    text_snippet: {match['metadata'].get('text', '')}\n")

    return matches
