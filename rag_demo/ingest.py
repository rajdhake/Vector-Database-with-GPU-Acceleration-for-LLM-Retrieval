
import json, uuid, httpx, argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--coordinator", default="http://localhost:8000")
parser.add_argument("--chunk_size", type=int, default=120)
args = parser.parse_args()

# Tiny sample corpus 
docs = {
  "vectordb_overview": """Vector databases store high-dimensional embeddings and provide fast nearest neighbor search.
They power RAG systems by retrieving semantically similar chunks. Index choices like Flat, IVF or HNSW trade recall for speed.""",
  "cuda_basics": """CUDA lets you write kernels that run thousands of threads on an NVIDIA GPU.
Pinned memory and async streams help overlap copies with compute and reduce tail latency.""",
  "sharding": """Sharding distributes data across nodes using a hash function. A coordinator fans out requests
and merges top-k results. Replication can improve availability and read throughput."""
}

# simple word-chunker
def chunk(text, n=120):
    toks = text.split()
    for i in range(0, len(toks), n):
        yield " ".join(toks[i:i+n])

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
id2text = {}

with httpx.Client(timeout=30) as client:
    for title, body in docs.items():
        for j, piece in enumerate(chunk(body, args.chunk_size)):
            emb = model.encode(piece, normalize_embeddings=True).tolist()
            vid = f"{title}::chunk{j}::{uuid.uuid4().hex[:8]}"
            id2text[vid] = piece
            r = client.post(f"{args.coordinator}/vectors",
                            json={"id": vid, "embedding": emb, "meta": {"title": title}})
            r.raise_for_status()

Path("rag_demo/id2text.json").write_text(json.dumps(id2text, indent=2))
print(f"Ingested {len(id2text)} chunks.")
