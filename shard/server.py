import os
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

app = FastAPI(title="VectorDB Shard")

REQS = Counter("shard_requests_total", "Total requests", ["route"])
LAT = Histogram(
    "shard_latency_seconds",
    "Latency",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)
)

# --- Index init ---
index = None
ids: List[str] = []  # maps row index -> external id

try:
    if USE_CUDA:
        from vdb import FlatIndex  # pybind module (GPU path)
        dim_default = int(os.getenv("DIM", "384"))
        index = FlatIndex(dim_default)
    else:
        from .fallback_numpy_index import NumpyFlatIndex as FlatIndex
        dim_default = int(os.getenv("DIM", "384"))
        index = FlatIndex(dim_default)
except Exception as e:
    # fallback to numpy if CUDA build missing
    from .fallback_numpy_index import NumpyFlatIndex as FlatIndex
    dim_default = int(os.getenv("DIM", "384"))
    index = FlatIndex(dim_default)

class InsertReq(BaseModel):
    id: str
    embedding: List[float]
    meta: Dict[str, Any] | None = None

class SearchReq(BaseModel):
    embedding: List[float]
    k: int = Field(ge=1, le=1000)
    metric: str = Field(default="cosine")

@app.get("/health")
async def health():
    return {"ok": True, "count": index.count(), "dim": dim_default, "use_cuda": USE_CUDA}

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/vectors")
async def insert(req: InsertReq):
    REQS.labels("insert").inc()
    x = np.asarray(req.embedding, dtype=np.float32)
    if x.ndim != 1 or x.shape[0] != index.dim():
        raise HTTPException(400, f"dim mismatch: got {x.shape}, expected ({index.dim()},)")
    with LAT.time():
        start = index.add_batch(x.reshape(1, -1))
    ids.extend([req.id])
    return {"ok": True, "start": start}

@app.post("/search")
async def search(req: SearchReq):
    REQS.labels("search").inc()
    if index.count() == 0:
        return []
    q = np.asarray(req.embedding, dtype=np.float32)
    if q.ndim != 1 or q.shape[0] != index.dim():
        raise HTTPException(400, f"dim mismatch: got {q.shape}, expected ({index.dim()},)")
    with LAT.time():
        res = index.search(q, req.k, req.metric)
    out = []
    for item in res:
        idx = int(item["index"]) if isinstance(item, dict) else int(item["index"])  # compat
        score = float(item["score"]) if isinstance(item, dict) else float(item["score"])
        out.append({"id": ids[idx], "score": score})
    return out