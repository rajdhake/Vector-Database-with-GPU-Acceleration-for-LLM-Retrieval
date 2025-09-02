import asyncio
import hashlib
import os
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="VectorDB Coordinator")

SHARDS = [s.strip() for s in os.getenv("SHARDS", "http://localhost:7001,http://localhost:7002").split(",") if s.strip()]

REQS = Counter("coordinator_requests_total", "Total requests", ["route"])
LAT = Histogram(
    "coordinator_latency_seconds",
    "Latency",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)
)

class InsertReq(BaseModel):
    id: str
    embedding: List[float]
    meta: Dict[str, Any] | None = None

class SearchReq(BaseModel):
    embedding: List[float]
    k: int = Field(ge=1, le=1000)
    metric: str = Field(default="cosine", description="cosine or l2")

@app.get("/health")
async def health():
    return {"ok": True, "shards": SHARDS}

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


def pick_shard(id_: str) -> str:
    h = int(hashlib.md5(id_.encode()).hexdigest(), 16)
    return SHARDS[h % len(SHARDS)]

@app.post("/vectors")
async def insert(req: InsertReq):
    REQS.labels("insert").inc()
    shard_url = pick_shard(req.id)
    async with httpx.AsyncClient(timeout=30) as client:
        with LAT.time():
            r = await client.post(f"{shard_url}/vectors", json=req.model_dump())
            if r.status_code != 200:
                raise HTTPException(r.status_code, r.text)
    return {"ok": True, "shard": shard_url}

@app.post("/search")
async def search(req: SearchReq):
    REQS.labels("search").inc()
    async with httpx.AsyncClient(timeout=60) as client:
        async def one(url: str):
            return await client.post(f"{url}/search", json=req.model_dump())
        with LAT.time():
            rs = await asyncio.gather(*[one(u) for u in SHARDS], return_exceptions=True)
    partials = []
    for r in rs:
        if isinstance(r, Exception):
            continue
        if r.status_code == 200:
            partials.extend(r.json())
    if not partials:
        raise HTTPException(502, "No shard replies")

    # Merge top-k by score (higher is better). For L2, shards already convert to similarity (negated distance)
    k = req.k
    top = []
    import heapq
    for item in partials:
        score = item["score"]
        if len(top) < k:
            heapq.heappush(top, (score, item))
        elif score > top[0][0]:
            heapq.heapreplace(top, (score, item))
    top_sorted = [it for _, it in sorted(top, key=lambda x: x[0], reverse=True)]
    return top_sorted