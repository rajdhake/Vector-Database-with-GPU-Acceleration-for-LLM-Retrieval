# Parallel Vector Database with GPU Acceleration for LLM Retrieval (CUDA + Multithreading + LLM-ready)

A mini distributed vector database: coordinator + N shards. Each shard keeps an in-memory matrix of embeddings and supports **GPU-accelerated cosine/L2 search** via CUDA (with CPU fallback). The coordinator handles **consistent-hash inserts** and **parallel fanout** for top-k search, and exposes a simple HTTP API. 

## Features
- CUDA C++17 flat index (cosine/L2) wrapped with pybind11
- Multithreaded shard server (FastAPI + uvicorn workers)
- Coordinator with async fanout + top-k merge
- **Coordinator + shards** to store and search embeddings.
- **SentenceTransformers** for embeddings (`all-MiniLM-L6-v2`, fast on CPU).
- **FLAN-T5-small** as the lightweight LLM to generate answers.
- Prometheus metrics: QPS & p50/p95/p99 (basic)
- CPU-only mode when CUDA not available (`USE_CUDA=0`)
- Supports **end-to-end RAG**: ingest documents → query with LLM answers.
- Optional **GPU shards (CUDA)**  NVIDIA T4 via google colab for accelerated search.



# Parallel Vector DB – Tiny RAG Demo


It works in two environments:

- **CPU demo** – Run locally on your **MacBook Pro** (no GPU required).
- **GPU demo** – Run on **Google Colab** with CUDA acceleration.

---


# A) CPU Demo on Mac (local, no GPU)

## 1) Start Vector DB (2 shards + coordinator)

Open **3 terminals** inside your `vectordb/` repo.

**Terminal A – Shard 1**
```bash
cd vectordb/shard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
USE_CUDA=0 uvicorn shard.server:app --host 0.0.0.0 --port 7001
```

**Terminal B – Shard 2**
```bash
cd vectordb/shard
source .venv/bin/activate
USE_CUDA=0 uvicorn shard.server:app --host 0.0.0.0 --port 7002
```

**Terminal C –Coordinator**
```bash
cd vectordb/coordinator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SHARDS="http://localhost:7001,http://localhost:7002"
uvicorn coordinator.main:app --host 0.0.0.0 --port 8000
```

***Run Demo**
```bash
python rag_demo/ingest.py --coordinator http://localhost:8000
python rag_demo/ask.py --coordinator http://localhost:8000 --question "How does sharding help a vector DB scale?"
python rag_demo/ask.py --coordinator http://localhost:8000 --question "Why are async CUDA streams useful?"
python rag_demo/ask.py --coordinator http://localhost:8000 --question "What indexes exist for vector search?"
```

# B) GPU Demo on Google Colab (CUDA)

## 1) Setup

Choose a GPU runtime (Runtime > Change runtime type > GPU).

```bash
!nvidia-smi
!python -V

!apt-get -y install build-essential cmake
!pip -q install fastapi uvicorn[standard] httpx prometheus-client pydantic==2.7.1 numpy pybind11
!pip -q install sentence-transformers transformers torch --index-url https://download.pytorch.org/whl/cpu
```

Build CUDA Module

```bash
%cd /content/vectordb/shard
!cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
!cmake --build build -j
!ls build
```

```bash
%cd /content/vectordb
!python rag_demo/ingest.py --coordinator http://localhost:8000
!python rag_demo/ask.py --coordinator http://localhost:8000 --question "How does sharding help a vector DB scale?"
!python rag_demo/ask.py --coordinator http://localhost:8000 --question "Why are async CUDA streams useful?"
```