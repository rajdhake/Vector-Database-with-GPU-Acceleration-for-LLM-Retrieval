import argparse
import asyncio
import time
import numpy as np
import httpx

parser = argparse.ArgumentParser()
parser.add_argument('--coordinator', type=str, default='http://localhost:8000')
parser.add_argument('--q', type=int, default=200)
parser.add_argument('--dim', type=int, default=384)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--concurrency', type=int, default=32)
parser.add_argument('--metric', type=str, default='cosine')
args = parser.parse_args()

async def worker(client, queries, latencies):
    for q in queries:
        t0 = time.perf_counter()
        r = await client.post(f"{args.coordinator}/search", json={"embedding": q.tolist(), "k": args.k, "metric": args.metric})
        r.raise_for_status()
        latencies.append(time.perf_counter() - t0)

async def main():
    Q = np.random.randn(args.q, args.dim).astype(np.float32)
    norms = np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    if args.metric=='cosine':
        Q = Q / norms
    latencies = []
    async with httpx.AsyncClient(timeout=60) as client:
        chunks = np.array_split(Q, args.concurrency)
        await asyncio.gather(*[worker(client, ch, latencies) for ch in chunks])
    lat = np.array(latencies)
    print(f"QPS: {args.q/lat.sum():.2f}")
    for p in [50, 95, 99]:
        print(f"p{p}: {np.percentile(lat*1000, p):.2f} ms")

if __name__ == '__main__':
    asyncio.run(main())