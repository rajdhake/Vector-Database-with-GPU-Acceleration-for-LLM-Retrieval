import argparse
import httpx
import numpy as np
import asyncio

parser = argparse.ArgumentParser()
parser.add_argument('--coordinator', type=str, default='http://localhost:8000')
parser.add_argument('--n', type=int, default=10000)
parser.add_argument('--dim', type=int, default=384)
args = parser.parse_args()

async def main():
    X = np.random.randn(args.n, args.dim).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = []
        for i in range(args.n):
            payload = {"id": f"vec-{i}", "embedding": X[i].tolist()}
            tasks.append(client.post(f"{args.coordinator}/vectors", json=payload))
            if len(tasks) >= 256:
                rs = await asyncio.gather(*tasks)
                tasks.clear()
        if tasks:
            await asyncio.gather(*tasks)
    print(f"Loaded {args.n} vectors of dim {args.dim}")

if __name__ == '__main__':
    asyncio.run(main())