import numpy as np

class NumpyFlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.X = np.empty((0, dim), dtype=np.float32)

    def add_batch(self, X: np.ndarray) -> int:
        assert X.ndim == 2 and X.shape[1] == self.dim
        # normalize rows for cosine
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        start = self.X.shape[0]
        self.X = np.vstack([self.X, Xn.astype(np.float32)])
        return start

    def count(self) -> int:
        return self.X.shape[0]

    def search(self, q: np.ndarray, k: int, metric: str = "cosine"):
        q = q.astype(np.float32)
        if metric == "cosine":
            qn = q / (np.linalg.norm(q) + 1e-12)
            scores = self.X @ qn
        else:
            # negative L2 distance as similarity
            dif = self.X - q
            scores = -np.einsum('ij,ij->i', dif, dif)
        if self.X.shape[0] == 0:
            return []
        idx = np.argpartition(-scores, kth=min(k, len(scores)-1))[:k]
        top_sorted = idx[np.argsort(-scores[idx])]
        return [{"index": int(i), "score": float(scores[i])} for i in top_sorted]