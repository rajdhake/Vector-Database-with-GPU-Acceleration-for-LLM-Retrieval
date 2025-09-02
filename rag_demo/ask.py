
import json, httpx, argparse, textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

parser = argparse.ArgumentParser()
parser.add_argument("--coordinator", default="http://localhost:8000")
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--question", required=True)
args = parser.parse_args()

id2text = json.loads(Path("rag_demo/id2text.json").read_text())

emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
q_emb = emb_model.encode(args.question, normalize_embeddings=True).tolist()

with httpx.Client(timeout=60) as client:
    res = client.post(f"{args.coordinator}/search",
                      json={"embedding": q_emb, "k": args.k, "metric": "cosine"})
    res.raise_for_status()
    hits = res.json()

context = "\n\n".join([f"- {id2text[h['id']]}" for h in hits])

prompt = f"""You are a helpful assistant.
Use the context to answer the question concisely.

Context:
{context}

Question: {args.question}
Answer:"""

tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
lm  = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
out = lm.generate(**inputs, max_new_tokens=160)
print("\n" + tok.decode(out[0], skip_special_tokens=True))
