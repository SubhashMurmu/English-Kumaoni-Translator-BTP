"""Build the translation memory from the TRAIN split only (never dev/test - that would leak)."""
import json
import pickle

import faiss
from sentence_transformers import SentenceTransformer

rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8") if l.strip()]
src, tgt = [r["src"] for r in rows], [r["tgt"] for r in rows]

emb = SentenceTransformer("all-MiniLM-L6-v2").encode(src, convert_to_numpy=True, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, "faiss_cache.index")
with open("translation_memory.pkl", "wb") as f:
    pickle.dump({"src": src, "tgt": tgt}, f)
print(f"Indexed {index.ntotal} sentences (train split only).")
