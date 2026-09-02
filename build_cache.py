import faiss
import pickle
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

INDEX_OUT = "faiss_cache.index"
LOOKUP_OUT = "translation_memory.pkl"

dataset = load_dataset("sourabsb/english-kumaoni_translation_dataset", split="train")

english_sentences = dataset["english"]
kumaoni_sentences = dataset["kumaoni"]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(english_sentences, convert_to_numpy=True, normalize_embeddings=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_OUT)

lookup = {i: kumaoni_sentences[i] for i in range(len(kumaoni_sentences))}
with open(LOOKUP_OUT, "wb") as f:
    pickle.dump(lookup, f)

print(f"Indexed {index.ntotal} sentences.")
