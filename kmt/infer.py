"""Inference: n-best generation + Hindi-leakage reranking, and a guarded translation memory."""
import pickle
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .common import PRESETS
from .text import leak_rate, normalize, tm_compatible

_LANG_CODES = re.compile(r"\b(kfy_IN|kfy_Deva|hi_IN|hin_Deva|en_XX|eng_Latn)\b")


def load_hindi_only(path):
    if path and Path(path).exists():
        return set(Path(path).read_text(encoding="utf-8").split())
    return set()


class Translator:
    def __init__(self, model_dir, preset="mbart", hindi_only=None, device=None):
        self.p = PRESETS[preset]
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.src_lang = self.p["src"]
        self.kmn_id = self.tok.convert_tokens_to_ids(self.p["kmn"])
        if self.kmn_id in (None, self.tok.unk_token_id):
            # Fail loudly: the old app silently fell back to the untrained base model.
            raise RuntimeError(f"{model_dir} has no {self.p['kmn']} token - wrong model directory?")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.hindi_only = hindi_only or set()

    @torch.inference_mode()
    def translate(self, texts, num_beams=5, n_best=5, alpha=0.0, bs=16):
        """alpha > 0 reranks the n-best list by  mean_logprob - alpha * hindi_leak_rate."""
        n_best = max(1, min(n_best, num_beams))
        out = []
        for i in range(0, len(texts), bs):
            batch = [normalize(t) for t in texts[i:i + bs]]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=128).to(self.device)
            g = self.model.generate(**enc, forced_bos_token_id=self.kmn_id, num_beams=num_beams,
                                    num_return_sequences=n_best, max_new_tokens=128,
                                    return_dict_in_generate=True, output_scores=True)
            hyps = [_LANG_CODES.sub("", h).strip()
                    for h in self.tok.batch_decode(g.sequences, skip_special_tokens=True)]
            scores = g.sequences_scores.tolist()
            for j in range(len(batch)):
                cand = list(zip(hyps[j * n_best:(j + 1) * n_best], scores[j * n_best:(j + 1) * n_best]))
                best = max(cand, key=lambda c: c[1] - alpha * leak_rate(c[0], self.hindi_only))
                out.append(best[0])
        return out


class TranslationMemory:
    """Exact + fuzzy lookup. Fuzzy hits must also pass the number/name/negation guard."""

    def __init__(self, index_path, pkl_path, threshold=0.95, embedder="all-MiniLM-L6-v2"):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.index = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        self.src, self.tgt = d["src"], d["tgt"]
        self.exact = {normalize(s).lower(): t for s, t in zip(self.src, self.tgt)}
        self.embedder, self.threshold = SentenceTransformer(embedder), threshold

    def lookup(self, query):
        q = normalize(query)
        if q.lower() in self.exact:
            return self.exact[q.lower()], 1.0
        v = self.embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(v, 1)
        sim, i = float(D[0][0]), int(I[0][0])
        if i >= 0 and sim >= self.threshold and tm_compatible(q, self.src[i]):
            return self.tgt[i], sim
        return None
