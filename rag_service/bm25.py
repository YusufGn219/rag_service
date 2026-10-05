"""Keyword search: a Turkish-aware tokenizer and a small BM25 index."""
import re
import unicodedata
from collections import Counter

import numpy as np

_APOSTROPHE_SUFFIX = re.compile(r"['’‘`´]\w*")
_RUN = re.compile(r"\w+")
_CAMEL = re.compile(
    r"(?<=[a-zçğıöşü0-9])(?=[A-ZÇĞİÖŞÜ])"  # fooBar -> foo Bar
    r"|(?<=[A-ZÇĞİÖŞÜ])(?=[A-ZÇĞİÖŞÜ][a-zçğıöşü])"  # HTTPServer -> HTTP Server
)


def _fold(s: str) -> str:
    """Lowercase the Turkish way and drop diacritics, so İ/I/ı/i and ç/c, ş/s... match."""
    s = s.replace("İ", "i").replace("I", "i").replace("ı", "i").lower()
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    """Split text into search tokens. No stemming: a word only matches itself.

    Identifiers (CamelCase, snake_case) yield the whole identifier plus its parts.
    """
    tokens: list[str] = []
    for run in _RUN.findall(_APOSTROPHE_SUFFIX.sub("", text)):
        parts: list[str] = []
        for seg in run.split("_"):
            if seg:
                parts.extend(_CAMEL.sub(" ", seg).split())
        if not parts:
            continue
        whole = _fold(run)
        folded = [_fold(p) for p in parts]
        candidates = [whole] + folded if len(parts) > 1 else [whole]
        tokens.extend(t for t in candidates if len(t) > 1)
    return tokens


class BM25Index:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) if self.n and self.lengths.sum() else 1.0

        ids: dict[str, list[int]] = {}
        tfs: dict[str, list[int]] = {}
        for doc_id, tokens in enumerate(docs):
            for term, tf in Counter(tokens).items():
                ids.setdefault(term, []).append(doc_id)
                tfs.setdefault(term, []).append(tf)
        self._postings = {
            t: (np.array(ids[t], dtype=np.int64), np.array(tfs[t], dtype=np.float64)) for t in ids
        }
        self._idf = {
            t: float(np.log(1 + (self.n - len(p[0]) + 0.5) / (len(p[0]) + 0.5)))
            for t, p in self._postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        for term in sorted(set(query_tokens)):  # a repeated query word counts once
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_ids, tf = posting
            norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_ids] / self.avgdl)
            out[doc_ids] += self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query_tokens: list[str], k: int = 10) -> list[tuple[int, float]]:
        """Top-k (doc_id, score) with score > 0, best first; ties go to the lower doc id."""
        s = self.scores(query_tokens)
        hits = np.flatnonzero(s > 0)
        if hits.size == 0:
            return []
        order = hits[np.lexsort((hits, -s[hits]))][:k]
        return [(int(i), float(s[i])) for i in order]


def index_chunks(chunks, **kwargs) -> BM25Index:
    """Index chunks by embed_text, so tags and heading paths are searchable too."""
    return BM25Index([tokenize(c.embed_text) for c in chunks], **kwargs)
