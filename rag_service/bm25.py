"""Keyword search: a Turkish-aware tokenizer and a small BM25 index."""
import json
import re
import unicodedata
from collections import Counter

import numpy as np

# Bump when tokenize() changes, so stored indexes built with the old rules get rebuilt.
TOKENIZER_VERSION = 1

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

        ids: dict[str, list[int]] = {}
        tfs: dict[str, list[int]] = {}
        for doc_id, tokens in enumerate(docs):
            for term, tf in Counter(tokens).items():
                ids.setdefault(term, []).append(doc_id)
                tfs.setdefault(term, []).append(tf)
        self._postings = {
            t: (np.array(ids[t], dtype=np.int64), np.array(tfs[t], dtype=np.float64)) for t in ids
        }
        self._finish()

    def _finish(self) -> None:
        """Derive avgdl and idf from lengths and postings (also used after loading)."""
        self.avgdl = float(self.lengths.mean()) if self.n and self.lengths.sum() else 1.0
        self._idf = {
            t: float(np.log(1 + (self.n - len(p[0]) + 0.5) / (len(p[0]) + 0.5)))
            for t, p in self._postings.items()
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Flat numpy form for saving (postings concatenated, with per-term offsets)."""
        terms = sorted(self._postings)
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        for i, t in enumerate(terms):
            offsets[i + 1] = offsets[i] + len(self._postings[t][0])
        if terms:
            doc_ids = np.concatenate([self._postings[t][0] for t in terms])
            tfs = np.concatenate([self._postings[t][1] for t in terms])
        else:
            doc_ids, tfs = np.zeros(0, dtype=np.int64), np.zeros(0)
        return {
            "terms": np.frombuffer(json.dumps(terms, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
            "offsets": offsets,
            "doc_ids": doc_ids.astype(np.int32),
            "tfs": tfs.astype(np.float32),
            "lengths": self.lengths.astype(np.float32),
            "params": np.array([self.k1, self.b], dtype=np.float64),
        }

    @classmethod
    def from_arrays(cls, arrays) -> "BM25Index":
        self = cls.__new__(cls)
        self.k1, self.b = (float(x) for x in arrays["params"])
        self.lengths = np.asarray(arrays["lengths"], dtype=np.float64)
        self.n = len(self.lengths)
        terms = json.loads(np.asarray(arrays["terms"], dtype=np.uint8).tobytes().decode("utf-8"))
        offsets = np.asarray(arrays["offsets"])
        doc_ids = np.asarray(arrays["doc_ids"], dtype=np.int64)
        tfs = np.asarray(arrays["tfs"], dtype=np.float64)
        self._postings = {
            t: (doc_ids[offsets[i]:offsets[i + 1]], tfs[offsets[i]:offsets[i + 1]])
            for i, t in enumerate(terms)
        }
        self._finish()
        return self

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
