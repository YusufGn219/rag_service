"""Hybrid search: meaning (embedding) + keywords (BM25), merged with Reciprocal Rank Fusion.

Try it:  python -m rag_service.search "your question"
"""
import argparse
import sys
from dataclasses import dataclass

import numpy as np

from rag_service.bm25 import tokenize
from rag_service.chunker import Chunk
from rag_service.config import Config, ConfigError, load_config
from rag_service.store import IndexData, index_path, load_index

MODES = ("hybrid", "dense", "bm25")
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATES = 50


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused (RRF) score; only meaningful for ordering
    dense_score: float | None  # cosine similarity to the query (None in bm25 mode)
    bm25_score: float  # keyword score (0.0 when no query word matched)


def rrf(rankings: list[list[int]], k: int = DEFAULT_RRF_K) -> list[tuple[int, float]]:
    """Merge ranked lists of doc ids: each list gives 1/(k+rank) per doc. Best first.

    Uses ranks, not raw scores, because cosine and BM25 scores live on different scales.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Searcher:
    def __init__(self, data: IndexData, embedder, *, candidates: int = DEFAULT_CANDIDATES,
                 rrf_k: int = DEFAULT_RRF_K):
        self._data = data
        self._embedder = embedder
        self._candidates = candidates
        self._rrf_k = rrf_k

    @classmethod
    def from_config(cls, cfg: Config, embedder, **kwargs) -> "Searcher":
        data = load_index(cfg)
        if data is None:
            raise FileNotFoundError(
                f"no usable index at {index_path(cfg)}; run `python -m rag_service.indexer` first"
            )
        return cls(data, embedder, **kwargs)

    def search(self, query: str, k: int = 5, *, mode: str = "hybrid",
               max_per_note: int | None = None) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        data = self._data
        if not query.strip() or not data.chunks:
            return []

        bm25_all = data.bm25.scores(tokenize(query))
        dense_all = None
        rankings: list[list[int]] = []
        if mode in ("hybrid", "dense"):
            dense_all = data.vectors @ np.asarray(self._embedder.embed_query(query), dtype=np.float32)
            rankings.append(np.argsort(-dense_all, kind="stable")[: self._candidates].tolist())
        if mode in ("hybrid", "bm25"):
            rankings.append([d for d, _ in data.bm25.search(tokenize(query), self._candidates)])

        hits: list[Hit] = []
        per_note: dict[str, int] = {}
        for doc, score in rrf(rankings, self._rrf_k):
            chunk = data.chunks[doc]
            if max_per_note is not None and per_note.get(chunk.path, 0) >= max_per_note:
                continue
            per_note[chunk.path] = per_note.get(chunk.path, 0) + 1
            hits.append(Hit(
                chunk=chunk,
                score=score,
                dense_score=None if dense_all is None else float(dense_all[doc]),
                bm25_score=float(bm25_all[doc]),
            ))
            if len(hits) == k:
                break
        return hits


def _snippet(text: str, width: int = 170) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag_service.search")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--max-per-note", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    from rag_service.embeddings import Embedder

    try:
        cfg = load_config()
        searcher = Searcher.from_config(cfg, Embedder.from_dir(cfg.model_dir))
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    hits = searcher.search(" ".join(args.query), k=args.k, mode=args.mode,
                           max_per_note=args.max_per_note)
    if not hits:
        print("no results")
    for i, h in enumerate(hits, start=1):
        dense = "-" if h.dense_score is None else f"{h.dense_score:.3f}"
        where = h.chunk.path + (f"  [{h.chunk.heading_path}]" if h.chunk.heading_path else "")
        print(f"{i}. {where}\n   rrf={h.score:.4f} dense={dense} bm25={h.bm25_score:.1f}")
        print(f"   {_snippet(h.chunk.text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
