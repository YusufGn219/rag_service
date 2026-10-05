"""Hybrid search: meaning (embedding) + keywords (BM25), merged with Reciprocal Rank Fusion.

Try it:  python -m rag_service.search "your question"
"""
import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from rag_service.bm25 import tokenize
from rag_service.chunker import Chunk, note_title
from rag_service.config import Config, ConfigError, load_config
from rag_service.recency import DEFAULT_HALF_LIFE_DAYS, freshness, is_recency_query, note_date
from rag_service.store import IndexData, index_path, load_index

MODES = ("hybrid", "dense", "bm25")
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATES = 50
# A fully fresh note gets its score multiplied by 1 + this; small on purpose, so recency
# only decides between notes that already match about equally well.
DEFAULT_RECENCY_WEIGHT = 0.2
# A note's score = its best chunk + this share of its next best chunks' scores.
NOTE_EXTRA_WEIGHT = 0.25
NOTE_EXTRA_CHUNKS = 2


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused (RRF) score; only meaningful for ordering
    dense_score: float | None  # cosine similarity to the query (None in bm25 mode)
    bm25_score: float  # keyword score (0.0 when no query word matched)


@dataclass(frozen=True)
class NoteHit:
    path: str
    title: str  # file name without .md
    date: date | None  # from the file name, if it has one
    score: float  # note-level score; only meaningful for ordering
    hits: tuple[Hit, ...]  # this note's matching chunks, best first

    @property
    def best(self) -> Hit:
        return self.hits[0]


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
                 rrf_k: int = DEFAULT_RRF_K, today: date | None = None,
                 recency_weight: float = DEFAULT_RECENCY_WEIGHT,
                 half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                 resolver: Callable[[str], Path] | None = None):
        self._data = data
        self._embedder = embedder
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._today = today
        self._recency_weight = recency_weight
        self._half_life_days = half_life_days
        self._resolver = resolver
        self._known_notes = {c.path for c in data.chunks}
        self._dates: dict[str, date | None] = {p: note_date(p) for p in self._known_notes}

    @classmethod
    def from_config(cls, cfg: Config, embedder, **kwargs) -> "Searcher":
        data = load_index(cfg)
        if data is None:
            raise FileNotFoundError(
                f"no usable index at {index_path(cfg)}; run `python -m rag_service.indexer` first"
            )
        kwargs.setdefault("resolver", cfg.resolve_key)
        return cls(data, embedder, **kwargs)

    def _fused(self, query: str, mode: str, recency: bool | None) -> list[Hit]:
        """Every candidate chunk, best first, with the recency boost applied."""
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

        use_recency = is_recency_query(query) if recency is None else recency
        today = self._today or date.today()
        hits: list[tuple[int, float]] = []
        for doc, score in rrf(rankings, self._rrf_k):
            if use_recency:
                fresh = freshness(self._dates[data.chunks[doc].path], today, self._half_life_days)
                score *= 1.0 + self._recency_weight * fresh
            hits.append((doc, score))
        if use_recency:
            hits.sort(key=lambda ds: (-ds[1], ds[0]))
        return [
            Hit(
                chunk=data.chunks[doc],
                score=score,
                dense_score=None if dense_all is None else float(dense_all[doc]),
                bm25_score=float(bm25_all[doc]),
            )
            for doc, score in hits
        ]

    def search(self, query: str, k: int = 5, *, mode: str = "hybrid",
               max_per_note: int | None = None, recency: bool | None = None) -> list[Hit]:
        """Best chunks. recency: None = boost newer notes only for 'güncel/son/...' questions."""
        hits: list[Hit] = []
        per_note: dict[str, int] = {}
        for hit in self._fused(query, mode, recency):
            path = hit.chunk.path
            if max_per_note is not None and per_note.get(path, 0) >= max_per_note:
                continue
            per_note[path] = per_note.get(path, 0) + 1
            hits.append(hit)
            if len(hits) == k:
                break
        return hits

    def search_notes(self, query: str, k: int = 5, *, mode: str = "hybrid",
                     recency: bool | None = None) -> list[NoteHit]:
        """Best notes: chunks of one note are grouped; several good chunks lift the note."""
        by_note: dict[str, list[Hit]] = {}
        for hit in self._fused(query, mode, recency):
            by_note.setdefault(hit.chunk.path, []).append(hit)  # already best first
        notes = []
        for path, hits in by_note.items():
            extra = sum(h.score for h in hits[1 : 1 + NOTE_EXTRA_CHUNKS])
            notes.append(NoteHit(
                path=path,
                title=note_title(path),
                date=self._dates[path],
                score=hits[0].score + NOTE_EXTRA_WEIGHT * extra,
                hits=tuple(hits),
            ))
        notes.sort(key=lambda n: (-n.score, n.path))
        return notes[:k]

    def read_note(self, path: str) -> str:
        """The whole note as it is on disk. Only notes that are in the index can be read."""
        if self._resolver is None:
            raise RuntimeError("this Searcher has no vault to read notes from")
        if path not in self._known_notes:
            raise KeyError(f"{path!r} is not an indexed note")
        return self._resolver(path).read_text(encoding="utf-8", errors="replace")


def _snippet(text: str, width: int = 170) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag_service.search")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--max-per-note", type=int, default=None)
    parser.add_argument("--notes", action="store_true", help="one result per note")
    parser.add_argument("--full", action="store_true",
                        help="print the whole text of the best note (implies --notes)")
    parser.add_argument("--recency", choices=("auto", "on", "off"), default="auto",
                        help="boost newer notes: auto = only for 'güncel/son/...' questions")
    args = parser.parse_args(argv)
    recency = {"auto": None, "on": True, "off": False}[args.recency]

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

    query = " ".join(args.query)
    if args.notes or args.full:
        notes = searcher.search_notes(query, k=args.k, mode=args.mode, recency=recency)
        if not notes:
            print("no results")
        for i, n in enumerate(notes, start=1):
            when = f"  ({n.date})" if n.date else ""
            sec = f"  [{n.best.chunk.heading_path}]" if n.best.chunk.heading_path else ""
            print(f"{i}. {n.path}{sec}\n   score={n.score:.4f} chunks={len(n.hits)}{when}")
            print(f"   {_snippet(n.best.chunk.text)}")
        if args.full and notes:
            print(f"\n===== {notes[0].path} =====\n{searcher.read_note(notes[0].path)}")
        return 0

    hits = searcher.search(query, k=args.k, mode=args.mode,
                           max_per_note=args.max_per_note, recency=recency)
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
