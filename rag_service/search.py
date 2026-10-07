"""Hybrid search: meaning (embedding) + keywords (BM25), merged with Reciprocal Rank Fusion.

Try it:  python -m rag_service.search "your question"
"""
import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np

from rag_service.bm25 import tokenize
from rag_service.chunker import Chunk, note_title
from rag_service.config import Config, ConfigError, load_config
from rag_service.recency import (
    DEFAULT_HALF_LIFE_DAYS,
    folder_units,
    freshness,
    is_recency_query,
    is_state_query,
    is_status_note,
    note_date,
)
from rag_service.store import IndexData, index_path, load_index

MODES = ("hybrid", "dense", "bm25")
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATES = 50
# Recency questions ("son", "yeni", ...): a fully fresh note (its file name has a date) gets its
# score multiplied by 1 + this. Small on purpose, so recency only decides between notes that
# already match about equally well (at 0.2 it pushed specific answers down in the eval).
DEFAULT_RECENCY_WEIGHT = 0.05
# "Where are we" questions ("neredeyiz", "güncel durum", ...): the answer is the note that sums
# a project up, so an index / "Devam" note gets its score multiplied by 1 + this...
DEFAULT_STATUS_WEIGHT = 5.0
# ...and a note in the project folder that the best results point to, by 1 + this.
# Measured on the eval set: "where are we" questions went from hit@1 33% to 81%.
DEFAULT_PROJECT_WEIGHT = 0.5
# The project folder is the one that holds at least this share of the best results' weight
# and is at least this many times more common there than in the whole vault.
FOCUS_TOP = 15
FOCUS_MIN_SHARE = 0.3
FOCUS_MIN_LIFT = 3.0
# A note's score = its best chunk + this share of its next best chunks' scores.
NOTE_EXTRA_WEIGHT = 0.10
NOTE_EXTRA_CHUNKS = 2
# How many of the best fused candidates the reranker reads (it is the slow stage).
DEFAULT_RERANK_TOP = 40


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 60.0), -60.0)))


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused (RRF) score; only meaningful for ordering
    dense_score: float | None  # cosine similarity to the query (None in bm25 mode)
    bm25_score: float  # keyword score (0.0 when no query word matched)
    rerank_score: float | None = None  # reranker's 0-1 score (None if it did not read this chunk)


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


class _Ranked(NamedTuple):
    scored: list[tuple[int, float]]  # (chunk row, score), not yet sorted when the reranker ran
    rerank_scores: dict[int, float]
    dense_all: "np.ndarray | None"
    bm25_all: np.ndarray
    reranked: bool


class Searcher:
    def __init__(self, data: IndexData, embedder, *, candidates: int = DEFAULT_CANDIDATES,
                 rrf_k: int = DEFAULT_RRF_K, today: date | None = None,
                 recency_weight: float = DEFAULT_RECENCY_WEIGHT,
                 status_weight: float = DEFAULT_STATUS_WEIGHT,
                 project_weight: float = DEFAULT_PROJECT_WEIGHT,
                 half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                 resolver: Callable[[str], Path] | None = None,
                 reranker=None, rerank_top: int = DEFAULT_RERANK_TOP):
        self._reranker = reranker
        self._rerank_top = rerank_top
        self._data = data
        self._embedder = embedder
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._today = today
        self._recency_weight = recency_weight
        self._status_weight = status_weight
        self._project_weight = project_weight
        self._half_life_days = half_life_days
        self._resolver = resolver
        self._known_notes = {c.path for c in data.chunks}
        self._dates: dict[str, date | None] = {p: note_date(p) for p in self._known_notes}
        self._status = {p for p in self._known_notes if is_status_note(p)}
        self._units = {p: folder_units(p) for p in self._known_notes}
        counts: dict[str, int] = {}
        for units in self._units.values():
            for u in units:
                counts[u] = counts.get(u, 0) + 1
        self._unit_share = {u: n / max(len(self._known_notes), 1) for u, n in counts.items()}

    @classmethod
    def from_config(cls, cfg: Config, embedder, **kwargs) -> "Searcher":
        data = load_index(cfg)
        if data is None:
            raise FileNotFoundError(
                f"no usable index at {index_path(cfg)}; run `python -m rag_service.indexer` first"
            )
        kwargs.setdefault("resolver", cfg.resolve_key)
        if "reranker" not in kwargs:
            from rag_service.rerank import load_reranker

            kwargs["reranker"] = load_reranker(cfg)
        return cls(data, embedder, **kwargs)

    def _ranked(self, query: str, mode: str, rerank: bool | None) -> "_Ranked | None":
        """Candidate chunks scored by fusion and (optionally) the reranker, before any
        recency adjustment. None when there is nothing to search."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if rerank and self._reranker is None:
            raise ValueError("rerank=True needs a reranker (pass reranker= to Searcher)")
        rerank = self._reranker is not None if rerank is None else rerank
        data = self._data
        if not query.strip() or not data.chunks:
            return None

        bm25_all = data.bm25.scores(tokenize(query))
        dense_all = None
        rankings: list[list[int]] = []
        if mode in ("hybrid", "dense"):
            dense_all = data.vectors @ np.asarray(self._embedder.embed_query(query), dtype=np.float32)
            rankings.append(np.argsort(-dense_all, kind="stable")[: self._candidates].tolist())
        if mode in ("hybrid", "bm25"):
            rankings.append([d for d, _ in data.bm25.search(tokenize(query), self._candidates)])

        scored: list[tuple[int, float]] = rrf(rankings, self._rrf_k)
        rerank_scores: dict[int, float] = {}
        if rerank:
            top = scored[: self._rerank_top]
            logits = self._reranker.score(query, [data.chunks[d].embed_text for d, _ in top])
            rerank_scores = {d: _sigmoid(x) for (d, _), x in zip(top, logits)}
            # reranked chunks go first by the reranker's score; the rest keep their RRF
            # order below them (scaled to stay under the lowest reranked score)
            floor = min(rerank_scores.values(), default=1.0)
            peak = scored[len(top)][1] if len(scored) > len(top) else 1.0
            scored = [(d, rerank_scores[d]) for d, _ in top] + [
                (d, 0.5 * floor * s / peak) for d, s in scored[len(top):]
            ]

        return _Ranked(scored, rerank_scores, dense_all, bm25_all, reranked=rerank)

    def _fused(self, query: str, mode: str, recency: bool | None, rerank: bool | None) -> list[Hit]:
        """Every candidate chunk, best first: fused, optionally reranked, recency boost applied."""
        ranked = self._ranked(query, mode, rerank)
        if ranked is None:
            return []
        scored = ranked.scored
        use_recency = is_recency_query(query) if recency is None else recency
        if use_recency:
            scored = self._boost(scored, state=is_state_query(query))
        if use_recency or ranked.reranked:
            scored.sort(key=lambda ds: (-ds[1], ds[0]))
        return self._hits(ranked, scored)

    def _hits(self, ranked: "_Ranked", scored: list[tuple[int, float]]) -> list[Hit]:
        data = self._data
        return [
            Hit(
                chunk=data.chunks[doc],
                score=score,
                dense_score=None if ranked.dense_all is None else float(ranked.dense_all[doc]),
                bm25_score=float(ranked.bm25_all[doc]),
                rerank_score=ranked.rerank_scores.get(doc),
            )
            for doc, score in scored
        ]

    def _boost(self, scored: list[tuple[int, float]], state: bool = False) -> list[tuple[int, float]]:
        """Score bonuses for recency questions: how new the note is; and, when the question asks
        where a project stands (`state`), also whether the note is an index/"Devam" note and
        whether it sits in the project folder the best results point to."""
        if not scored:
            return scored
        chunks = self._data.chunks
        today = self._today or date.today()
        focus = self._focus_unit(scored) if state and self._project_weight else None
        boosted = []
        for d, s in scored:
            path = chunks[d].path
            bonus = self._recency_weight * freshness(self._dates[path], today, self._half_life_days)
            if state and path in self._status:
                bonus += self._status_weight
            if focus is not None and focus in self._units[path]:
                bonus += self._project_weight
            boosted.append((d, s * (1.0 + bonus)))
        return boosted

    def _focus_unit(self, scored: list[tuple[int, float]]) -> str | None:
        """The folder name that the best results cluster in, if one stands out.

        Among the best FOCUS_TOP notes, a folder counts when it holds at least FOCUS_MIN_SHARE of
        their weight and is FOCUS_MIN_LIFT times more common there than in the whole vault
        (that skips folders everything lives under, like the vault root).
        """
        best: dict[str, float] = {}
        for d, s in sorted(scored, key=lambda ds: -ds[1]):
            path = self._data.chunks[d].path
            if path not in best:
                best[path] = s
                if len(best) == FOCUS_TOP:
                    break
        total = sum(best.values())
        if total <= 0:
            return None
        weight: dict[str, float] = {}
        for path, s in best.items():
            for u in self._units[path]:
                weight[u] = weight.get(u, 0.0) + s
        pick, pick_score = None, 0.0
        for u, w in weight.items():
            share = w / total
            lift = share / self._unit_share[u]
            if share >= FOCUS_MIN_SHARE and lift >= FOCUS_MIN_LIFT and share * lift > pick_score:
                pick, pick_score = u, share * lift
        return pick

    def search(self, query: str, k: int = 5, *, mode: str = "hybrid",
               max_per_note: int | None = None, recency: bool | None = None,
               rerank: bool | None = None) -> list[Hit]:
        """Best chunks. recency: None = boost newer notes only for 'güncel/son/...' questions.

        rerank: None = use the reranker if the Searcher has one.
        """
        hits: list[Hit] = []
        per_note: dict[str, int] = {}
        for hit in self._fused(query, mode, recency, rerank):
            path = hit.chunk.path
            if max_per_note is not None and per_note.get(path, 0) >= max_per_note:
                continue
            per_note[path] = per_note.get(path, 0) + 1
            hits.append(hit)
            if len(hits) == k:
                break
        return hits

    def search_notes(self, query: str, k: int = 5, *, mode: str = "hybrid",
                     recency: bool | None = None, rerank: bool | None = None) -> list[NoteHit]:
        """Best notes: chunks of one note are grouped; several good chunks lift the note."""
        return self._note_hits(self._fused(query, mode, recency, rerank))[:k]

    def _note_hits(self, hits: list[Hit]) -> list[NoteHit]:
        by_note: dict[str, list[Hit]] = {}
        for hit in hits:
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
        return notes

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
    parser.add_argument("--rerank", choices=("auto", "off"), default="auto",
                        help="auto = use the reranker when its model is installed")
    parser.add_argument("--recency", choices=("auto", "on", "off"), default="auto",
                        help="boost newer notes: auto = only for 'güncel/son/...' questions")
    args = parser.parse_args(argv)
    recency = {"auto": None, "on": True, "off": False}[args.recency]
    rerank = False if args.rerank == "off" else None

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
        notes = searcher.search_notes(query, k=args.k, mode=args.mode, recency=recency,
                                      rerank=rerank)
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
                           max_per_note=args.max_per_note, recency=recency, rerank=rerank)
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
