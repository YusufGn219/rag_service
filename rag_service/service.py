"""The search service core: models load on first use, unload when idle, the index stays fresh.

No web code here; api.py and mcp_server.py are thin layers over SearchService.
"""
import gc
import threading
import time
from collections.abc import Callable
from dataclasses import asdict

from rag_service.config import Config
from rag_service.errors import Busy, NoIndex, ServiceError
from rag_service.indexer import run_index
from rag_service.lock import IndexLocked, lock_held
from rag_service.manifest import load_manifest
from rag_service.search import Searcher
from rag_service.store import index_path

SNIPPET_CHARS = 600
EXTRA_CHUNKS = 2
MAX_RESULTS = 20


def _default_embedder(cfg: Config):
    from rag_service.embeddings import Embedder

    return Embedder.from_dir(cfg.model_dir)


def _default_searcher(cfg: Config, embedder) -> Searcher:
    return Searcher.from_config(cfg, embedder)


def _clip(text: str, limit: int = SNIPPET_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class _LazyEmbedder:
    """Stands in for the embedder during a reindex: the real one loads only if it is needed."""

    def __init__(self, get):
        self._get = get

    def __getattr__(self, name):
        return getattr(self._get(), name)


class SearchService:
    def __init__(self, cfg: Config, *, embedder_factory: Callable = _default_embedder,
                 searcher_factory: Callable = _default_searcher, clock: Callable[[], float] = time.monotonic):
        self._cfg = cfg
        self._embedder_factory = embedder_factory
        self._searcher_factory = searcher_factory
        self._clock = clock
        self._mu = threading.RLock()  # guards the loaded models
        self._embedder = None
        self._searcher = None
        self._last_used = clock()
        self._last_reindex_try = clock()
        self._last_reindex: dict | None = None
        self._last_reindex_error: str | None = None

    # ---- models ----

    def _get_embedder(self):
        with self._mu:
            if self._embedder is None:
                self._embedder = self._embedder_factory(self._cfg)
            return self._embedder

    def _get_searcher(self) -> Searcher:
        with self._mu:
            if self._searcher is None:
                if not index_path(self._cfg).is_file():
                    raise NoIndex("there is no index yet; call reindex first (it takes a while)")
                try:
                    self._searcher = self._searcher_factory(self._cfg, self._get_embedder())
                except FileNotFoundError as exc:
                    raise NoIndex(f"{exc}; call reindex") from exc
            self._last_used = self._clock()
            return self._searcher

    def _unload(self) -> None:
        with self._mu:
            self._searcher = None
            self._embedder = None
        gc.collect()

    # ---- operations ----

    def search(self, query: str, k: int = 5) -> list[dict]:
        """The best notes for a question, as plain JSON-ready dicts."""
        if not query.strip():
            return []
        k = max(1, min(int(k), MAX_RESULTS))
        results = []
        for note in self._get_searcher().search_notes(query, k=k):
            best, extras = note.hits[0], note.hits[1:1 + EXTRA_CHUNKS]
            results.append({
                "path": note.path,
                "title": note.title,
                "date": note.date.isoformat() if note.date else None,
                "score": round(note.score, 4),
                "matched_chunks": len(note.hits),
                "section": best.chunk.heading_path,
                "text": _clip(best.chunk.text),
                "extra": [{"section": h.chunk.heading_path, "text": _clip(h.chunk.text)} for h in extras],
            })
        return results

    def read_note(self, path: str) -> str:
        """A whole note, but only one that is in the index (so `../` and stray paths fail)."""
        if path not in load_manifest(self._cfg):
            raise ServiceError(f"{path!r} is not an indexed note")
        try:
            file = self._cfg.resolve_key(path)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        if not any(file.resolve().is_relative_to(root) for root in self._cfg.vault_roots):
            raise ServiceError(f"{path!r} is outside the vault")
        try:
            return file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ServiceError(f"cannot read {path!r}: {exc}") from exc

    def status(self) -> dict:
        with self._mu:
            loaded = self._searcher is not None or self._embedder is not None
            idle = round(self._clock() - self._last_used, 1) if loaded else None
        return {
            "loaded": loaded,
            "idle_seconds": idle,
            "index_present": index_path(self._cfg).is_file(),
            "notes": len(load_manifest(self._cfg)),
            "reindexing": lock_held(self._cfg),
            "last_reindex": self._last_reindex,
            "last_reindex_error": self._last_reindex_error,
        }

    def reindex(self) -> dict:
        """Bring the index up to date. Searches keep using the old index until it is done."""
        try:
            stats = run_index(self._cfg, _LazyEmbedder(self._get_embedder))
        except IndexLocked as exc:
            raise Busy(str(exc)) from exc
        result = asdict(stats)
        result["seconds"] = round(result["seconds"], 1)
        with self._mu:
            self._last_reindex = result
            self._last_reindex_error = None
            if self._embedder is not None:
                self._last_used = self._clock()
            if stats.changed or stats.deleted or stats.full_rebuild:
                self._searcher = None  # reload lazily so searches see the new index
                if self._embedder is not None:
                    try:
                        self._searcher = self._searcher_factory(self._cfg, self._embedder)
                    except FileNotFoundError:
                        pass
        return result

    def maintain(self) -> list[str]:
        """One housekeeping round (the server calls it every so often): unload if idle, reindex if due."""
        done = []
        now = self._clock()
        idle_s, every_s = self._cfg.idle_minutes * 60, self._cfg.reindex_minutes * 60
        with self._mu:
            loaded = self._searcher is not None or self._embedder is not None
            idle_for = now - self._last_used
        if idle_s > 0 and loaded and idle_for >= idle_s:
            self._unload()
            done.append("unloaded")
        if every_s > 0 and now - self._last_reindex_try >= every_s:
            self._last_reindex_try = now
            try:
                self.reindex()
                done.append("reindexed")
            except Busy:
                pass
            except Exception as exc:  # a failed refresh must not kill the housekeeping thread
                self._last_reindex_error = f"{type(exc).__name__}: {exc}"
        return done
