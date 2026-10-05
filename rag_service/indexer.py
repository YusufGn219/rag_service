"""Build and update the search index. Only new or changed notes are re-chunked and re-embedded.

Run with:  python -m rag_service.indexer
"""
import sys
import time
from dataclasses import dataclass

import numpy as np

from rag_service.bm25 import TOKENIZER_VERSION, index_chunks
from rag_service.chunker import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    chunk_note,
)
from rag_service.config import Config, ConfigError, load_config
from rag_service.embeddings import MODEL_REPO
from rag_service.manifest import compute_changes, load_manifest, save_manifest, snapshot
from rag_service.scanner import scan_notes
from rag_service.store import IndexData, load_index, save_index

FORMAT_VERSION = 1
_EMBED_BATCH = 256


@dataclass(frozen=True)
class IndexStats:
    notes: int  # notes found in the vault
    changed: int  # new or modified notes processed this run
    deleted: int  # notes removed from the index this run
    chunks_total: int  # chunks in the index afterwards
    chunks_embedded: int  # chunks embedded this run
    full_rebuild: bool  # True when nothing from an older index could be reused
    seconds: float


def run_index(
    cfg: Config,
    embedder,
    *,
    model_id: str = MODEL_REPO,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    progress=None,
) -> IndexStats:
    """Bring the stored index in line with the vault.

    Order matters for crash safety: embed first, then write the index, and the manifest
    last. If anything fails before the manifest is written, the next run simply redoes
    the same notes.
    """
    started = time.monotonic()
    settings = {
        "format": FORMAT_VERSION,
        "model": model_id,
        "tokenizer": TOKENIZER_VERSION,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "min_tokens": min_tokens,
    }

    snap = snapshot(cfg, scan_notes(cfg))
    data = load_index(cfg)
    full = data is None or data.settings != settings
    if full:
        data = None
    old_manifest = {} if full else load_manifest(cfg)
    changes = compute_changes(old_manifest, snap)

    if data is not None and not changes.changed and not changes.deleted:
        return IndexStats(len(snap), 0, 0, len(data.chunks), 0, False, time.monotonic() - started)

    drop = set(changes.changed) | set(changes.deleted)
    kept_rows = [i for i, c in enumerate(data.chunks) if c.path not in drop] if data else []
    kept_chunks = [data.chunks[i] for i in kept_rows]
    kept_vectors = data.vectors[kept_rows] if kept_rows else None

    new_chunks = []
    for rel in changes.changed:
        try:
            text = (cfg.vault_root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            snap.pop(rel, None)  # vanished since the scan; treat as deleted next time
            continue
        new_chunks.extend(
            chunk_note(rel, text, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
                       min_tokens=min_tokens, count=embedder.count_tokens)
        )

    texts = [c.embed_text for c in new_chunks]
    parts = []
    for start in range(0, len(texts), _EMBED_BATCH):
        parts.append(embedder.embed_passages(texts[start:start + _EMBED_BATCH]))
        if progress:
            progress(min(start + _EMBED_BATCH, len(texts)), len(texts))
    new_vectors = np.vstack(parts) if parts else None

    chunks = kept_chunks + new_chunks
    vector_parts = [v for v in (kept_vectors, new_vectors) if v is not None]
    vectors = np.vstack(vector_parts).astype(np.float32) if vector_parts else np.zeros((0, 0), np.float32)
    order = sorted(range(len(chunks)), key=lambda i: (chunks[i].path, chunks[i].index))
    if order:
        chunks = [chunks[i] for i in order]
        vectors = vectors[order]

    save_index(cfg, IndexData(chunks=chunks, vectors=vectors, bm25=index_chunks(chunks), settings=settings))
    save_manifest(cfg, snap)
    return IndexStats(
        notes=len(snap),
        changed=len(changes.changed),
        deleted=len(changes.deleted),
        chunks_total=len(chunks),
        chunks_embedded=len(new_chunks),
        full_rebuild=full,
        seconds=time.monotonic() - started,
    )


def _print_progress(done: int, total: int) -> None:
    print(f"\r  embedding chunks: {done}/{total}", end="", flush=True)
    if done == total:
        print()


def main() -> int:
    from rag_service.embeddings import Embedder

    try:
        cfg = load_config()
        embedder = Embedder.from_dir(cfg.model_dir)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    stats = run_index(cfg, embedder, progress=_print_progress)
    print(
        f"notes: {stats.notes} | changed: {stats.changed} | deleted: {stats.deleted} | "
        f"chunks: {stats.chunks_total} (embedded now: {stats.chunks_embedded}) | "
        f"{'full rebuild' if stats.full_rebuild else 'incremental'} | {stats.seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
