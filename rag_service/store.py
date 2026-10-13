"""Save and load the whole search index (chunks + vectors + BM25) as one file.

One file, replaced atomically, so the three parts can never disagree after a crash.
"""
import json
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

from rag_service.bm25 import BM25Index
from rag_service.chunker import Chunk, Parent
from rag_service.config import Config

_INDEX_NAME = "index.npz"
_BM25_PREFIX = "bm25_"


@dataclass
class IndexData:
    chunks: list[Chunk]
    vectors: np.ndarray  # (len(chunks), dim) float32, one unit-length row per chunk
    bm25: BM25Index
    settings: dict  # what the index was built with (model, chunk sizes, tokenizer...)
    parents: list[Parent] = field(default_factory=list)  # sections that Chunk.parent_id points to


def index_path(cfg: Config):
    return cfg.index_dir / _INDEX_NAME


def _pack(obj) -> np.ndarray:
    return np.frombuffer(json.dumps(obj, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)


def _unpack(arr: np.ndarray):
    return json.loads(np.asarray(arr, dtype=np.uint8).tobytes().decode("utf-8"))


def save_index(cfg: Config, data: IndexData) -> None:
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    path = index_path(cfg)
    tmp = path.with_name("index.tmp.npz")
    arrays = {
        "vectors": np.asarray(data.vectors, dtype=np.float32),
        "chunks": _pack(
            [
                {"path": c.path, "index": c.index, "heading_path": c.heading_path,
                 "text": c.text, "tags": list(c.tags), "parent_id": c.parent_id}
                for c in data.chunks
            ]
        ),
        "parents": _pack(
            [
                {"path": p.path, "id": p.id, "heading_path": p.heading_path, "text": p.text}
                for p in data.parents
            ]
        ),
        "settings": _pack(data.settings),
    }
    arrays.update({_BM25_PREFIX + k: v for k, v in data.bm25.to_arrays().items()})
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_index(cfg: Config) -> IndexData | None:
    """The stored index, or None if it is missing, unreadable or inconsistent."""
    try:
        with np.load(index_path(cfg), allow_pickle=False) as z:
            chunks = [
                Chunk(path=d["path"], index=d["index"], heading_path=d["heading_path"],
                      text=d["text"], tags=tuple(d["tags"]), parent_id=d.get("parent_id", -1))
                for d in _unpack(z["chunks"])
            ]
            parents = (
                [Parent(path=d["path"], id=d["id"], heading_path=d["heading_path"], text=d["text"])
                 for d in _unpack(z["parents"])]
                if "parents" in z.files else []
            )
            vectors = np.asarray(z["vectors"], dtype=np.float32)
            settings = _unpack(z["settings"])
            bm25 = BM25Index.from_arrays(
                {k[len(_BM25_PREFIX):]: z[k] for k in z.files if k.startswith(_BM25_PREFIX)}
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None
    if not (len(chunks) == vectors.shape[0] == bm25.n):
        return None
    return IndexData(chunks=chunks, vectors=vectors, bm25=bm25, settings=settings, parents=parents)
