import hashlib

import numpy as np
import pytest

from rag_service.bm25 import tokenize
from rag_service.config import load_config
from rag_service.indexer import run_index
from rag_service.manifest import load_manifest
from rag_service.store import index_path, load_index


class FakeEmbedder:
    """Deterministic 8-d vectors from a hash of the text; records what it was asked to embed."""

    def __init__(self):
        self.embedded: list[str] = []
        self.fail = False

    def count_tokens(self, text):
        return len(text.split())

    def embed_passages(self, texts):
        if self.fail:
            raise RuntimeError("embedding exploded")
        self.embedded.extend(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()[:8]
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32) + 1
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def _setup(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    cfg = load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx")})
    return vault, cfg


def _note(vault, rel, text):
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _bump_mtime(p):
    st = p.stat()
    import os

    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


BODY = "# Baslik\n" + "kelime " * 80


def test_first_run_indexes_everything(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    _note(vault, "v/b.md", "# Baska\n" + "icerik " * 80)
    emb = FakeEmbedder()
    stats = run_index(cfg, emb)
    data = load_index(cfg)
    assert stats.full_rebuild and stats.notes == 2 and stats.changed == 2
    assert len(data.chunks) == data.vectors.shape[0] == stats.chunks_total == stats.chunks_embedded > 0
    assert set(load_manifest(cfg)) == {"a.md", "v/b.md"}
    assert data.bm25.search(tokenize("icerik"))


def test_second_run_with_no_changes_does_nothing(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    run_index(cfg, FakeEmbedder())
    before = index_path(cfg).stat().st_mtime_ns
    emb = FakeEmbedder()
    stats = run_index(cfg, emb)
    assert emb.embedded == []
    assert stats.changed == 0 and stats.chunks_embedded == 0 and not stats.full_rebuild
    assert index_path(cfg).stat().st_mtime_ns == before


def test_modified_note_only_reembeds_that_note(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    b = _note(vault, "b.md", "# B\n" + "eski " * 80)
    run_index(cfg, FakeEmbedder())
    old = load_index(cfg)
    a_old = [(c, v) for c, v in zip(old.chunks, old.vectors) if c.path == "a.md"]

    _note(vault, "b.md", "# B\n" + "yepyeni " * 80)
    _bump_mtime(b)
    emb = FakeEmbedder()
    stats = run_index(cfg, emb)
    new = load_index(cfg)

    assert stats.changed == 1 and not stats.full_rebuild
    assert emb.embedded and all("yepyeni" in t for t in emb.embedded)
    a_new = [(c, v) for c, v in zip(new.chunks, new.vectors) if c.path == "a.md"]
    assert [c for c, _ in a_new] == [c for c, _ in a_old]
    assert all(np.array_equal(x, y) for (_, x), (_, y) in zip(a_new, a_old))
    assert new.bm25.search(tokenize("yepyeni"))
    assert not new.bm25.search(tokenize("eski"))


def test_new_note_only_embeds_new_note(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    run_index(cfg, FakeEmbedder())
    _note(vault, "c.md", "# C\n" + "tazecik " * 80)
    emb = FakeEmbedder()
    run_index(cfg, emb)
    assert emb.embedded and all("tazecik" in t for t in emb.embedded)
    assert {c.path for c in load_index(cfg).chunks} == {"a.md", "c.md"}


def test_deleted_note_removed_from_index_and_bm25(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    b = _note(vault, "b.md", "# B\n" + "silinecek " * 80)
    run_index(cfg, FakeEmbedder())
    b.unlink()
    emb = FakeEmbedder()
    stats = run_index(cfg, emb)
    data = load_index(cfg)
    assert stats.deleted == 1 and emb.embedded == []
    assert {c.path for c in data.chunks} == {"a.md"}
    assert data.vectors.shape[0] == len(data.chunks)
    assert not data.bm25.search(tokenize("silinecek"))
    assert "b.md" not in load_manifest(cfg)


def test_chunk_order_is_sorted_by_path_then_index(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "z.md", BODY)
    _note(vault, "a.md", "# A\n" + "x " * 600)
    run_index(cfg, FakeEmbedder())
    keys = [(c.path, c.index) for c in load_index(cfg).chunks]
    assert keys == sorted(keys)


def test_changed_chunk_settings_force_full_rebuild(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    run_index(cfg, FakeEmbedder())
    emb = FakeEmbedder()
    stats = run_index(cfg, emb, max_tokens=50)
    assert stats.full_rebuild and emb.embedded
    assert load_index(cfg).settings["max_tokens"] == 50


def test_changed_model_id_forces_full_rebuild(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    run_index(cfg, FakeEmbedder(), model_id="model-one")
    stats = run_index(cfg, FakeEmbedder(), model_id="model-two")
    assert stats.full_rebuild


def test_failed_embedding_leaves_index_and_manifest_untouched(tmp_path):
    vault, cfg = _setup(tmp_path)
    a = _note(vault, "a.md", BODY)
    run_index(cfg, FakeEmbedder())
    index_before = index_path(cfg).read_bytes()
    manifest_before = load_manifest(cfg)

    _note(vault, "a.md", "# A\n" + "degisti " * 80)
    _bump_mtime(a)
    bad = FakeEmbedder()
    bad.fail = True
    with pytest.raises(RuntimeError):
        run_index(cfg, bad)
    assert index_path(cfg).read_bytes() == index_before
    assert load_manifest(cfg) == manifest_before

    run_index(cfg, FakeEmbedder())  # retry succeeds and picks the change up
    assert load_index(cfg).bm25.search(tokenize("degisti"))


def test_missing_manifest_reprocesses_without_duplicates(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    run_index(cfg, FakeEmbedder())
    n = len(load_index(cfg).chunks)
    (cfg.index_dir / "manifest.json").unlink()
    emb = FakeEmbedder()
    run_index(cfg, emb)
    assert emb.embedded
    assert len(load_index(cfg).chunks) == n


def test_missing_index_with_manifest_rebuilds_everything(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    run_index(cfg, FakeEmbedder())
    index_path(cfg).unlink()
    emb = FakeEmbedder()
    stats = run_index(cfg, emb)
    assert stats.full_rebuild and emb.embedded and load_index(cfg) is not None


def test_empty_vault_produces_empty_index(tmp_path):
    _, cfg = _setup(tmp_path)
    stats = run_index(cfg, FakeEmbedder())
    data = load_index(cfg)
    assert stats.chunks_total == 0 and data.chunks == []


def test_note_with_no_text_is_tracked_but_has_no_chunks(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "empty.md", "")
    _note(vault, "a.md", BODY)
    run_index(cfg, FakeEmbedder())
    assert "empty.md" in load_manifest(cfg)
    assert {c.path for c in load_index(cfg).chunks} == {"a.md"}
    emb = FakeEmbedder()
    run_index(cfg, emb)
    assert emb.embedded == []


def test_progress_callback_reports_totals(tmp_path):
    vault, cfg = _setup(tmp_path)
    _note(vault, "a.md", BODY)
    calls = []
    run_index(cfg, FakeEmbedder(), progress=lambda done, total: calls.append((done, total)))
    assert calls and calls[-1][0] == calls[-1][1] > 0


def test_indexes_several_roots_and_updates_only_the_changed_one(tmp_path):
    import os

    a, b = tmp_path / "alfa", tmp_path / "beta"
    _note(a, "a.md", BODY)
    nb = _note(b, "sub/b.md", "# B\n" + "eski " * 80)
    cfg = load_config({"RAG_VAULT_ROOT": os.pathsep.join([str(a), str(b)]),
                       "RAG_INDEX_DIR": str(tmp_path / "idx")})
    run_index(cfg, FakeEmbedder())
    assert {c.path for c in load_index(cfg).chunks} == {"alfa/a.md", "beta/sub/b.md"}

    _note(b, "sub/b.md", "# B\n" + "yepyeni " * 80)
    _bump_mtime(nb)
    emb = FakeEmbedder()
    stats = run_index(cfg, emb)
    assert stats.changed == 1 and all("yepyeni" in t for t in emb.embedded)
    assert load_index(cfg).bm25.search(tokenize("yepyeni"))


# ---- index lock ----

def test_run_index_refuses_while_locked(tmp_path):
    from rag_service.lock import IndexLocked, index_lock

    vault, cfg = _setup(tmp_path)
    (vault / "a.md").write_text("# A\n\nsome text here", encoding="utf-8")
    with index_lock(cfg):
        with pytest.raises(IndexLocked):
            run_index(cfg, FakeEmbedder())
    assert load_index(cfg) is None


def test_run_index_releases_lock_after_failure(tmp_path):
    from rag_service.lock import lock_path

    vault, cfg = _setup(tmp_path)
    (vault / "a.md").write_text("# A\n\nsome text here", encoding="utf-8")
    emb = FakeEmbedder()
    emb.fail = True
    with pytest.raises(RuntimeError):
        run_index(cfg, emb)
    assert not lock_path(cfg).exists()
