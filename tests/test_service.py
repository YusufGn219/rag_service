import json

import numpy as np
import pytest

from rag_service.config import load_config
from rag_service.indexer import run_index
from rag_service.lock import index_lock
from rag_service.search import Searcher
from rag_service.service import Busy, NoIndex, SearchService, ServiceError


class FakeEmbedder:
    """Counts how often it was created; vectors are deterministic per text."""

    created = 0

    def __init__(self):
        type(self).created += 1

    def count_tokens(self, text):
        return len(text.split())

    def _vec(self, text):
        v = np.zeros(8, dtype=np.float32)
        for w in text.lower().split():
            v[sum(map(ord, w)) % 8] += 1.0
        v[0] += 0.01
        return v / np.linalg.norm(v)

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, 0), np.float32)

    def embed_query(self, text):
        return self._vec(text)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def _reset_counter():
    FakeEmbedder.created = 0


def _make(tmp_path, *, idle=10, reindex=30, build=True, notes=None):
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    for name, text in (notes or {"kedi.md": "# Kedi\n\nkedi süt içiyor sessizce\n",
                                 "araba.md": "# Araba\n\naraba bozuldu tamirciye gitti\n"}).items():
        (vault / name).write_text(text, encoding="utf-8")
    cfg = load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx"),
                       "RAG_IDLE_MINUTES": str(idle), "RAG_REINDEX_MINUTES": str(reindex),
                       "RAG_RERANK_DIR": ""})
    if build:
        run_index(cfg, FakeEmbedder())
        FakeEmbedder.created = 0
    clock = Clock()
    svc = SearchService(cfg, embedder_factory=lambda c: FakeEmbedder(),
                        searcher_factory=lambda c, e: Searcher.from_config(c, e, reranker=None),
                        clock=clock)
    return svc, cfg, vault, clock


def test_models_load_lazily_and_once(tmp_path):
    svc, *_ = _make(tmp_path)
    assert FakeEmbedder.created == 0
    assert svc.status()["loaded"] is False
    svc.search("kedi")
    svc.search("araba")
    assert FakeEmbedder.created == 1
    assert svc.status()["loaded"] is True


def test_result_shape(tmp_path):
    svc, *_ = _make(tmp_path)
    res = svc.search("kedi süt", k=3)
    top = res[0]
    assert top["path"] == "kedi.md" and top["title"] == "kedi"
    assert set(top) >= {"path", "title", "date", "score", "matched_chunks", "section", "text", "extra"}
    assert "süt" in top["text"]
    assert top["extra"] == []
    json.dumps(res)  # must be plain JSON data


def test_text_is_truncated_and_extras_capped(tmp_path):
    long = "# Baslik\n\n" + "\n\n".join(f"## B{i}\n\nkedi " + "kelime " * 150 for i in range(5))
    svc, *_ = _make(tmp_path, notes={"uzun.md": long})
    top = svc.search("kedi kelime")[0]
    assert len(top["text"]) <= 600
    assert len(top["extra"]) <= 2
    assert all(len(e["text"]) <= 600 for e in top["extra"])


def test_empty_query_returns_nothing(tmp_path):
    svc, *_ = _make(tmp_path)
    assert svc.search("   ") == []


def test_idle_unload(tmp_path):
    svc, _, _, clock = _make(tmp_path, idle=10)
    svc.search("kedi")
    clock.now += 9 * 60
    assert "unloaded" not in svc.maintain()
    assert svc.status()["loaded"] is True
    clock.now += 2 * 60
    assert "unloaded" in svc.maintain()
    assert svc.status()["loaded"] is False


def test_recent_use_postpones_unload(tmp_path):
    svc, _, _, clock = _make(tmp_path, idle=10)
    svc.search("kedi")
    clock.now += 9 * 60
    svc.search("kedi")
    clock.now += 9 * 60
    assert svc.maintain() == []
    assert svc.status()["loaded"] is True


def test_reload_after_unload(tmp_path):
    svc, _, _, clock = _make(tmp_path, idle=1, reindex=0)
    svc.search("kedi")
    clock.now += 120
    svc.maintain()
    assert svc.search("kedi")[0]["path"] == "kedi.md"
    assert FakeEmbedder.created == 2


def test_zero_idle_never_unloads(tmp_path):
    svc, _, _, clock = _make(tmp_path, idle=0)
    svc.search("kedi")
    clock.now += 10**6
    assert "unloaded" not in svc.maintain()
    assert svc.status()["loaded"] is True


def test_read_note_and_safety(tmp_path):
    svc, cfg, vault, _ = _make(tmp_path)
    assert "kedi süt" in svc.read_note("kedi.md")
    (tmp_path / "secret.md").write_text("gizli", encoding="utf-8")
    (vault / "unindexed.md").write_text("henüz yok", encoding="utf-8")
    for bad in ("../secret.md", "unindexed.md", "nope.md", str(tmp_path / "secret.md"), ""):
        with pytest.raises(ServiceError):
            svc.read_note(bad)


def test_status_fields(tmp_path):
    svc, *_ = _make(tmp_path)
    st = svc.status()
    assert st["index_present"] is True and st["notes"] == 2
    assert st["reindexing"] is False and st["last_reindex"] is None


def test_reindex_sees_new_note(tmp_path):
    svc, _, vault, _ = _make(tmp_path)
    (vault / "yeni.md").write_text("# Yeni\n\nbalık tutmak güzel\n", encoding="utf-8")
    stats = svc.reindex()
    assert stats["changed"] == 1
    assert any(r["path"] == "yeni.md" for r in svc.search("balık tutmak"))
    assert svc.status()["last_reindex"]["changed"] == 1


def test_reindex_without_changes_loads_nothing(tmp_path):
    svc, *_ = _make(tmp_path)
    stats = svc.reindex()
    assert stats["changed"] == 0
    assert FakeEmbedder.created == 0
    assert svc.status()["loaded"] is False


def test_reindex_refreshes_loaded_searcher(tmp_path):
    svc, _, vault, _ = _make(tmp_path)
    svc.search("kedi")  # loaded, old index
    (vault / "yeni.md").write_text("# Yeni\n\nbalık tutmak güzel\n", encoding="utf-8")
    svc.reindex()
    assert any(r["path"] == "yeni.md" for r in svc.search("balık tutmak"))


def test_reindex_busy_when_locked(tmp_path):
    svc, cfg, *_ = _make(tmp_path)
    with index_lock(cfg):
        with pytest.raises(Busy):
            svc.reindex()
    assert svc.status()["reindexing"] is False


def test_search_still_works_while_indexing(tmp_path):
    svc, cfg, *_ = _make(tmp_path)
    with index_lock(cfg):
        assert svc.search("kedi")[0]["path"] == "kedi.md"
        assert svc.status()["reindexing"] is True  # lock held by someone (here: this process)


def test_no_index_gives_clear_error(tmp_path):
    svc, *_ = _make(tmp_path, build=False)
    with pytest.raises(NoIndex, match="reindex"):
        svc.search("kedi")
    assert svc.status()["index_present"] is False
    with pytest.raises(ServiceError):
        svc.read_note("kedi.md")


def test_maintain_reindexes_on_schedule(tmp_path):
    svc, _, vault, clock = _make(tmp_path, idle=0, reindex=30)
    (vault / "yeni.md").write_text("# Yeni\n\nbalık tutmak güzel\n", encoding="utf-8")
    clock.now += 29 * 60
    assert "reindexed" not in svc.maintain()
    clock.now += 2 * 60
    assert "reindexed" in svc.maintain()
    assert svc.status()["last_reindex"]["changed"] == 1
    clock.now += 5 * 60
    assert "reindexed" not in svc.maintain()  # next one only after another full interval


def test_zero_reindex_minutes_disables_it(tmp_path):
    svc, _, _, clock = _make(tmp_path, reindex=0)
    clock.now += 10**6
    assert "reindexed" not in svc.maintain()


def test_maintain_survives_busy_and_errors(tmp_path):
    svc, cfg, vault, clock = _make(tmp_path, reindex=1)
    clock.now += 120
    with index_lock(cfg):
        assert svc.maintain() == []  # busy: skipped quietly
    clock.now += 120
    (vault / "yeni.md").write_text("# Yeni\n\nbalık\n", encoding="utf-8")

    def boom(_):
        raise RuntimeError("disk on fire")

    svc._embedder_factory = boom
    assert svc.maintain() == []
    assert "disk on fire" in svc.status()["last_reindex_error"]
