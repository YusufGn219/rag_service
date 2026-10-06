from datetime import date

import numpy as np
import pytest

from rag_service.bm25 import index_chunks
from rag_service.chunker import Chunk
from rag_service.config import load_config
from rag_service.search import NoteHit, Searcher, rrf
from rag_service.store import IndexData, save_index

# 3-d "meaning space": x = animals, y = vehicles, z = finance
ANIMALS = np.array([1.0, 0.0, 0.0], dtype=np.float32)
VEHICLES = np.array([0.0, 1.0, 0.0], dtype=np.float32)
FINANCE = np.array([0.0, 0.0, 1.0], dtype=np.float32)


class FakeEmbedder:
    """embed_query maps a few known query strings to fixed vectors."""

    QUERIES = {
        "evcil hayvan": ANIMALS,
        "otomobil": VEHICLES,
        "kredi": FINANCE,
        "ConfigError": FINANCE,  # deliberately 'wrong' meaning: only the keyword can find it
    }

    def embed_query(self, text):
        return self.QUERIES.get(text, np.ones(3, dtype=np.float32) / np.sqrt(3))


def _chunk(path, idx, text, heading=""):
    return Chunk(path=path, index=idx, heading_path=heading, text=text, tags=())


def _searcher(**kw):
    chunks = [
        _chunk("kedi.md", 0, "kedi uyuyor sessizce"),  # 0 animals
        _chunk("kedi.md", 1, "kedi süt içiyor"),  # 1 animals
        _chunk("araba.md", 0, "araba bozuldu tamirciye gitti"),  # 2 vehicles
        _chunk("banka.md", 0, "faiz oranları yükseldi"),  # 3 finance
        _chunk("kod.md", 0, "ConfigError hatası fırlatılır"),  # 4 (meaning: vehicles)
    ]
    vectors = np.stack([ANIMALS, ANIMALS, VEHICLES, FINANCE, VEHICLES])
    data = IndexData(chunks=chunks, vectors=vectors, bm25=index_chunks(chunks), settings={})
    return Searcher(data, FakeEmbedder(), **kw)


# ---- rrf ----

def test_rrf_scores_by_rank():
    out = dict(rrf([[10, 20, 30]], k=60))
    assert out[10] == pytest.approx(1 / 61)
    assert out[20] == pytest.approx(1 / 62)
    assert out[30] == pytest.approx(1 / 63)


def test_rrf_item_in_both_lists_beats_item_in_one():
    fused = rrf([[1, 2, 3], [4, 1, 5]])
    assert fused[0][0] == 1


def test_rrf_ties_resolve_by_id_and_empty_input():
    assert [d for d, _ in rrf([[7], [3]])] == [3, 7]
    assert rrf([]) == [] and rrf([[], []]) == []


# ---- searcher ----

def test_dense_mode_finds_by_meaning_without_shared_words():
    hits = _searcher().search("evcil hayvan", k=2, mode="dense")
    assert {h.chunk.path for h in hits} == {"kedi.md"}


def test_bm25_mode_finds_by_keyword_only():
    hits = _searcher().search("faiz", k=3, mode="bm25")
    assert [h.chunk.path for h in hits] == ["banka.md"]


def test_hybrid_surfaces_both_semantic_and_exact_matches():
    paths = [h.chunk.path for h in _searcher().search("ConfigError", k=5)]
    assert "kod.md" in paths[:2]  # exact keyword found although its meaning vector is elsewhere
    paths = [h.chunk.path for h in _searcher().search("evcil hayvan", k=2)]
    assert set(paths) == {"kedi.md"}  # semantic match, no shared keyword


def test_hybrid_ranks_chunk_found_by_both_methods_first():
    # "otomobil" -> vehicles vector (araba.md, kod.md); no keyword match anywhere.
    # "araba otomobil" has the keyword 'araba' too, but the embedder maps it to the neutral vector.
    s = _searcher()
    hits = s.search("araba", k=3)  # neutral dense vector, keyword hits araba.md only
    assert hits[0].chunk.path == "araba.md"
    assert hits[0].bm25_score > 0


def test_hit_carries_scores():
    hit = _searcher().search("evcil hayvan", k=1)[0]
    assert hit.dense_score == pytest.approx(1.0)
    assert hit.bm25_score == 0.0
    assert hit.score > 0


def test_k_limits_results_and_empty_query_returns_nothing():
    s = _searcher()
    assert len(s.search("evcil hayvan", k=1)) == 1
    assert s.search("", k=5) == []
    assert s.search("   ", k=5) == []


def test_max_per_note_caps_chunks_from_one_note():
    hits = _searcher().search("evcil hayvan", k=5, max_per_note=1)
    counts = {}
    for h in hits:
        counts[h.chunk.path] = counts.get(h.chunk.path, 0) + 1
    assert max(counts.values()) == 1
    assert len(hits) == 4  # one chunk per note


def test_empty_index_returns_nothing():
    data = IndexData(chunks=[], vectors=np.zeros((0, 0), dtype=np.float32),
                     bm25=index_chunks([]), settings={})
    assert Searcher(data, FakeEmbedder()).search("kredi") == []


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        _searcher().search("kredi", mode="magic")


# ---- loading ----

def test_from_config_loads_saved_index_and_fails_clearly_without_one(tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    cfg = load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx"),
                       "RAG_RERANK_DIR": ""})
    with pytest.raises(FileNotFoundError, match="indexer"):
        Searcher.from_config(cfg, FakeEmbedder())
    chunks = [_chunk("kedi.md", 0, "kedi uyuyor")]
    save_index(cfg, IndexData(chunks=chunks, vectors=np.stack([ANIMALS]),
                              bm25=index_chunks(chunks), settings={}))
    assert Searcher.from_config(cfg, FakeEmbedder()).search("evcil hayvan")[0].chunk.path == "kedi.md"


# ---- recency ----

TODAY = date(2026, 10, 5)


def _dated_searcher(**kw):
    chunks = [
        _chunk("eski (2025-01-01).md", 0, "proje durumu raporu"),  # 0
        _chunk("yeni (2026-10-01).md", 0, "proje durumu raporu"),  # 1 identical text
        _chunk("tarihsiz.md", 0, "proje durumu raporu"),  # 2
    ]
    vectors = np.stack([ANIMALS] * 3)
    data = IndexData(chunks=chunks, vectors=vectors, bm25=index_chunks(chunks), settings={})
    return Searcher(data, FakeEmbedder(), today=TODAY, **kw)


def test_without_recency_words_dates_change_nothing():
    s = _dated_searcher()
    assert s.search("proje durumu", k=3) == s.search("proje durumu", k=3, recency=False)


def test_recency_query_lifts_newer_note():
    paths = [h.chunk.path for h in _dated_searcher().search("güncel proje durumu", k=3)]
    assert paths[0] == "yeni (2026-10-01).md"


def test_recency_can_be_forced_on_or_off():
    s = _dated_searcher()
    assert s.search("proje durumu", k=1, recency=True)[0].chunk.path == "yeni (2026-10-01).md"
    assert s.search("güncel proje durumu", k=1, recency=False)[0].chunk.path == "eski (2025-01-01).md"


def test_recency_boost_is_small_and_never_beats_a_much_better_match():
    chunks = [
        _chunk("alakali.md", 0, "kredi faiz oranları"),  # 0 matches the question well
        _chunk("yeni (2026-10-04).md", 0, "tamamen baska bir konu"),  # 1
    ]
    vectors = np.stack([FINANCE, ANIMALS])
    data = IndexData(chunks=chunks, vectors=vectors, bm25=index_chunks(chunks), settings={})
    s = Searcher(data, FakeEmbedder(), today=TODAY)
    assert s.search("güncel kredi", k=1)[0].chunk.path == "alakali.md"


# ---- note-level results ----

def test_search_notes_returns_one_result_per_note_best_chunk_first():
    notes = _searcher().search_notes("evcil hayvan", k=5)
    paths = [n.path for n in notes]
    assert len(paths) == len(set(paths))
    kedi = next(n for n in notes if n.path == "kedi.md")
    assert isinstance(kedi, NoteHit)
    assert len(kedi.hits) == 2 and kedi.best is kedi.hits[0]
    assert kedi.title == "kedi"


def test_search_notes_k_limits_notes_not_chunks():
    assert len(_searcher().search_notes("evcil hayvan", k=1)) == 1


def test_note_with_several_matching_chunks_beats_note_with_one():
    chunks = [
        _chunk("tek.md", 0, "rapor ozeti"),  # 0
        _chunk("cok.md", 0, "rapor ozeti"),  # 1
        _chunk("cok.md", 1, "rapor ozeti"),  # 2
        _chunk("cok.md", 2, "rapor ozeti"),  # 3
    ]
    vectors = np.stack([ANIMALS] * 4)
    data = IndexData(chunks=chunks, vectors=vectors, bm25=index_chunks(chunks), settings={})
    notes = Searcher(data, FakeEmbedder()).search_notes("rapor", k=2)
    assert [n.path for n in notes] == ["cok.md", "tek.md"]


def test_search_notes_carries_date_and_applies_recency():
    notes = _dated_searcher().search_notes("güncel proje durumu", k=3)
    assert notes[0].path == "yeni (2026-10-01).md"
    assert notes[0].date == date(2026, 10, 1)
    assert notes[-1].date is None or notes[-1].path != notes[0].path


def test_search_notes_empty_query_and_unknown_mode():
    assert _searcher().search_notes("  ") == []
    with pytest.raises(ValueError, match="mode"):
        _searcher().search_notes("kredi", mode="magic")


# ---- reading a whole note ----

def test_read_note_returns_file_text_and_refuses_unknown_keys(tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "kedi.md").write_text("# Kedi\nuyuyor", encoding="utf-8", newline="\n")
    (tmp_path / "gizli.md").write_text("gizli", encoding="utf-8")
    cfg = load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx"),
                       "RAG_RERANK_DIR": ""})
    chunks = [_chunk("kedi.md", 0, "kedi uyuyor")]
    save_index(cfg, IndexData(chunks=chunks, vectors=np.stack([ANIMALS]),
                              bm25=index_chunks(chunks), settings={}))
    s = Searcher.from_config(cfg, FakeEmbedder())
    assert s.read_note("kedi.md") == "# Kedi\nuyuyor"
    with pytest.raises(KeyError):
        s.read_note("../gizli.md")


def test_read_note_without_a_vault_resolver_fails_clearly():
    with pytest.raises(RuntimeError, match="vault"):
        _searcher().read_note("kedi.md")


# ---- reranking ----

class FakeReranker:
    """Likes passages containing 'hedef'; records what it was asked to score."""

    def __init__(self):
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [5.0 if "hedef" in p else -5.0 for p in passages]


def _rerank_searcher(**kw):
    chunks = [
        _chunk("a.md", 0, "kedi uyuyor"),  # 0 best by meaning for 'evcil hayvan'
        _chunk("b.md", 0, "kedi süt içiyor"),  # 1
        _chunk("c.md", 0, "hedef burada başka konu"),  # 2 meaning-wise far
        _chunk("d.md", 0, "alakasız"),  # 3
    ]
    vectors = np.stack([ANIMALS, ANIMALS * 0.9 + FINANCE * 0.1, VEHICLES, FINANCE])
    data = IndexData(chunks=chunks, vectors=vectors, bm25=index_chunks(chunks), settings={})
    rr = FakeReranker()
    return Searcher(data, FakeEmbedder(), reranker=rr, **kw), rr


def test_reranker_reorders_candidates_by_its_own_score():
    s, _ = _rerank_searcher()
    plain = s.search("evcil hayvan", k=4, rerank=False)
    assert plain[0].chunk.path == "a.md"
    reranked = s.search("evcil hayvan", k=4)
    assert reranked[0].chunk.path == "c.md"
    assert reranked[0].rerank_score is not None and plain[0].rerank_score is None


def test_reranker_sees_chunk_embed_text_and_only_top_candidates():
    s, rr = _rerank_searcher(rerank_top=2)
    s.search("evcil hayvan", k=4)
    query, passages = rr.calls[0]
    assert query == "evcil hayvan" and len(passages) == 2
    assert all(p.startswith("Note: ") for p in passages)  # embed_text, not bare text


def test_candidates_beyond_rerank_top_still_returned_after_reranked_ones():
    s, _ = _rerank_searcher(rerank_top=2)
    hits = s.search("evcil hayvan", k=4)
    assert len(hits) == 4
    assert [h.rerank_score is not None for h in hits] == [True, True, False, False]
    assert hits[1].score > hits[2].score


def test_rerank_can_be_turned_off_per_call_and_requires_a_reranker():
    s, rr = _rerank_searcher()
    s.search("evcil hayvan", rerank=False)
    assert rr.calls == []
    with pytest.raises(ValueError, match="reranker"):
        _searcher().search("evcil hayvan", rerank=True)
    assert _searcher().search("evcil hayvan")  # default without a reranker: just works


def test_search_notes_uses_reranked_order_and_recency_still_applies():
    s, _ = _rerank_searcher()
    assert s.search_notes("evcil hayvan", k=1)[0].path == "c.md"

