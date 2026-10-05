import numpy as np
import pytest

from rag_service.bm25 import index_chunks
from rag_service.chunker import Chunk
from rag_service.config import load_config
from rag_service.search import Searcher, rrf
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
    cfg = load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx")})
    with pytest.raises(FileNotFoundError, match="indexer"):
        Searcher.from_config(cfg, FakeEmbedder())
    chunks = [_chunk("kedi.md", 0, "kedi uyuyor")]
    save_index(cfg, IndexData(chunks=chunks, vectors=np.stack([ANIMALS]),
                              bm25=index_chunks(chunks), settings={}))
    assert Searcher.from_config(cfg, FakeEmbedder()).search("evcil hayvan")[0].chunk.path == "kedi.md"
