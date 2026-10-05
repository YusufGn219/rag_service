import numpy as np

from rag_service.bm25 import BM25Index, index_chunks, tokenize
from rag_service.chunker import Chunk
from rag_service.config import load_config
from rag_service.store import IndexData, index_path, load_index, save_index


def _cfg(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    return load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx")})


def _data():
    chunks = [
        Chunk("a.md", 0, "Ana > Alt", "kedi uyuyor şişman", ("hayvan", "ev")),
        Chunk("b.md", 0, "", "araba bozuldu", ()),
        Chunk("b.md", 1, "Son", "kedi ve köpek", ("hayvan",)),
    ]
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    return IndexData(chunks=chunks, vectors=vectors, bm25=index_chunks(chunks), settings={"model": "m", "n": 1})


def test_missing_index_loads_as_none(tmp_path):
    assert load_index(_cfg(tmp_path)) is None


def test_roundtrip_preserves_chunks_vectors_settings(tmp_path):
    cfg = _cfg(tmp_path)
    save_index(cfg, _data())
    back = load_index(cfg)
    assert back.chunks == _data().chunks
    assert np.array_equal(back.vectors, _data().vectors)
    assert back.vectors.dtype == np.float32
    assert back.settings == {"model": "m", "n": 1}


def test_roundtrip_bm25_gives_identical_results(tmp_path):
    cfg = _cfg(tmp_path)
    original = _data()
    save_index(cfg, original)
    back = load_index(cfg)
    for q in ["kedi", "araba bozuldu", "hayvan", "yok boyle bir sey", "ana alt"]:
        assert back.bm25.search(tokenize(q)) == original.bm25.search(tokenize(q))
        assert np.allclose(back.bm25.scores(tokenize(q)), original.bm25.scores(tokenize(q)))


def test_bm25_array_roundtrip_standalone():
    idx = BM25Index([["a", "b", "b"], ["a"], []], k1=1.2, b=0.6)
    back = BM25Index.from_arrays(idx.to_arrays())
    assert back.k1 == 1.2 and back.b == 0.6 and back.n == 3
    assert np.allclose(back.scores(["a", "b"]), idx.scores(["a", "b"]))


def test_empty_index_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    empty = IndexData(chunks=[], vectors=np.zeros((0, 0), dtype=np.float32), bm25=BM25Index([]), settings={})
    save_index(cfg, empty)
    back = load_index(cfg)
    assert back.chunks == [] and back.vectors.shape[0] == 0
    assert back.bm25.search(["x"]) == []


def test_corrupt_file_loads_as_none(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.index_dir.mkdir(parents=True)
    index_path(cfg).write_bytes(b"this is not an npz file")
    assert load_index(cfg) is None


def test_save_leaves_no_temp_files_and_overwrites(tmp_path):
    cfg = _cfg(tmp_path)
    save_index(cfg, _data())
    save_index(cfg, _data())
    assert sorted(p.name for p in cfg.index_dir.iterdir()) == ["index.npz"]
