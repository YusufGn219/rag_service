import math

import numpy as np

from rag_service.bm25 import BM25Index, index_chunks, tokenize
from rag_service.chunker import chunk_note


# ---- tokenizer ----

def test_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("  ... --- !!! ") == []


def test_turkish_case_and_diacritics_fold_together():
    assert tokenize("ISPARTA İstanbul ışık") == ["isparta", "istanbul", "isik"]
    assert tokenize("işlem") == tokenize("islem")
    assert tokenize("ÇAĞRI ŞÖYLE ÜÇ") == tokenize("cagri soyle uc")


def test_english_uppercase_matches_lowercase_query():
    assert tokenize("API") == tokenize("api") == ["api"]


def test_camel_case_keeps_whole_and_parts():
    assert tokenize("ConfigError") == ["configerror", "config", "error"]
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_snake_case_keeps_whole_and_parts():
    assert tokenize("RAG_VAULT_ROOT") == ["rag_vault_root", "rag", "vault", "root"]


def test_apostrophe_suffix_dropped():
    assert tokenize("Konsey'in notları") == ["konsey", "notlari"]
    assert tokenize("Claude’s") == ["claude"]


def test_single_letters_dropped_digits_kept():
    assert tokenize("a v2 8b x") == ["v2", "8b"]


def test_different_words_stay_different():
    # no stemming/prefix truncation: araba and arabesk must never share a token
    assert tokenize("araba") == ["araba"]
    assert tokenize("arabesk") == ["arabesk"]
    assert set(tokenize("araba")).isdisjoint(tokenize("arabesk"))


def test_file_name_split_on_dots_and_dashes():
    assert tokenize("Sabitler.md") == ["sabitler", "md"]
    assert tokenize("ar-kiyafet") == ["ar", "kiyafet"]


# ---- BM25 ----

def _idx(*docs, **kw):
    return BM25Index([d.split() for d in docs], **kw)


def test_score_matches_hand_computed_formula():
    idx = BM25Index([["a", "b"], ["a"]], k1=1.5, b=0.75)
    scores = idx.scores(["a"])
    idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    avgdl = 1.5

    def expected(dl):
        return idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

    assert np.isclose(scores[0], expected(2))
    assert np.isclose(scores[1], expected(1))


def test_rare_term_outweighs_common_term():
    idx = _idx("ortak sey", "ortak baska", "ortak nadir", "ortak diger")
    top = idx.search(["ortak", "nadir"], k=4)
    assert top[0][0] == 2


def test_shorter_doc_wins_for_same_term_count():
    idx = _idx("kedi", "kedi " + " ".join(f"x{i}" for i in range(30)))
    top = idx.search(["kedi"])
    assert [d for d, _ in top] == [0, 1]


def test_more_occurrences_score_higher():
    idx = _idx("kedi kedi kedi bos bos", "kedi bos bos bos bos")
    top = idx.search(["kedi"])
    assert top[0][0] == 0


def test_no_match_returns_nothing():
    idx = _idx("kedi koyun", "inek")
    assert idx.search(["yok"]) == []
    assert idx.search([]) == []
    assert not idx.scores(["yok"]).any()


def test_unknown_query_terms_ignored_when_mixed():
    idx = _idx("kedi koyun", "inek")
    assert [d for d, _ in idx.search(["kedi", "yok"])] == [0]


def test_search_limits_to_k_and_is_sorted_with_stable_ties():
    idx = _idx("a x", "a y", "a z", "a w", "q")
    top = idx.search(["a"], k=3)
    assert len(top) == 3
    assert [d for d, _ in top] == [0, 1, 2]  # equal scores -> lower doc id first
    assert all(top[i][1] >= top[i + 1][1] for i in range(2))


def test_repeated_query_term_counts_once():
    idx = _idx("kedi koyun", "koyun")
    assert idx.scores(["kedi", "kedi"])[0] == idx.scores(["kedi"])[0]


def test_empty_corpus():
    idx = BM25Index([])
    assert idx.search(["a"]) == []


# ---- chunk integration ----

def test_index_chunks_uses_embed_text_so_tags_and_headings_are_searchable():
    chunks = chunk_note("n.md", "---\ntags: [mimari]\n---\n# Gizli Baslik\n" + "kelime " * 60)
    chunks += chunk_note("m.md", "baska bir not " * 30)
    idx = index_chunks(chunks)
    assert idx.search(tokenize("mimari"))[0][0] == 0
    assert idx.search(tokenize("gizli baslik"))[0][0] == 0
