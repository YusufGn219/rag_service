import random
from collections import Counter

import numpy as np

from rag_service.bm25 import index_chunks
from rag_service.chunker import Chunk
from rag_service.noise import Query, count_appearances, find_noise, make_queries
from rag_service.store import IndexData


def _words(seed: int, n: int) -> str:
    rng = random.Random(seed)
    return " ".join(f"w{rng.randrange(10**9)}" for _ in range(n))


def _data(notes: dict[str, str], *, headings: set[str] = frozenset(), chunk_words: int = 60) -> IndexData:
    chunks = []
    for path, text in notes.items():
        words = text.split()
        for i, start in enumerate(range(0, max(len(words), 1), chunk_words)):
            chunks.append(Chunk(path=path, index=i, heading_path="Baslik" if path in headings else "",
                                text=" ".join(words[start:start + chunk_words]), tags=()))
    return IndexData(chunks=chunks, vectors=np.zeros((len(chunks), 1), np.float32),
                     bm25=index_chunks(chunks), settings={})


# ---- make_queries ----

def test_one_title_query_per_note_with_date_removed():
    data = _data({"a/Plan (2026-09-25).md": _words(1, 100), "b/Not.md": _words(2, 100)})
    titles = {q.text: q.exclude for q in make_queries(data, snippets_per_note=0)}
    assert titles == {"Plan": frozenset({"a/Plan (2026-09-25).md"}), "Not": frozenset({"b/Not.md"})}


def test_snippets_are_windows_of_the_notes_own_text():
    text = _words(3, 200)
    data = _data({"a.md": text})
    snippets = [q for q in make_queries(data, snippets_per_note=3, snippet_words=12) if q.text != "a"]
    assert len(snippets) == 3
    for q in snippets:
        assert len(q.text.split()) == 12 and q.text in text
        assert q.exclude == frozenset({"a.md"})


def test_snippets_are_deterministic_and_seed_dependent():
    data = _data({f"n{i}.md": _words(i, 150) for i in range(5)})
    one = [q.text for q in make_queries(data, seed=1)]
    assert one == [q.text for q in make_queries(data, seed=1)]
    assert one != [q.text for q in make_queries(data, seed=2)]


def test_tiny_chunks_give_no_snippet():
    data = _data({"a.md": "sadece birkac kelime"})
    assert [q.text for q in make_queries(data)] == ["a"]


def test_extra_questions_are_added_with_their_expected_notes():
    data = _data({"a.md": _words(4, 100)})
    extra = [("bir soru", {"a.md", "b/"})]
    qs = make_queries(data, snippets_per_note=0, extra=extra)
    assert Query("bir soru", frozenset({"a.md", "b/"})) in qs


# ---- count_appearances ----

def test_counts_other_notes_in_the_top_k_only():
    answers = {"q1": ["x.md", "own.md", "y.md"], "q2": ["x.md", "y.md", "z.md"]}
    queries = [Query("q1", frozenset({"own.md"})), Query("q2", frozenset())]
    counts = count_appearances(lambda q: answers[q], queries, k=2)
    assert counts == Counter({"x.md": 2, "y.md": 1})  # own.md skipped; z.md is beyond k


def test_expected_folder_entries_are_not_counted_against_a_note():
    queries = [Query("q", frozenset({"Chef/"}))]
    counts = count_appearances(lambda q: ["Chef/a.md", "other.md"], queries, k=5)
    assert counts == Counter({"other.md": 1})


# ---- find_noise ----

def _noise_setup():
    notes = {f"n{i}.md": _words(i, 80) for i in range(30)}
    notes["hub_long.md"] = _words(100, 3500)  # long, no links, no headings
    notes["hub_ok.md"] = _words(101, 400) + " [[n1]] [[n2]]"
    notes["linker.md"] = "[[hub_ok]] [[hub_ok]] " + _words(102, 80)
    data = _data(notes, headings={"hub_ok.md"})
    counts = Counter({f"n{i}.md": 3 for i in range(30)})
    counts["hub_long.md"] = 60
    counts["hub_ok.md"] = 60
    return data, counts


def test_hubs_are_found_and_split_by_evidence():
    data, counts = _noise_setup()
    found = {c.path: c for c in find_noise(data, counts, n_queries=1000)}
    assert set(found) == {"hub_long.md", "hub_ok.md"}
    assert found["hub_long.md"].strength == "strong"
    assert set(found["hub_long.md"].flags) == {"long", "orphan", "unstructured"}
    assert found["hub_ok.md"].strength == "look"  # linked to, has headings, not long
    assert found["hub_ok.md"].flags == ()
    assert found["hub_ok.md"].inbound == 1


def test_strong_comes_first_then_by_rate():
    data, counts = _noise_setup()
    found = find_noise(data, counts, n_queries=1000)
    assert [c.path for c in found] == ["hub_long.md", "hub_ok.md"]
    assert found[0].appearances == 60 and found[0].rate == 0.06


def test_a_note_that_is_often_returned_but_alone_is_not_a_hub_when_everyone_is():
    data, _ = _noise_setup()
    counts = Counter({p: 40 for p in {c.path for c in data.chunks}})
    assert find_noise(data, counts, n_queries=1000) == []


def test_rarely_returned_notes_are_never_listed():
    data, _ = _noise_setup()
    counts = Counter({"hub_long.md": 3, "n1.md": 1})
    assert find_noise(data, counts, n_queries=1000) == []


def test_no_counts_no_noise():
    data, _ = _noise_setup()
    assert find_noise(data, Counter(), n_queries=100) == []
    assert find_noise(data, Counter({"a.md": 5}), n_queries=0) == []


def test_flag_thresholds_are_adjustable():
    data, counts = _noise_setup()
    found = {c.path: c for c in find_noise(data, counts, n_queries=1000, long_words=10_000)}
    assert "long" not in found["hub_long.md"].flags
    assert found["hub_long.md"].strength == "strong"  # still orphan + unstructured


def test_path_hints_are_reported():
    notes = {f"n{i}.md": _words(i, 80) for i in range(30)}
    notes["proje-eski/ana.md"] = _words(200, 100)
    data = _data(notes)
    counts = Counter({f"n{i}.md": 3 for i in range(30)})
    counts["proje-eski/ana.md"] = 50
    found = find_noise(data, counts, n_queries=1000)
    assert found[0].path == "proje-eski/ana.md" and found[0].hints == ("eski",)


def test_within_a_kind_more_flags_come_first_then_higher_rate():
    notes = {f"n{i}.md": _words(i, 80) for i in range(30)}
    notes["one_flag_often.md"] = _words(300, 100) + " [[n1]]"  # unstructured only
    notes["two_flags_less.md"] = _words(301, 100)  # orphan + unstructured
    data = _data(notes)
    counts = Counter({f"n{i}.md": 3 for i in range(30)})
    counts["one_flag_often.md"] = 90
    counts["two_flags_less.md"] = 40
    assert [c.path for c in find_noise(data, counts, n_queries=1000)] == ["two_flags_less.md", "one_flag_often.md"]
