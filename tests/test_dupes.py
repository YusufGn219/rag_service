import random
from datetime import date

import numpy as np

from rag_service.bm25 import index_chunks
from rag_service.chunker import Chunk
from rag_service.dupes import find_duplicates
from rag_service.store import IndexData


def _words(seed: int, n: int) -> str:
    """n distinct-looking words, the same for the same seed."""
    rng = random.Random(seed)
    return " ".join(f"kelime{rng.randrange(10**9)}" for _ in range(n))


def _data(notes: dict[str, str], *, chunk_words: int = 60) -> IndexData:
    """Index data with each note split into chunks of about chunk_words words."""
    chunks = []
    for path, text in notes.items():
        words = text.split()
        for i, start in enumerate(range(0, max(len(words), 1), chunk_words)):
            chunks.append(Chunk(path=path, index=i, heading_path="",
                                text=" ".join(words[start:start + chunk_words]), tags=()))
    return IndexData(chunks=chunks, vectors=np.zeros((len(chunks), 1), np.float32),
                     bm25=index_chunks(chunks), settings={})


def _mtime(day: int) -> dict:
    return {"mtime_ns": (1_780_000_000 + day * 86_400) * 10**9, "size": 1}


def _paths(group):
    return sorted(n.path for n in group.notes)


def test_unrelated_notes_give_nothing():
    data = _data({f"n{i}.md": _words(i, 200) for i in range(5)})
    assert find_duplicates(data, {}) == []


def test_identical_notes_are_an_exact_group():
    text = _words(1, 200)
    groups = find_duplicates(_data({"a/x.md": text, "b/y.md": text, "c/z.md": _words(2, 200)}), {})
    assert len(groups) == 1
    assert groups[0].kind == "exact"
    assert _paths(groups[0]) == ["a/x.md", "b/y.md"]


def test_case_spacing_and_turkish_letters_do_not_hide_a_copy():
    a = "Bağlam ışık ÇOK önemli şey " * 20
    b = "baglam isik cok onemli sey\n\n" * 20
    groups = find_duplicates(_data({"a.md": a, "b.md": b}), {})
    assert len(groups) == 1 and groups[0].kind == "exact"


def test_copy_with_a_few_extra_words_is_an_overlap_group():
    base = _words(3, 200)
    groups = find_duplicates(_data({"a.md": base, "b.md": base + " " + _words(4, 12)}), {})
    assert len(groups) == 1
    assert groups[0].kind == "overlap"
    assert _paths(groups[0]) == ["a.md", "b.md"]


def test_a_note_made_of_other_notes_is_found_with_all_of_them():
    parts = {f"gun{i}.md": _words(10 + i, 150) for i in range(3)}
    notes = {**parts, "Kronoloji (Yedek).md": " ".join(parts.values()), "baska.md": _words(99, 150)}
    groups = find_duplicates(_data(notes), {})
    assert len(groups) == 1
    assert _paths(groups[0]) == sorted(["Kronoloji (Yedek).md", *parts])
    assert "baska.md" not in _paths(groups[0])


def test_backup_hint_decides_which_to_exclude():
    parts = {f"gun{i}.md": _words(10 + i, 150) for i in range(3)}
    notes = {**parts, "Kronoloji (Yedek).md": " ".join(parts.values())}
    g = find_duplicates(_data(notes), {})[0]
    assert g.exclude == ("Kronoloji (Yedek).md",)
    assert set(g.keep) == set(parts)


def test_inbound_links_decide_when_there_is_no_hint():
    parts = {f"gun{i}.md": _words(10 + i, 150) for i in range(3)}
    big = " ".join(parts.values())
    notes = {**parts, "Genel.md": big,
             "indeks.md": "[[gun0]] [[gun1]] [[gun2]] " + _words(50, 100)}
    g = find_duplicates(_data(notes), {})[0]
    assert g.exclude == ("Genel.md",)
    by_path = {n.path: n for n in g.notes}
    assert by_path["gun0.md"].inbound == 1 and by_path["Genel.md"].inbound == 0


def test_exact_pair_keeps_the_linked_one_then_the_newer():
    text = _words(5, 200)
    notes = {"a.md": text, "b.md": text, "link.md": "[[b]] " + _words(6, 100)}
    g = find_duplicates(_data(notes), {})[0]
    assert g.exclude == ("a.md",)

    notes = {"a.md": text, "b.md": text}
    g = find_duplicates(_data(notes), {"a.md": _mtime(1), "b.md": _mtime(20)})[0]
    assert g.exclude == ("a.md",)  # b is newer
    g = find_duplicates(_data(notes), {"a.md": _mtime(20), "b.md": _mtime(1)})[0]
    assert g.exclude == ("b.md",)


def test_three_identical_copies_exclude_two():
    text = _words(7, 200)
    g = find_duplicates(_data({"a.md": text, "b.md": text, "c.md": text}), {})[0]
    assert len(g.exclude) == 2 and len(g.notes) == 3


def test_shared_boilerplate_alone_is_not_a_duplicate():
    boiler = _words(8, 12)
    notes = {f"n{i}.md": boiler + " " + _words(100 + i, 200) for i in range(4)}
    assert find_duplicates(_data(notes), {}) == []


def test_short_notes_are_ignored():
    assert find_duplicates(_data({"a.md": "kisa not metni", "b.md": "kisa not metni"}), {}) == []


def test_names_that_differ_only_in_letters_and_share_text_are_a_name_group():
    a_words = _words(20, 100)
    b_words = " ".join(a_words.split()[:40]) + " " + _words(21, 60)  # about a third shared
    groups = find_duplicates(_data({"x/bağlam.md": a_words, "y/baglam.md": b_words}), {})
    assert len(groups) == 1 and groups[0].kind == "name"
    assert _paths(groups[0]) == ["x/bağlam.md", "y/baglam.md"]
    assert groups[0].exclude == ()


def test_names_that_differ_only_in_letters_but_share_nothing_are_not_a_group():
    notes = {"x/bağlam.md": _words(22, 100), "y/baglam.md": _words(23, 100)}
    assert find_duplicates(_data(notes), {}) == []


def test_mostly_similar_notes_are_a_similar_group_without_a_suggestion():
    a_words = _words(24, 200)
    b_words = " ".join(a_words.split()[:150]) + " " + _words(25, 50)  # about 73% shared
    groups = find_duplicates(_data({"a.md": a_words, "b.md": b_words}), {})
    assert len(groups) == 1 and groups[0].kind == "similar"
    assert groups[0].exclude == () and groups[0].keep == ()


def test_notes_already_in_a_strong_group_are_not_listed_again_as_similar():
    base = _words(26, 200)
    notes = {"a.md": base, "b.md": base, "c.md": " ".join(base.split()[:150]) + " " + _words(27, 50)}
    groups = find_duplicates(_data(notes), {})
    assert [g.kind for g in groups] == ["exact"]


def test_hint_note_is_excluded_even_when_only_one_direction_is_covered():
    a_words = _words(28, 200)
    current = a_words
    old = " ".join(a_words.split()[:180]) + " " + _words(29, 14)  # ~88%/~92% covered
    g = find_duplicates(_data({"proje/kararlar.md": current, "proje-eski/kararlar.md": old}), {})[0]
    assert g.exclude == ("proje-eski/kararlar.md",)


def test_same_name_in_different_folders_is_normal():
    notes = {"a/README.md": _words(30, 100), "b/README.md": _words(31, 100)}
    assert find_duplicates(_data(notes), {}) == []


def test_note_info_has_chunks_words_and_date():
    text = _words(9, 200)
    g = find_duplicates(_data({"a.md": text, "b.md": text}), {"a.md": _mtime(3)})[0]
    a = next(n for n in g.notes if n.path == "a.md")
    assert a.chunks == 4 and a.words == 200 and a.modified is not None
    b = next(n for n in g.notes if n.path == "b.md")
    assert b.modified is None


def test_group_key_is_stable_and_order_independent():
    text = _words(11, 200)
    k1 = find_duplicates(_data({"a.md": text, "b.md": text}), {})[0].key
    k2 = find_duplicates(_data({"b.md": text, "a.md": text}), {})[0].key
    other = find_duplicates(_data({"a.md": text, "c.md": text}), {})[0].key
    assert k1 == k2 and k1 != other


def test_thresholds_are_adjustable():
    a_words = _words(12, 200)
    shared = " ".join(a_words.split()[:160])  # b repeats 80% of a, then goes its own way
    data = _data({"a.md": a_words, "b.md": shared + " " + _words(13, 200)})
    assert [g.kind for g in find_duplicates(data, {}, threshold=0.9)] == ["similar"]
    assert [g.kind for g in find_duplicates(data, {}, threshold=0.7)] == ["overlap"]
    assert find_duplicates(data, {}, threshold=0.9, similar_threshold=0.9) == []
