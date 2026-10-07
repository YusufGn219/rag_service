"""'Where are we' questions: the note that sums a project up (index / Devam) should come first."""
from datetime import date

import numpy as np
import pytest

from rag_service.bm25 import index_chunks
from rag_service.chunker import Chunk
from rag_service.search import Searcher
from rag_service.store import IndexData

TODAY = date(2026, 10, 7)


class FakeEmbedder:
    def embed_query(self, text):
        return np.ones(3, dtype=np.float32) / np.sqrt(3)


def _searcher(notes: dict[str, str], **kw) -> Searcher:
    chunks = [Chunk(path=p, index=0, heading_path="", text=t, tags=()) for p, t in notes.items()]
    data = IndexData(chunks=chunks, vectors=np.zeros((len(chunks), 3), np.float32),
                     bm25=index_chunks(chunks), settings={})
    return Searcher(data, FakeEmbedder(), today=TODAY, **kw)


def _top(s, query, k=3):
    return [n.path for n in s.search_notes(query, k=k, mode="bm25")]


# One project whose detail note matches the word "alfa" better than its index note does, so
# without any bonus the detail note is first.
BASIC = {
    "P/alfa/alfa - İndeks.md": "alfa ozet notu",
    "P/alfa/02 Ayrinti.md": "alfa alfa alfa alfa ayrinti",
    "P/gama/03 Baska.md": "gama baska",
}


def test_without_a_state_word_the_better_matching_note_wins():
    assert _top(_searcher(BASIC), "alfa")[0] == "P/alfa/02 Ayrinti.md"


def test_a_state_question_puts_the_index_note_first():
    assert _top(_searcher(BASIC), "alfa neredeyiz")[0] == "P/alfa/alfa - İndeks.md"
    assert _top(_searcher(BASIC), "alfa güncel durum")[0] == "P/alfa/alfa - İndeks.md"
    assert _top(_searcher(BASIC), "alfa en son ne yaptık")[0] == "P/alfa/alfa - İndeks.md"


def test_a_devam_note_counts_as_a_status_note():
    notes = {"P/alfa/05 Devam Adımları.md": "alfa notlar", "P/alfa/02 Ayrinti.md": "alfa alfa alfa alfa ayrinti"}
    assert _top(_searcher(notes), "alfa neredeyiz")[0] == "P/alfa/05 Devam Adımları.md"


def test_a_recency_word_that_is_not_a_state_question_does_not_boost_index_notes():
    s = _searcher(BASIC)
    for q in ["son olarak alfa", "alfa yeni", "alfa acil", "güncel alfa", "en son alfa"]:
        assert _top(s, q)[0] == "P/alfa/02 Ayrinti.md", q


def test_recency_can_be_switched_off_even_for_a_state_question():
    s = _searcher(BASIC)
    notes = s.search_notes("alfa neredeyiz", k=3, mode="bm25", recency=False)
    assert notes[0].path == "P/alfa/02 Ayrinti.md"


def test_forcing_recency_on_does_not_make_a_plain_question_a_state_question():
    notes = _searcher(BASIC).search_notes("alfa", k=3, mode="bm25", recency=True)
    assert notes[0].path == "P/alfa/02 Ayrinti.md"


def test_the_bonus_is_adjustable_and_can_be_zero():
    off = _searcher(BASIC, status_weight=0.0, project_weight=0.0)
    assert _top(off, "alfa neredeyiz")[0] == "P/alfa/02 Ayrinti.md"
    on = _searcher(BASIC, status_weight=5.0, project_weight=0.0)
    assert _top(on, "alfa neredeyiz")[0] == "P/alfa/alfa - İndeks.md"


def test_questions_without_recency_words_are_unchanged_by_the_new_bonuses():
    with_bonus = _searcher(BASIC)
    without = _searcher(BASIC, status_weight=0.0, project_weight=0.0, recency_weight=0.0)
    for q in ["alfa", "alfa ayrinti", "gama baska", "alfa ozet"]:
        a = [(h.chunk.path, round(h.score, 9)) for h in with_bonus.search(q, k=5, mode="bm25")]
        b = [(h.chunk.path, round(h.score, 9)) for h in without.search(q, k=5, mode="bm25")]
        assert a == b, q


def test_the_search_notes_result_keeps_its_date_and_shape():
    notes = {"P/alfa/alfa - İndeks.md": "alfa ozet notu", "P/alfa/Not (2026-10-01).md": "alfa alfa alfa"}
    top = _searcher(notes).search_notes("alfa neredeyiz", k=2, mode="bm25")
    assert [n.path for n in top][0] == "P/alfa/alfa - İndeks.md"
    assert next(n for n in top if n.path.endswith("(2026-10-01).md")).date == date(2026, 10, 1)


# ---- the project folder the best results point to ----

def _two_projects() -> dict[str, str]:
    """Two projects whose index notes match "alfa" exactly equally well; on a tie the note listed
    later ranks higher, so the "abc" project wins unless the project bonus says otherwise. The
    other 'alfa' notes dominate the results."""
    notes = {
        "P/alfa/alfa - İndeks.md": "alfa alfa ozet",
        "P/abc/abc - İndeks.md": "alfa alfa ozet",
        "P/alfa/01 Bir.md": "alfa alfa alfa alfa alfa bir",
        "P/alfa/02 Iki.md": "alfa alfa alfa alfa alfa iki",
        "P/alfa/03 Uc.md": "alfa alfa alfa alfa alfa uc",
    }
    for i in range(40):  # a bigger vault: the "alfa" folder is rare in it, "P" is everywhere
        notes[f"P/dolgu{i}/not.md"] = f"dolgu{i} baska kelime"
    return notes


def _best_after_boost(searcher: Searcher, state: bool) -> str:
    """The path with the highest score after the bonuses, for hand-made base scores: the other
    project's index note scores slightly higher than the alfa index note, and the alfa notes
    fill the top of the results."""
    row = {c.path: i for i, c in enumerate(searcher._data.chunks)}
    scored = [(row["P/alfa/01 Bir.md"], 0.0164), (row["P/alfa/02 Iki.md"], 0.0161),
              (row["P/alfa/03 Uc.md"], 0.0159), (row["P/abc/abc - İndeks.md"], 0.0156),
              (row["P/alfa/alfa - İndeks.md"], 0.0154)]
    best_row, _ = max(searcher._boost(scored, state=state), key=lambda ds: ds[1])
    return searcher._data.chunks[best_row].path


def test_the_project_the_results_point_to_decides_between_two_index_notes():
    notes = _two_projects()
    assert _best_after_boost(_searcher(notes, project_weight=0.0), state=True) == "P/abc/abc - İndeks.md"
    assert _best_after_boost(_searcher(notes), state=True) == "P/alfa/alfa - İndeks.md"


def test_without_a_state_question_the_boost_does_not_pick_any_index_note():
    assert _best_after_boost(_searcher(_two_projects()), state=False) == "P/alfa/01 Bir.md"


def test_the_project_bonus_only_applies_to_state_questions():
    notes = _two_projects()
    # a plain question: no index bonus, no project bonus; the 'alfa' detail notes win
    assert _top(_searcher(notes), "alfa", k=1)[0].startswith("P/alfa/0")


def test_focus_unit_names_the_folder_the_results_cluster_in():
    s = _searcher(_two_projects())
    ranked = s._ranked("alfa", "bm25", None)
    assert s._focus_unit(ranked.scored) == "alfa"


def test_no_folder_stands_out_when_the_results_are_spread():
    notes = {f"P/k{i}/not.md": "alfa ortak kelime" for i in range(10)}
    s = _searcher(notes)
    assert s._focus_unit(s._ranked("alfa", "bm25", None).scored) is None


def test_a_folder_everything_lives_under_is_never_the_focus():
    s = _searcher(_two_projects())
    assert s._focus_unit(s._ranked("alfa", "bm25", None).scored) != "p"


@pytest.mark.parametrize("weight", [0.0, 0.5, 2.0])
def test_a_vault_with_no_notes_matching_returns_nothing_for_a_state_question(weight):
    s = _searcher(BASIC, project_weight=weight)
    assert s.search_notes("zzzz neredeyiz", k=3, mode="bm25") == []
