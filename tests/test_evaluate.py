import json
from types import SimpleNamespace

import pytest

from rag_service.evaluate import (
    Question,
    add_question,
    format_report,
    load_questions,
    matches,
    run_eval,
)


class FakeSearcher:
    """search_notes returns canned note paths per question text."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def search_notes(self, query, k=5, **kw):
        self.calls.append((query, k, kw))
        return [SimpleNamespace(path=p) for p in self.answers.get(query, [])][:k]


# ---- matching ----

def test_matches_exact_note_and_folder_prefix():
    assert matches("a/b.md", ("a/b.md",))
    assert not matches("a/b.md", ("a/c.md",))
    assert matches("Staj/Hafta 1/g.md", ("Staj/",))
    assert not matches("Stajyer/g.md", ("Staj/",))
    assert matches("x.md", ("y.md", "x.md"))


# ---- loading ----

def test_load_questions_reads_jsonl_and_skips_blank_lines(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(
        json.dumps({"question": "bir", "expected": ["a.md"]}, ensure_ascii=False) + "\n\n"
        + json.dumps({"id": "x", "question": "iki", "expected": "b.md"}) + "\n"
        + json.dumps({"question": "üç"}) + "\n",
        encoding="utf-8",
    )
    qs = load_questions(f)
    assert [q.text for q in qs] == ["bir", "iki", "üç"]
    assert qs[0].expected == ("a.md",) and qs[1].expected == ("b.md",) and qs[2].expected == ()
    assert qs[1].id == "x" and qs[0].id == "q1"


def test_load_questions_reports_bad_line_number(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text('{"question": "ok"}\n{bozuk\n', encoding="utf-8")
    with pytest.raises(ValueError, match="2"):
        load_questions(f)


def test_load_questions_requires_question_text(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text('{"expected": ["a.md"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_questions(f)


def test_add_question_appends_and_round_trips_turkish_text(tmp_path):
    f = tmp_path / "sub" / "q.jsonl"
    add_question(f, "Tryess neredeyiz?", ["Claude_Code/ar-kiyafet/"])
    add_question(f, "ikinci", [])
    qs = load_questions(f)
    assert [q.text for q in qs] == ["Tryess neredeyiz?", "ikinci"]
    assert qs[0].expected == ("Claude_Code/ar-kiyafet/",)
    assert "Tryess neredeyiz?" in f.read_text(encoding="utf-8")  # not \u-escaped


# ---- running ----

def _qs():
    return [
        Question("q1", "a", ("n1.md",)),
        Question("q2", "b", ("n2.md",)),
        Question("q3", "c", ("n3.md",)),
        Question("q4", "d", ()),  # unlabeled
    ]


def test_ranks_and_metrics():
    s = FakeSearcher({
        "a": ["n1.md", "x.md"],  # rank 1
        "b": ["x.md", "y.md", "n2.md"],  # rank 3
        "c": ["x.md", "y.md"],  # miss
        "d": ["z.md"],
    })
    rep = run_eval(s, _qs(), k=10)
    assert [r.rank for r in rep.results] == [1, 3, None]
    assert rep.n == 3
    assert rep.hit1 == pytest.approx(1 / 3)
    assert rep.hit5 == pytest.approx(2 / 3)
    assert rep.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert [q.id for q in rep.unlabeled] == ["q4"]


def test_hit5_excludes_rank_six_and_k_is_passed_through():
    s = FakeSearcher({"a": ["x1", "x2", "x3", "x4", "x5", "n1.md"]})
    rep = run_eval(s, [Question("q1", "a", ("n1.md",))], k=10, mode="dense", recency=False)
    assert rep.results[0].rank == 6 and rep.hit5 == 0.0
    assert s.calls[0] == ("a", 10, {"mode": "dense", "recency": False})


def test_empty_report_has_zero_metrics():
    rep = run_eval(FakeSearcher({}), [Question("q", "a", ())])
    assert rep.n == 0 and rep.hit1 == rep.hit5 == rep.mrr == 0.0


def test_format_report_lists_misses_and_summary():
    s = FakeSearcher({"a": ["n1.md"], "b": ["x.md", "y.md"], "c": []})
    text = format_report(run_eval(s, _qs()))
    assert "hit@1" in text and "hit@5" in text and "MRR" in text
    assert "q2" in text and "x.md" in text  # a miss shows what came back instead
    assert "q4" in text  # unlabeled questions are mentioned
