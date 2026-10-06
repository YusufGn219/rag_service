import json
from dataclasses import replace
from datetime import date

import pytest

from rag_service.audit import (
    ExclusionError,
    apply_env_exclusions,
    compare_snapshots,
    entry_for_key,
    eval_snapshot,
    format_report,
    load_decisions,
    mark_excluded,
    mark_kept,
    noise_id,
    plan_exclusion,
    save_decisions,
)
from rag_service.config import load_config
from rag_service.dupes import DupGroup, NoteInfo
from rag_service.evaluate import Question
from rag_service.noise import NoiseCandidate


def _info(path, **kw):
    base = dict(path=path, chunks=3, words=120, modified=date(2026, 5, 3), inbound=0, hints=(), covered=1.0)
    base.update(kw)
    return NoteInfo(**base)


def _group(key="abc123abc123", kind="exact", paths=("a/x.md", "b/x.md"), exclude=("b/x.md",), detail="birebir aynı metin"):
    return DupGroup(key=key, kind=kind, notes=tuple(_info(p) for p in paths), detail=detail,
                    keep=tuple(p for p in paths if p not in exclude) if exclude else (), exclude=tuple(exclude))


def _noise(path="n/big.md", strength="strong", flags=("long", "orphan"), rate=0.06, **kw):
    base = dict(path=path, appearances=60, rate=rate, flags=flags, inbound=0, outbound=0, words=7000,
                chunks=20, hints=(), strength=strength)
    base.update(kw)
    return NoiseCandidate(**base)


# ---- decisions ----

def test_decisions_start_empty_and_survive_a_round_trip(tmp_path):
    path = tmp_path / "data" / "kararlar.json"
    d = load_decisions(path)
    assert d == {"kept_groups": {}, "kept_notes": {}, "excluded": {}}
    mark_kept(d, group="abc", today=date(2026, 10, 8))
    mark_kept(d, note="n/big.md", today=date(2026, 10, 8))
    mark_excluded(d, ["b/x.md"], today=date(2026, 10, 8))
    save_decisions(path, d)
    assert load_decisions(path) == d
    assert json.loads(path.read_text(encoding="utf-8"))["kept_groups"] == {"abc": "2026-10-08"}


def test_broken_decisions_file_is_an_error_not_silently_empty(tmp_path):
    path = tmp_path / "k.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="k.json"):
        load_decisions(path)


# ---- report ----

def test_report_lists_groups_and_candidates_with_ids_and_commands():
    text, data = format_report([_group()], [_noise()], load_decisions_empty())
    assert "abc123abc123" in text and "a/x.md" in text and "b/x.md" in text
    assert "birebir aynı metin" in text
    assert noise_id("n/big.md") in text and "n/big.md" in text
    assert "audit keep" in text and "audit exclude" in text
    assert data["groups"]["abc123abc123"]["exclude"] == ["b/x.md"]
    assert data["noise"][noise_id("n/big.md")]["path"] == "n/big.md"


def load_decisions_empty():
    return {"kept_groups": {}, "kept_notes": {}, "excluded": {}}


def test_report_marks_which_note_is_suggested_to_leave():
    text, _ = format_report([_group()], [], load_decisions_empty())
    lines = {line.split()[1]: line.split()[0] for line in text.splitlines() if line.strip().startswith(("ÇIKAR", "kalsın"))}
    assert lines == {"b/x.md": "ÇIKAR", "a/x.md": "kalsın"}


def test_groups_without_a_suggestion_say_so():
    text, data = format_report([_group(kind="similar", exclude=(), detail="benzer")], [], load_decisions_empty())
    assert "öneri yok" in text
    assert data["groups"]["abc123abc123"]["exclude"] == []


def test_decided_items_are_hidden_and_counted():
    d = load_decisions_empty()
    mark_kept(d, group="abc123abc123", today=date(2026, 10, 8))
    mark_kept(d, note="n/big.md", today=date(2026, 10, 8))
    text, data = format_report([_group()], [_noise()], d)
    assert "abc123abc123" not in text and "n/big.md" not in text
    assert "1 kopya grubu" in text and "1 gürültü adayı" in text  # "decided, hidden" counters
    assert data["groups"] == {} and data["noise"] == {}


def test_already_excluded_notes_hide_their_group_and_candidate():
    d = load_decisions_empty()
    mark_excluded(d, ["b/x.md", "n/big.md"], today=date(2026, 10, 8))
    text, data = format_report([_group()], [_noise()], d)
    assert data["groups"] == {} and data["noise"] == {}


def test_look_candidates_are_limited_but_strong_are_all_shown():
    strong = [_noise(f"s{i}.md") for i in range(5)]
    look = [_noise(f"l{i}.md", strength="look", flags=(), rate=0.02 - i / 10000) for i in range(30)]
    text, data = format_report([], strong + look, load_decisions_empty(), limit_look=10)
    assert all(f"s{i}.md" in text for i in range(5))
    assert "l9.md" in text and "l10.md" not in text
    assert "+20 daha" in text and "--all" in text
    text, _ = format_report([], strong + look, load_decisions_empty(), limit_look=10, show_all=True)
    assert "l29.md" in text


def test_empty_report_says_nothing_found():
    text, data = format_report([], [], load_decisions_empty())
    assert "bulunamadı" in text.lower() or "yok" in text.lower()
    assert data == {"groups": {}, "noise": {}}


def test_noise_id_is_stable_and_short():
    assert noise_id("a.md") == noise_id("a.md") != noise_id("b.md")
    assert noise_id("a.md").startswith("n") and len(noise_id("a.md")) == 9


# ---- exclusions ----

def _vault(tmp_path, files, roots=("vault",)):
    made = []
    for root in roots:
        r = tmp_path / root
        for f in files:
            p = r / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# n\n\nmetin", encoding="utf-8")
        made.append(str(r))
    return made


def _cfg(tmp_path, roots, exclude=None):
    env = {"RAG_VAULT_ROOT": ";".join(roots) if len(roots) > 1 else roots[0],
           "RAG_INDEX_DIR": str(tmp_path / "idx")}
    if exclude is not None:
        env["RAG_EXCLUDE_DIRS"] = exclude
    return load_config(env)


def test_entry_for_key_single_root_is_the_key(tmp_path):
    cfg = _cfg(tmp_path, _vault(tmp_path, ["a/x.md"]))
    assert entry_for_key(cfg, "a/x.md") == "a/x.md"


def test_entry_for_key_strips_the_root_name_with_several_roots(tmp_path):
    cfg = _cfg(tmp_path, _vault(tmp_path, ["a/x.md"], roots=("r1", "r2")))
    assert entry_for_key(cfg, "r2/a/x.md") == "a/x.md"


def test_top_level_notes_and_commas_cannot_be_excluded_safely(tmp_path):
    cfg = _cfg(tmp_path, _vault(tmp_path, ["top.md", "a/b,c.md"]))
    with pytest.raises(ExclusionError, match="her yerde"):
        entry_for_key(cfg, "top.md")
    with pytest.raises(ExclusionError, match="virgül"):
        entry_for_key(cfg, "a/b,c.md")


def test_plan_removes_exactly_the_chosen_notes(tmp_path):
    cfg = _cfg(tmp_path, _vault(tmp_path, ["a/x.md", "b/x.md", "b/y.md"]))
    plan = plan_exclusion(cfg, ["b/x.md"])
    assert plan.entries == ["b/x.md"]
    assert plan.removed == ["b/x.md"] and plan.extra == []


def test_plan_reports_other_notes_the_entry_would_also_remove(tmp_path):
    cfg = _cfg(tmp_path, _vault(tmp_path, ["a/x.md"], roots=("r1", "r2")))
    plan = plan_exclusion(cfg, ["r1/a/x.md"])
    assert plan.removed == ["r1/a/x.md", "r2/a/x.md"]
    assert plan.extra == ["r2/a/x.md"]


def test_plan_ignores_notes_that_are_already_excluded(tmp_path):
    cfg = _cfg(tmp_path, _vault(tmp_path, ["a/x.md", "b/x.md"]), exclude="b/x.md")
    plan = plan_exclusion(cfg, ["b/x.md"])
    assert plan.entries == [] and plan.removed == []


def test_env_line_is_extended_and_the_rest_kept(tmp_path):
    env = tmp_path / ".env"
    env.write_text("RAG_VAULT_ROOT=/v\nRAG_EXCLUDE_DIRS=.git,.obsidian\nOTHER=1\n", encoding="utf-8")
    cfg = _cfg(tmp_path, _vault(tmp_path, ["a/x.md"]), exclude=".git,.obsidian")
    apply_env_exclusions(env, cfg, ["a/x.md"])
    assert env.read_text(encoding="utf-8") == (
        "RAG_VAULT_ROOT=/v\nRAG_EXCLUDE_DIRS=.git,.obsidian,a/x.md\nOTHER=1\n")


def test_env_without_the_line_gets_one_with_the_defaults_kept(tmp_path):
    env = tmp_path / ".env"
    env.write_text("RAG_VAULT_ROOT=/v\n", encoding="utf-8")
    cfg = _cfg(tmp_path, _vault(tmp_path, ["a/x.md"]))
    apply_env_exclusions(env, cfg, ["a/x.md"])
    assert env.read_text(encoding="utf-8").splitlines()[-1] == "RAG_EXCLUDE_DIRS=.git,.obsidian,.trash,a/x.md"


def test_env_crlf_endings_are_kept_and_duplicates_not_added(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"RAG_VAULT_ROOT=/v\r\nRAG_EXCLUDE_DIRS=.git,a/x.md\r\n")
    cfg = _cfg(tmp_path, _vault(tmp_path, ["a/x.md"]), exclude=".git,a/x.md")
    apply_env_exclusions(env, cfg, ["a/x.md", "b/y.md"])
    assert env.read_bytes() == b"RAG_VAULT_ROOT=/v\r\nRAG_EXCLUDE_DIRS=.git,a/x.md,b/y.md\r\n"


def test_exclusions_from_the_real_environment_are_refused(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("RAG_VAULT_ROOT=/v\n", encoding="utf-8")
    monkeypatch.setenv("RAG_EXCLUDE_DIRS", ".git")
    cfg = _cfg(tmp_path, _vault(tmp_path, ["a/x.md"]))
    with pytest.raises(ExclusionError, match="ortam değişkeni"):
        apply_env_exclusions(env, cfg, ["a/x.md"])


# ---- before / after measurement ----

class _Hit:
    def __init__(self, path):
        self.path = path


class FakeSearcher:
    def __init__(self, answers):
        self.answers = answers

    def search_notes(self, text, k=10, **kw):
        return [_Hit(p) for p in self.answers[text]]


QS = [Question("q1", "soru bir", ("a.md",)), Question("q2", "soru iki", ("b.md",)),
      Question("q3", "soru üç", ("c/",)), Question("q4", "etiketsiz", ())]


def test_snapshot_has_metrics_and_per_question_ranks():
    s = FakeSearcher({"soru bir": ["a.md"], "soru iki": ["x.md", "b.md"], "soru üç": ["z.md"], "etiketsiz": []})
    snap = eval_snapshot(s, QS, notes=10)
    assert snap["n"] == 3 and snap["notes"] == 10
    assert snap["ranks"] == {"q1": 1, "q2": 2, "q3": None}
    assert snap["hit1"] == pytest.approx(1 / 3) and snap["hit5"] == pytest.approx(2 / 3)


def test_compare_shows_regressions_improvements_and_note_counts():
    old = {"n": 3, "notes": 100, "hit1": 2 / 3, "hit5": 1.0, "mrr": 0.8,
           "ranks": {"q1": 1, "q2": 1, "q3": 3}, "questions": {"q1": "soru bir", "q2": "soru iki", "q3": "soru üç"}}
    new = {"n": 3, "notes": 90, "hit1": 1 / 3, "hit5": 2 / 3, "mrr": 0.5,
           "ranks": {"q1": 1, "q2": None, "q3": 1}, "questions": old["questions"]}
    text = compare_snapshots(old, new)
    assert "100" in text and "90" in text  # indexed notes before / after
    worse, _, better = text.partition("İYİLEŞEN")
    assert "KÖTÜLEŞEN" in worse and "q2" in worse and "soru iki" in worse and "q3" not in worse
    assert "q3" in better and "q2" not in better
    assert "q1" not in text  # unchanged questions are not listed


def test_compare_with_no_change_says_so():
    snap = {"n": 1, "notes": 5, "hit1": 1.0, "hit5": 1.0, "mrr": 1.0, "ranks": {"q1": 1}, "questions": {"q1": "s"}}
    assert "değişmedi" in compare_snapshots(snap, dict(snap)).lower()
