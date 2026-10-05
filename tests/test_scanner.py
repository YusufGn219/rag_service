from rag_service.config import load_config
from rag_service.scanner import scan_notes


def _touch(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")


def _scan(root, **extra):
    return scan_notes(load_config({"RAG_VAULT_ROOT": str(root), **extra}))


def test_finds_md_recursively_sorted(tmp_path):
    _touch(tmp_path, "b.md")
    _touch(tmp_path, "vault1/a.md")
    _touch(tmp_path, "vault1/sub/c.md")
    assert _scan(tmp_path) == [
        tmp_path.resolve() / "b.md",
        tmp_path.resolve() / "vault1" / "a.md",
        tmp_path.resolve() / "vault1" / "sub" / "c.md",
    ]


def test_ignores_non_markdown(tmp_path):
    _touch(tmp_path, "a.md")
    _touch(tmp_path, "b.txt")
    _touch(tmp_path, "c.png")
    assert [p.name for p in _scan(tmp_path)] == ["a.md"]


def test_extension_case_insensitive(tmp_path):
    _touch(tmp_path, "a.MD")
    assert [p.name for p in _scan(tmp_path)] == ["a.MD"]


def test_default_excludes_skipped_at_any_depth(tmp_path):
    _touch(tmp_path, "keep.md")
    _touch(tmp_path, ".git/x.md")
    _touch(tmp_path, "v/.obsidian/y.md")
    _touch(tmp_path, "v/.trash/z.md")
    assert [p.name for p in _scan(tmp_path)] == ["keep.md"]


def test_custom_excludes_replace_defaults(tmp_path):
    _touch(tmp_path, "private/s.md")
    _touch(tmp_path, ".obsidian/o.md")
    names = [p.name for p in _scan(tmp_path, RAG_EXCLUDE_DIRS="private")]
    assert names == ["o.md"]


def test_empty_vault(tmp_path):
    assert _scan(tmp_path) == []


def test_exclude_entry_with_slash_is_relative_path(tmp_path):
    _touch(tmp_path, "Konsey/Medusa/logs/a.md")
    _touch(tmp_path, "Proje/logs/b.md")
    names = [p.name for p in _scan(tmp_path, RAG_EXCLUDE_DIRS="Konsey/Medusa/logs")]
    assert names == ["b.md"]


def test_exclude_single_file_by_relative_path(tmp_path):
    _touch(tmp_path, "Medusa/alarmlar.md")
    _touch(tmp_path, "Medusa/baska.md")
    _touch(tmp_path, "Diger/alarmlar.md")
    found = _scan(tmp_path, RAG_EXCLUDE_DIRS="Medusa/alarmlar.md")
    assert sorted(p.relative_to(tmp_path.resolve()).as_posix() for p in found) == [
        "Diger/alarmlar.md",
        "Medusa/baska.md",
    ]


def test_exclude_bare_name_matches_file_anywhere(tmp_path):
    _touch(tmp_path, "a/draft.md")
    _touch(tmp_path, "b/keep.md")
    assert [p.name for p in _scan(tmp_path, RAG_EXCLUDE_DIRS="draft.md")] == ["keep.md"]


def test_exclude_path_tolerates_backslash_and_edge_slashes(tmp_path):
    _touch(tmp_path, "A/B/x.md")
    _touch(tmp_path, "keep.md")
    assert [p.name for p in _scan(tmp_path, RAG_EXCLUDE_DIRS="/A\\B/")] == ["keep.md"]


def test_scans_every_root_and_applies_excludes_per_root(tmp_path):
    import os

    a, b = tmp_path / "a", tmp_path / "b"
    _touch(a, "one.md")
    _touch(a, "logs/skip.md")
    _touch(b, "two.md")
    _touch(b, "logs/skip2.md")
    _touch(b, ".obsidian/x.md")
    cfg = load_config({"RAG_VAULT_ROOT": os.pathsep.join([str(a), str(b)]),
                       "RAG_INDEX_DIR": str(tmp_path / "idx"), "RAG_EXCLUDE_DIRS": ".obsidian,logs"})
    assert sorted(p.name for p in scan_notes(cfg)) == ["one.md", "two.md"]
