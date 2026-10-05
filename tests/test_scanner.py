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
