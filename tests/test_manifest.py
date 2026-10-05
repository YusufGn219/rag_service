import json
import os

from rag_service.config import load_config
from rag_service.manifest import compute_changes, load_manifest, save_manifest, snapshot
from rag_service.scanner import scan_notes


def _setup(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    cfg = load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx")})
    return vault, cfg


def _write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_missing_manifest_is_empty(tmp_path):
    _, cfg = _setup(tmp_path)
    assert load_manifest(cfg) == {}


def test_save_then_load_roundtrip(tmp_path):
    _, cfg = _setup(tmp_path)
    state = {"a.md": {"mtime_ns": 1, "size": 2}}
    save_manifest(cfg, state)
    assert load_manifest(cfg) == state


def test_corrupt_manifest_treated_as_empty(tmp_path):
    _, cfg = _setup(tmp_path)
    cfg.index_dir.mkdir(parents=True)
    (cfg.index_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    assert load_manifest(cfg) == {}


def test_snapshot_uses_posix_relative_keys(tmp_path):
    vault, cfg = _setup(tmp_path)
    _write(vault / "v1" / "a.md", "hello")
    snap = snapshot(cfg, scan_notes(cfg))
    assert list(snap) == ["v1/a.md"]
    assert snap["v1/a.md"]["size"] == 5


def test_first_run_everything_is_changed(tmp_path):
    vault, cfg = _setup(tmp_path)
    _write(vault / "a.md")
    _write(vault / "b.md")
    snap = snapshot(cfg, scan_notes(cfg))
    changes = compute_changes({}, snap)
    assert changes.changed == ["a.md", "b.md"]
    assert changes.deleted == []


def test_unchanged_files_skipped(tmp_path):
    vault, cfg = _setup(tmp_path)
    _write(vault / "a.md")
    snap = snapshot(cfg, scan_notes(cfg))
    changes = compute_changes(snap, snapshot(cfg, scan_notes(cfg)))
    assert changes.changed == []
    assert changes.deleted == []


def test_modified_file_detected(tmp_path):
    vault, cfg = _setup(tmp_path)
    f = vault / "a.md"
    _write(f, "one")
    old = snapshot(cfg, scan_notes(cfg))
    _write(f, "one two")
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    new = snapshot(cfg, scan_notes(cfg))
    assert compute_changes(old, new).changed == ["a.md"]


def test_deleted_file_detected(tmp_path):
    vault, cfg = _setup(tmp_path)
    _write(vault / "a.md")
    _write(vault / "b.md")
    old = snapshot(cfg, scan_notes(cfg))
    (vault / "b.md").unlink()
    new = snapshot(cfg, scan_notes(cfg))
    changes = compute_changes(old, new)
    assert changes.changed == []
    assert changes.deleted == ["b.md"]


def test_manifest_file_is_valid_json(tmp_path):
    _, cfg = _setup(tmp_path)
    save_manifest(cfg, {"a.md": {"mtime_ns": 1, "size": 2}})
    data = json.loads((cfg.index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data == {"a.md": {"mtime_ns": 1, "size": 2}}


def test_snapshot_skips_files_that_vanished(tmp_path):
    vault, cfg = _setup(tmp_path)
    _write(vault / "a.md")
    gone = vault / "gone.md"
    snap = snapshot(cfg, [vault.resolve() / "a.md", gone.resolve()])
    assert list(snap) == ["a.md"]


def test_snapshot_keys_are_prefixed_with_several_roots(tmp_path):
    import os

    a, b = tmp_path / "alfa", tmp_path / "beta"
    _write(a / "x.md")
    _write(b / "sub" / "y.md")
    cfg = load_config({"RAG_VAULT_ROOT": os.pathsep.join([str(a), str(b)]),
                       "RAG_INDEX_DIR": str(tmp_path / "idx")})
    assert sorted(snapshot(cfg, scan_notes(cfg))) == ["alfa/x.md", "beta/sub/y.md"]
