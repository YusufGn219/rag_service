import pytest

from rag_service.config import ConfigError, load_config


def test_missing_root_raises():
    with pytest.raises(ConfigError, match="RAG_VAULT_ROOT is not set"):
        load_config({})


def test_root_must_be_dir(tmp_path):
    with pytest.raises(ConfigError, match="not a directory"):
        load_config({"RAG_VAULT_ROOT": str(tmp_path / "nope")})


def test_defaults(tmp_path):
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path)})
    assert cfg.vault_root == tmp_path.resolve()
    assert cfg.exclude_dirs == (".git", ".obsidian", ".trash")
    assert cfg.index_dir.parts[-2:] == ("data", "index")


def test_index_inside_vault_rejected(tmp_path):
    with pytest.raises(ConfigError, match="outside"):
        load_config({"RAG_VAULT_ROOT": str(tmp_path), "RAG_INDEX_DIR": str(tmp_path / "idx")})


def test_custom_excludes(tmp_path):
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path), "RAG_EXCLUDE_DIRS": "a, b"})
    assert cfg.exclude_dirs == ("a", "b")
