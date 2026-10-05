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


def test_model_dir_default_and_override(tmp_path):
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path)})
    assert cfg.model_dir.parts[-2:] == ("models", "multilingual-e5-small")
    custom = tmp_path.parent / "mymodel"
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path), "RAG_MODEL_DIR": str(custom)})
    assert cfg.model_dir == custom.resolve()


# ---- .env file ----

def test_dotenv_file_is_read_when_no_explicit_env(tmp_path, monkeypatch):
    vault = tmp_path / "v"
    vault.mkdir()
    envfile = tmp_path / ".env"
    envfile.write_text(
        '# comment\n\nRAG_VAULT_ROOT="%s"\nRAG_EXCLUDE_DIRS=a, b  \nEMPTY=\n' % vault, encoding="utf-8"
    )
    monkeypatch.delenv("RAG_VAULT_ROOT", raising=False)
    monkeypatch.delenv("RAG_EXCLUDE_DIRS", raising=False)
    cfg = load_config(dotenv_path=envfile)
    assert cfg.vault_root == vault.resolve()
    assert cfg.exclude_dirs == ("a", "b")


def test_real_environment_overrides_dotenv_file(tmp_path, monkeypatch):
    file_vault, env_vault = tmp_path / "f", tmp_path / "e"
    file_vault.mkdir()
    env_vault.mkdir()
    envfile = tmp_path / ".env"
    envfile.write_text(f"RAG_VAULT_ROOT={file_vault}\n", encoding="utf-8")
    monkeypatch.setenv("RAG_VAULT_ROOT", str(env_vault))
    assert load_config(dotenv_path=envfile).vault_root == env_vault.resolve()


def test_missing_dotenv_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_VAULT_ROOT", str(tmp_path))
    assert load_config(dotenv_path=tmp_path / "nope.env").vault_root == tmp_path.resolve()


def test_explicit_env_dict_ignores_dotenv_file(tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text(f"RAG_VAULT_ROOT={tmp_path}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not set"):
        load_config({}, dotenv_path=envfile)
