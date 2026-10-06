import os

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
    assert cfg.vault_roots == (tmp_path.resolve(),)
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
    assert cfg.vault_roots == (vault.resolve(),)
    assert cfg.exclude_dirs == ("a", "b")


def test_real_environment_overrides_dotenv_file(tmp_path, monkeypatch):
    file_vault, env_vault = tmp_path / "f", tmp_path / "e"
    file_vault.mkdir()
    env_vault.mkdir()
    envfile = tmp_path / ".env"
    envfile.write_text(f"RAG_VAULT_ROOT={file_vault}\n", encoding="utf-8")
    monkeypatch.setenv("RAG_VAULT_ROOT", str(env_vault))
    assert load_config(dotenv_path=envfile).vault_roots == (env_vault.resolve(),)


def test_missing_dotenv_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_VAULT_ROOT", str(tmp_path))
    assert load_config(dotenv_path=tmp_path / "nope.env").vault_roots == (tmp_path.resolve(),)


def test_explicit_env_dict_ignores_dotenv_file(tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text(f"RAG_VAULT_ROOT={tmp_path}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not set"):
        load_config({}, dotenv_path=envfile)


# ---- several vault roots ----

def _roots(tmp_path, *names):
    out = []
    for n in names:
        d = tmp_path / n
        d.mkdir(exist_ok=True)
        out.append(d)
    return out


def test_multiple_roots_split_on_path_separator(tmp_path):
    a, b = _roots(tmp_path, "a", "b")
    cfg = load_config({"RAG_VAULT_ROOT": os.pathsep.join([str(a), " ", str(b)]),
                       "RAG_INDEX_DIR": str(tmp_path / "idx")})
    assert cfg.vault_roots == (a.resolve(), b.resolve())


def test_every_root_must_exist(tmp_path):
    (a,) = _roots(tmp_path, "a")
    with pytest.raises(ConfigError, match="not a directory.*missing"):
        load_config({"RAG_VAULT_ROOT": os.pathsep.join([str(a), str(tmp_path / "missing")])})


def test_roots_with_same_folder_name_rejected(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    a, b = tmp_path / "x" / "notes", tmp_path / "y" / "notes"
    a.mkdir()
    b.mkdir()
    with pytest.raises(ConfigError, match="same folder name"):
        load_config({"RAG_VAULT_ROOT": os.pathsep.join([str(a), str(b)])})


def test_nested_roots_rejected(tmp_path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    with pytest.raises(ConfigError, match="inside another"):
        load_config({"RAG_VAULT_ROOT": os.pathsep.join([str(outer), str(inner)])})


def test_index_dir_must_be_outside_every_root(tmp_path):
    a, b = _roots(tmp_path, "a", "b")
    with pytest.raises(ConfigError, match="outside"):
        load_config({"RAG_VAULT_ROOT": os.pathsep.join([str(a), str(b)]),
                     "RAG_INDEX_DIR": str(b / "idx")})


def test_note_key_single_root_is_plain_relative_path(tmp_path):
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path)})
    p = tmp_path.resolve() / "v" / "a.md"
    assert cfg.note_key(p) == "v/a.md"
    assert cfg.resolve_key("v/a.md") == p


def test_note_key_with_several_roots_is_prefixed_by_root_name(tmp_path):
    a, b = _roots(tmp_path, "alfa", "beta")
    cfg = load_config({"RAG_VAULT_ROOT": os.pathsep.join([str(a), str(b)]),
                       "RAG_INDEX_DIR": str(tmp_path / "idx")})
    p = b.resolve() / "sub" / "n.md"
    assert cfg.note_key(p) == "beta/sub/n.md"
    assert cfg.resolve_key("beta/sub/n.md") == p
    assert cfg.resolve_key("alfa/x.md") == a.resolve() / "x.md"


def test_rerank_dir_default_override_and_disable(tmp_path):
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path)})
    assert cfg.rerank_dir.parts[-2:] == ("models", "reranker")
    assert cfg.rerank_required is False  # default location: used only if the model is there
    custom = tmp_path / "rr"
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path), "RAG_RERANK_DIR": str(custom)})
    assert cfg.rerank_dir == custom.resolve() and cfg.rerank_required is True
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path), "RAG_RERANK_DIR": ""})
    assert cfg.rerank_dir is None  # explicitly empty = reranking off


# ---- service settings ----

def test_service_defaults(tmp_path):
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path)})
    assert (cfg.port, cfg.idle_minutes, cfg.reindex_minutes, cfg.api_key) == (2190, 10, 30, None)


def test_service_overrides(tmp_path):
    cfg = load_config({"RAG_VAULT_ROOT": str(tmp_path), "RAG_PORT": "3000", "RAG_IDLE_MINUTES": "0",
                       "RAG_REINDEX_MINUTES": "1.5", "RAG_API_KEY": " secret "})
    assert (cfg.port, cfg.idle_minutes, cfg.reindex_minutes, cfg.api_key) == (3000, 0, 1.5, "secret")


@pytest.mark.parametrize("key,val", [("RAG_PORT", "abc"), ("RAG_PORT", "0"), ("RAG_PORT", "70000"),
                                     ("RAG_IDLE_MINUTES", "-1"), ("RAG_REINDEX_MINUTES", "x")])
def test_bad_service_settings_rejected(tmp_path, key, val):
    with pytest.raises(ConfigError, match=key):
        load_config({"RAG_VAULT_ROOT": str(tmp_path), key: val})


def test_empty_api_key_means_none(tmp_path):
    assert load_config({"RAG_VAULT_ROOT": str(tmp_path), "RAG_API_KEY": "  "}).api_key is None
