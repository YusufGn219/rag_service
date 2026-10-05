"""Configuration, read from environment variables only (no personal defaults)."""
import os
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EXCLUDES = (".git", ".obsidian", ".trash")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    vault_root: Path
    index_dir: Path
    exclude_dirs: tuple[str, ...]
    model_dir: Path


def _read_dotenv(path) -> dict[str, str]:
    """Minimal .env reader: KEY=VALUE lines, # comments, optional quotes around values."""
    values: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def load_config(env=None, dotenv_path=None) -> Config:
    """Read settings from the environment; a .env file fills in what the environment lacks.

    An explicit `env` dict is used as-is (no .env, no os.environ) - handy for tests.
    """
    if env is None:
        env = {**_read_dotenv(dotenv_path or _PROJECT_ROOT / ".env"), **os.environ}

    raw_root = env.get("RAG_VAULT_ROOT", "").strip()
    if not raw_root:
        raise ConfigError("RAG_VAULT_ROOT is not set (see .env.example)")
    vault_root = Path(raw_root).expanduser().resolve()
    if not vault_root.is_dir():
        raise ConfigError(f"RAG_VAULT_ROOT is not a directory: {vault_root}")

    raw_index = env.get("RAG_INDEX_DIR", "").strip()
    index_dir = Path(raw_index).expanduser().resolve() if raw_index else _PROJECT_ROOT / "data" / "index"
    if index_dir == vault_root or vault_root in index_dir.parents:
        raise ConfigError(
            f"RAG_INDEX_DIR must be outside RAG_VAULT_ROOT (self-indexing): {index_dir}"
        )

    raw_ex = env.get("RAG_EXCLUDE_DIRS")
    exclude = (
        tuple(p.strip() for p in raw_ex.split(",") if p.strip())
        if raw_ex is not None
        else _DEFAULT_EXCLUDES
    )
    raw_model = env.get("RAG_MODEL_DIR", "").strip()
    model_dir = (
        Path(raw_model).expanduser().resolve()
        if raw_model
        else _PROJECT_ROOT / "data" / "models" / "multilingual-e5-small"
    )
    return Config(vault_root=vault_root, index_dir=index_dir, exclude_dirs=exclude, model_dir=model_dir)
