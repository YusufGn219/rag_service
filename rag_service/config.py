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
    vault_roots: tuple[Path, ...]
    index_dir: Path
    exclude_dirs: tuple[str, ...]
    model_dir: Path
    # Folder of the reranker model (None = reranking off). rerank_required is True when the
    # user named the folder: a missing model is then an error, not a silent "off".
    rerank_dir: Path | None = None
    rerank_required: bool = False
    # Service settings (see service.py / server.py)
    port: int = 2190
    idle_minutes: float = 10  # unload models after this long without a search; 0 = never
    reindex_minutes: float = 30  # update the index this often; 0 = never
    api_key: str | None = None  # when set, the HTTP API requires it

    def note_key(self, path: Path) -> str:
        """Stable id of a note: its path relative to its root ("root-name/..." with several roots)."""
        for root in self.vault_roots:
            if path.is_relative_to(root):
                rel = path.relative_to(root).as_posix()
                return rel if len(self.vault_roots) == 1 else f"{root.name}/{rel}"
        raise ValueError(f"{path} is not inside any vault root")

    def resolve_key(self, key: str) -> Path:
        """Inverse of note_key: the absolute path of a note."""
        if len(self.vault_roots) == 1:
            return self.vault_roots[0] / key
        name, _, rel = key.partition("/")
        for root in self.vault_roots:
            if root.name == name:
                return root / rel
        raise ValueError(f"no vault root named {name!r} for note {key!r}")


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


def _number(env, key: str, default, *, kind=float, low=0, high=None):
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from None
    if value < low or (high is not None and value > high):
        bound = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise ConfigError(f"{key} must be {bound}, got {raw!r}")
    return value


def load_config(env=None, dotenv_path=None) -> Config:
    """Read settings from the environment; a .env file fills in what the environment lacks.

    An explicit `env` dict is used as-is (no .env, no os.environ) - handy for tests.
    """
    if env is None:
        env = {**_read_dotenv(dotenv_path or _PROJECT_ROOT / ".env"), **os.environ}

    parts = [p.strip() for p in env.get("RAG_VAULT_ROOT", "").split(os.pathsep) if p.strip()]
    if not parts:
        raise ConfigError("RAG_VAULT_ROOT is not set (see .env.example)")
    roots = []
    for part in parts:
        root = Path(part).expanduser().resolve()
        if not root.is_dir():
            raise ConfigError(f"RAG_VAULT_ROOT is not a directory: {root}")
        roots.append(root)
    names = [r.name for r in roots]
    if len(set(names)) != len(names):
        raise ConfigError(f"RAG_VAULT_ROOT folders must not share the same folder name: {names}")
    for a in roots:
        for b in roots:
            if a != b and a in b.parents:
                raise ConfigError(f"RAG_VAULT_ROOT folder {b} is inside another root ({a})")

    raw_index = env.get("RAG_INDEX_DIR", "").strip()
    index_dir = Path(raw_index).expanduser().resolve() if raw_index else _PROJECT_ROOT / "data" / "index"
    for root in roots:
        if index_dir == root or root in index_dir.parents:
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
    if "RAG_RERANK_DIR" not in env:
        rerank_dir, rerank_required = _PROJECT_ROOT / "data" / "models" / "reranker", False
    elif not env["RAG_RERANK_DIR"].strip():
        rerank_dir, rerank_required = None, False
    else:
        rerank_dir, rerank_required = Path(env["RAG_RERANK_DIR"].strip()).expanduser().resolve(), True
    return Config(vault_roots=tuple(roots), index_dir=index_dir, exclude_dirs=exclude,
                  model_dir=model_dir, rerank_dir=rerank_dir, rerank_required=rerank_required,
                  port=_number(env, "RAG_PORT", 2190, kind=int, low=1, high=65535),
                  idle_minutes=_number(env, "RAG_IDLE_MINUTES", 10),
                  reindex_minutes=_number(env, "RAG_REINDEX_MINUTES", 30),
                  api_key=env.get("RAG_API_KEY", "").strip() or None)
