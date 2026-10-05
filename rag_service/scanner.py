"""Find the markdown notes under the configured vault root."""
import os
from pathlib import Path

from rag_service.config import Config


def _split_excludes(entries: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """Entries with a slash are paths relative to the vault root; the rest are bare names."""
    names: set[str] = set()
    paths: set[str] = set()
    for entry in entries:
        norm = entry.replace("\\", "/").strip("/")
        if not norm:
            continue
        (paths if "/" in norm else names).add(norm)
    return names, paths


def scan_notes(cfg: Config) -> list[Path]:
    """Return all .md files under cfg.vault_root, sorted, minus excluded dirs/files.

    An exclude entry like "logs" skips any dir or file with that name anywhere;
    "a/b/logs" skips only that path (a dir or a single file) relative to the vault root.
    """
    names, paths = _split_excludes(cfg.exclude_dirs)
    root = cfg.vault_root
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = [d for d in dirnames if d not in names and prefix + d not in paths]
        for name in filenames:
            if name in names or prefix + name in paths:
                continue
            if name.lower().endswith(".md"):
                found.append(Path(dirpath) / name)
    return sorted(found)
