"""Find the markdown notes under the configured vault root."""
import os
from pathlib import Path

from rag_service.config import Config


def scan_notes(cfg: Config) -> list[Path]:
    """Return all .md files under cfg.vault_root, sorted, skipping excluded dir names."""
    excluded = set(cfg.exclude_dirs)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(cfg.vault_root):
        dirnames[:] = [d for d in dirnames if d not in excluded]
        for name in filenames:
            if name.lower().endswith(".md"):
                found.append(Path(dirpath) / name)
    return sorted(found)
