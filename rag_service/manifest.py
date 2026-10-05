"""Remember which notes were already indexed, so unchanged ones can be skipped."""
import json
from dataclasses import dataclass
from pathlib import Path

from rag_service.config import Config

_MANIFEST_NAME = "manifest.json"

Snapshot = dict[str, dict[str, int]]


@dataclass(frozen=True)
class Changes:
    changed: list[str]  # new or modified notes (relative posix paths)
    deleted: list[str]  # in the old manifest, gone from the vault


def snapshot(cfg: Config, paths: list[Path]) -> Snapshot:
    """Current mtime/size of each note, keyed by note key (path relative to its vault root)."""
    snap: Snapshot = {}
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue  # vanished between the scan and now
        key = cfg.note_key(p)
        snap[key] = {"mtime_ns": st.st_mtime_ns, "size": st.st_size}
    return snap


def compute_changes(old: Snapshot, new: Snapshot) -> Changes:
    changed = sorted(k for k, v in new.items() if old.get(k) != v)
    deleted = sorted(k for k in old if k not in new)
    return Changes(changed=changed, deleted=deleted)


def load_manifest(cfg: Config) -> Snapshot:
    """Missing or unreadable manifest means 'nothing indexed yet' (full re-index)."""
    try:
        data = json.loads((cfg.index_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_manifest(cfg: Config, snap: Snapshot) -> None:
    """Write atomically so a crash mid-write cannot leave a half-written manifest."""
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.index_dir / _MANIFEST_NAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
