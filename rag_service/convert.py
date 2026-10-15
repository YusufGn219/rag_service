"""Convert files next to themselves. The work behind the bridge's inspect / convert / scan / status.

Safety rules (they live here, not in the caller):
- Only files under a vault root are touched, and never inside an excluded folder.
- The only thing written is `<file>.<ext>.md` beside the source. A new note is created, never
  overwritten; an existing note is replaced only when its header says rag_service made it.
- The source file itself is never changed, moved or deleted.
"""
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rag_service import readers, sidecar
from rag_service.config import Config
from rag_service.errors import ServiceError
from rag_service.scanner import _split_excludes

MAX_BYTES = 200 * 1024 * 1024  # larger files are refused
MAX_SCAN_ROWS = 200  # files listed by scan() at most
PREVIEW_CHARS = 300  # how much of the text inspect_file shows
_PAGE_MARKER = re.compile(r"^## Sayfa \d+$")

WRITTEN = "yazıldı"
REFRESHED = "yenilendi"
SKIPPED = "atlandı"  # up to date, or a hand-written note is in the way
HARD = "zor"  # nothing usable came out; not written


def _excluded(cfg: Config, root: Path, path: Path) -> bool:
    names, paths = _split_excludes(cfg.exclude_dirs)
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    return any(p in names for p in parts) or any(rel == x or rel.startswith(x + "/") for x in paths)


def _root_of(cfg: Config, path: Path) -> Path:
    for root in cfg.vault_roots:
        if path.is_relative_to(root):
            return root
    raise ServiceError(f"{path} is not inside a vault root")


def _resolve(cfg: Config, raw: str) -> Path:
    """An absolute path, or a path as search results give it (relative to a vault root)."""
    path = Path(raw)
    if not path.is_absolute():
        try:
            path = cfg.resolve_key(raw)
        except ValueError as exc:
            raise ServiceError(f"cannot find {raw!r}: {exc}") from None
    path = path.resolve()
    root = _root_of(cfg, path)
    if _excluded(cfg, root, path):
        raise ServiceError(f"{raw!r} is inside an excluded folder")
    return path


def _source(cfg: Config, raw: str) -> Path:
    path = _resolve(cfg, raw)
    if not path.is_file():
        raise ServiceError(f"{raw!r} is not a file")
    if readers.kind_of(path) is None:
        raise ServiceError(f"{path.suffix or path.name!r} is not a supported type "
                           f"({', '.join(sorted(readers.SUPPORTED))})")
    if path.stat().st_size > MAX_BYTES:
        raise ServiceError(f"{raw!r} is larger than {MAX_BYTES // (1024 * 1024)} MB")
    return path


def _read(path: Path) -> readers.Extracted:
    try:
        return readers.extract(path)
    except readers.ReadError as exc:
        raise ServiceError(str(exc)) from None


def _preview(markdown: str, limit: int = PREVIEW_CHARS) -> str:
    """The first `limit` characters of the real content: without the title line (it is just the
    file name) and the "## Sayfa N" markers, whitespace squeezed to single spaces."""
    lines = [ln for ln in markdown.split("\n")[1:] if not _PAGE_MARKER.match(ln)]
    text = " ".join(" ".join(lines).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def inspect_file(cfg: Config, raw: str) -> dict:
    """Look at a file without writing anything: type, size, pages, whether text comes out, when it
    last changed and how its text starts (to tell versions of one document from different ones)."""
    src = _source(cfg, raw)
    ex = _read(src)
    stat = src.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")
    return {
        "path": cfg.note_key(src), "kind": ex.kind, "bytes": stat.st_size, "pages": ex.pages,
        "chars": len(ex.markdown), "hard": ex.hard, "reason": ex.reason, "modified": modified,
        "preview": _preview(ex.markdown),
        "status": sidecar.plan(src), "note": cfg.note_key(sidecar.sidecar_path(src)),
    }


def _body(src: Path, markdown: str) -> str:
    text = markdown.strip()
    return (text if text.startswith("# ") else f"# {src.stem}\n\n{text}") + "\n"


def _write(target: Path, text: str, *, replace: bool) -> bool:
    """Write the note. A new note is created exclusively (never over an existing file); a
    replacement goes through a temp file so a crash cannot leave half a note. False when a
    file appeared in the meantime."""
    if not replace:
        try:
            with open(target, "x", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        except FileExistsError:
            return False
        return True
    tmp = target.with_name("." + target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, target)
    return True


def convert_file(cfg: Config, raw: str, markdown: str | None = None) -> dict:
    """Write `<file>.md` next to the source. With `markdown`, that text (made by hand or by an
    agent for a file the reader could not handle) is used instead of the reader's output."""
    src = _source(cfg, raw)
    target = sidecar.sidecar_path(src)
    state = sidecar.plan(src)
    result = {"path": cfg.note_key(src), "note": cfg.note_key(target)}
    if state == sidecar.FOREIGN:
        return {**result, "action": SKIPPED, "reason": "bu adla elle yazılmış bir not var, dokunulmadı"}
    if state == sidecar.CURRENT and markdown is None:
        return {**result, "action": SKIPPED, "reason": "not güncel, kaynak değişmemiş"}

    version = sidecar.source_version(src)  # taken before reading, see sidecar.render
    if markdown is None:
        ex = _read(src)
        if ex.hard:
            return {**result, "action": HARD, "kind": ex.kind, "pages": ex.pages, "reason": ex.reason}
        text = sidecar.render(src, ex.markdown, ex.kind, ex.pages, version=version)
    else:
        if not markdown.strip():
            raise ServiceError("markdown is empty")
        text = sidecar.render(src, _body(src, markdown), readers.kind_of(src), version=version,
                              extra={"okuma": "agent"})
    created = _write(target, text, replace=state != sidecar.NEW)
    if not created:  # someone else created the note between plan() and the write
        return {**result, "action": SKIPPED, "reason": "bu adla bir not az önce oluşmuş, dokunulmadı"}
    return {**result, "action": WRITTEN if state == sidecar.NEW else REFRESHED, "chars": len(text)}


def _sources(cfg: Config, base: Path | None = None):
    """Every supported file under `base` (default: all vault roots), minus excluded folders."""
    bases = [base] if base is not None else list(cfg.vault_roots)
    for start in bases:
        root = _root_of(cfg, start)
        for dirpath, dirnames, filenames in os.walk(start):
            here = Path(dirpath)
            dirnames[:] = sorted(d for d in dirnames if not _excluded(cfg, root, here / d))
            for name in sorted(filenames):
                if readers.kind_of(Path(name)) and not _excluded(cfg, root, here / name):
                    yield here / name


def _summary(cfg: Config, base: Path | None, limit: int) -> dict:
    counts: Counter = Counter()
    kinds: Counter = Counter()
    rows = []
    for src in _sources(cfg, base):
        state = sidecar.plan(src)
        counts[state] += 1
        kinds[readers.kind_of(src)] += 1
        rows.append({"path": cfg.note_key(src), "kind": readers.kind_of(src),
                     "bytes": src.stat().st_size, "status": state})
    order = {sidecar.NEW: 0, sidecar.REFRESH: 1, sidecar.FOREIGN: 2, sidecar.CURRENT: 3}
    rows.sort(key=lambda r: (order[r["status"]], r["path"]))
    return {"total": len(rows), "by_status": dict(counts), "by_kind": dict(kinds),
            "files": rows[:limit], "truncated": len(rows) > limit}


def scan(cfg: Config, folder: str | None = None, limit: int = MAX_SCAN_ROWS) -> dict:
    """List convertible files with their state: yeni (not converted yet), yenile (source changed),
    güncel (up to date), atla (a hand-written note has that name). Actionable ones come first."""
    base = None
    if folder:
        base = _resolve(cfg, folder)
        if not base.is_dir():
            raise ServiceError(f"{folder!r} is not a folder")
    return _summary(cfg, base, max(0, min(limit, MAX_SCAN_ROWS)))


def status(cfg: Config) -> dict:
    """Counts over all vault roots (no file list)."""
    summary = _summary(cfg, None, 0)
    return {k: summary[k] for k in ("total", "by_status", "by_kind")}
