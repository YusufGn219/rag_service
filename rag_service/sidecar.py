"""The converted note that sits next to a source file: its name, its header and its freshness.

`Taslak.pdf` becomes `Taslak.pdf.md` in the same folder. The header (frontmatter) says the
note was made by rag_service and which version of the source it came from, so a later run can
tell "mine and up to date", "mine but the source changed" and "written by hand - leave alone".
"""
import re
from datetime import datetime, timezone
from pathlib import Path

PRODUCER = "rag_service"

# What plan() says about a source file:
NEW = "yeni"  # no converted note yet
REFRESH = "yenile"  # our note, source changed since
CURRENT = "güncel"  # our note, source unchanged
FOREIGN = "atla"  # a note with this name exists but we did not make it

_FRONTMATTER = re.compile(r"---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)", re.S)


def sidecar_path(source: Path) -> Path:
    """`dir/Taslak.pdf` -> `dir/Taslak.pdf.md`."""
    source = Path(source)
    return source.with_name(source.name + ".md")


def source_version(source: Path) -> tuple[str, int]:
    """(modified time as UTC text, size in bytes) - what we compare to spot a changed source."""
    stat = Path(source).stat()
    stamp = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="microseconds")
    return stamp, stat.st_size


def render(source: Path, markdown: str, kind: str, pages: int | None = None, hard: bool = False,
           version: tuple[str, int] | None = None, extra: dict[str, str] | None = None) -> str:
    """The full note text: header + the converted body.

    `version` is the source's version taken *before* it was read (default: now), so a source
    that changes while being converted is seen as changed next time. `extra` adds header lines.
    """
    stamp, size = version or source_version(source)
    lines = [
        "---",
        f"tags: [{kind}, kaynak-dosya]",
        f"uretici: {PRODUCER}",
        f"kaynak: {Path(source).name}",
        f"tur: {kind}",
        f"kaynak_degisim: {stamp}",
        f"kaynak_boyut: {size}",
    ]
    if pages is not None:
        lines.append(f"sayfa: {pages}")
    if hard:
        lines.append("zor: evet")
    for key, value in (extra or {}).items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n\n" + markdown


def read_header(text: str) -> dict[str, str]:
    """The `key: value` lines of the frontmatter ({} when there is none)."""
    m = _FRONTMATTER.match(text.replace("\r\n", "\n").lstrip("﻿"))
    if not m:
        return {}
    header: dict[str, str] = {}
    for line in m.group(1).split("\n"):
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key.startswith((" ", "\t", "-")):
            header[key.strip()] = value.strip()
    return header


def is_ours(text: str) -> bool:
    return read_header(text).get("uretici") == PRODUCER


def plan(source: Path) -> str:
    """What converting `source` would do: NEW, REFRESH, CURRENT or FOREIGN."""
    target = sidecar_path(source)
    if not target.exists():
        return NEW
    try:
        header = read_header(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return FOREIGN
    if header.get("uretici") != PRODUCER:
        return FOREIGN
    stamp, size = source_version(source)
    if header.get("kaynak_degisim") == stamp and header.get("kaynak_boyut") == str(size):
        return CURRENT
    return REFRESH
