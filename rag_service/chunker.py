"""Split a markdown note into overlapping, heading-aware chunks."""
import math
import re
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_MAX_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50
DEFAULT_MIN_TOKENS = 60

# Rough default estimate; pass Embedder.count_tokens to chunk_note for exact sizing.
_TOKENS_PER_WORD = 1.5

_FRONTMATTER = re.compile(r"---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)", re.S)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_MAX_HEADING_CHARS = 150
_INLINE_CODE = re.compile(r"`[^`]*`")
_INLINE_TAG = re.compile(r"(?<![\w#&/])#([^\W\d_][\w/-]*)")


@dataclass(frozen=True)
class Chunk:
    path: str  # note path relative to the vault root
    index: int  # position within the note
    heading_path: str  # e.g. "Mimari > Embedding"; "" before the first heading
    text: str
    tags: tuple[str, ...]

    @property
    def embed_text(self) -> str:
        """What the embedding model and BM25 should see: note title, folder, tags, heading path, body."""
        return _header(self.path, self.tags, self.heading_path) + self.text


def note_title(path: str) -> str:
    """File name without the .md extension: 'a/b/Not (2026-09-25).md' -> 'Not (2026-09-25)'."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return name[:-3] if name.lower().endswith(".md") else name


def _folder(path: str) -> str:
    parts = path.replace("\\", "/").split("/")[:-1]
    return " > ".join(p for p in parts if p)


def _header(path: str, tags: tuple[str, ...], heading_path: str) -> str:
    """The lines put in front of a chunk's body in embed_text (each ends with a newline)."""
    out = "Note: " + note_title(path) + "\n"
    folder = _folder(path)
    if folder:
        out += "Folder: " + folder + "\n"
    if tags:
        out += "Tags: " + ", ".join(tags) + "\n"
    if heading_path:
        out += "Section: " + heading_path + "\n"
    return out


def count_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * _TOKENS_PER_WORD)


def parse_frontmatter(text: str) -> tuple[list[str], str]:
    """Return (tags, body). The frontmatter block is removed from the body."""
    text = text.replace("\r\n", "\n")
    m = _FRONTMATTER.match(text)
    if not m:
        return [], text
    return _parse_tags(m.group(1)), text[m.end():]


def _clean_tag(raw: str) -> str:
    return raw.strip().strip("'\"").lstrip("#").strip()


def _parse_tags(block: str) -> list[str]:
    lines = block.split("\n")
    for i, line in enumerate(lines):
        if not line.lower().startswith("tags:"):
            continue
        value = line.split(":", 1)[1].strip()
        if value:
            value = value.strip("[]")
            raw = value.split(",")
        else:
            raw = []
            for nxt in lines[i + 1:]:
                if not nxt.lstrip().startswith("-"):
                    break
                raw.append(nxt.lstrip()[1:])
        return [t for t in (_clean_tag(r) for r in raw) if t]
    return []


def _is_fence(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("```") or s.startswith("~~~")


def _inline_tags(body: str) -> list[str]:
    """Collect #tag words from prose (not headings, code fences, inline code or URLs)."""
    found: list[str] = []
    in_fence = False
    for line in body.split("\n"):
        if _is_fence(line):
            in_fence = not in_fence
            continue
        if in_fence or _HEADING.match(line):
            continue
        found.extend(_INLINE_TAG.findall(_INLINE_CODE.sub("", line)))
    return found


def _merge_tags(*groups: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for tag in group:
            if tag.lower() not in seen:
                seen.add(tag.lower())
                out.append(tag)
    return tuple(out)


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split on headings (ignoring '#' lines inside code fences) -> [(heading_path, text)]."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    path = ""
    buf: list[str] = []
    in_fence = False

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((path, text))
        buf.clear()

    for line in body.split("\n"):
        if _is_fence(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m and len(m.group(2)) > _MAX_HEADING_CHARS:
            m = None  # a whole note pasted onto one "## ..." line is body text, not a title
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
            path = " > ".join(t for _, t in stack)
        buf.append(line)
    flush()
    return sections


def _merge_small(sections, limit_for, count, min_tokens: int):
    def join(a, b):
        # the section with more content names the merged chunk
        path = a[0] if count(a[1]) >= count(b[1]) else b[0]
        return path, a[1] + "\n\n" + b[1]

    merged: list[tuple[str, str]] = []
    carry: tuple[str, str] | None = None
    for sec in sections:
        if carry:
            joined = join(carry, sec)
            if count(joined[1]) <= limit_for(joined[0]):
                sec = joined
            else:
                merged.append(carry)
            carry = None
        if count(sec[1]) < min_tokens:
            carry = sec
        else:
            merged.append(sec)
    if carry:
        if merged:
            joined = join(merged[-1], carry)
            if count(joined[1]) <= limit_for(joined[0]):
                merged[-1] = joined
                carry = None
        if carry:
            merged.append(carry)
    return merged


def _paragraphs(text: str) -> list[str]:
    paras: list[str] = []
    buf: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _is_fence(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if buf:
                paras.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        paras.append("\n".join(buf))
    return paras


def _split_word(word: str, limit: int, count) -> list[str]:
    """Cut one whitespace-free run (long numbers, minified code) into pieces within limit."""
    pieces: list[str] = []
    while word:
        if count(word) <= limit:
            pieces.append(word)
            break
        lo, hi = 1, min(len(word), limit * 16)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(word[:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        pieces.append(word[:lo])
        word = word[lo:]
    return pieces


def _window(text: str, limit: int, overlap: int, count) -> list[str]:
    """One paragraph longer than limit: overlapping windows of words (long words are cut)."""
    pieces = [part for w in text.split() for part in _split_word(w, limit, count)]
    sizes = [count(p) for p in pieces]
    out: list[str] = []
    start = 0
    while start < len(pieces):
        end, total = start, 0
        while end < len(pieces) and total + sizes[end] <= limit:
            total += sizes[end]
            end += 1
        end = max(end, start + 1)
        # per-piece counts are only approximately additive; verify and trim
        while end > start + 1 and count(" ".join(pieces[start:end])) > limit:
            end -= 1
        out.append(" ".join(pieces[start:end]))
        if end >= len(pieces):
            break
        back, acc = end, 0
        while back - 1 > start and acc + sizes[back - 1] <= overlap:
            back -= 1
            acc += sizes[back]
        start = back
    return out


def _split_long(text: str, limit: int, overlap: int, count) -> list[str]:
    units: list[str] = []
    for para in _paragraphs(text):
        if count(para) > limit:
            units.extend(_window(para, limit, overlap, count))
        else:
            units.append(para)

    chunks: list[str] = []
    cur: list[str] = []
    for unit in units:
        if cur and count("\n\n".join(cur + [unit])) > limit:
            chunks.append("\n\n".join(cur))
            carry: list[str] = []
            for prev in reversed(cur):
                if count("\n\n".join([prev] + carry)) > overlap:
                    break
                carry.insert(0, prev)
            if carry and count("\n\n".join(carry + [unit])) > limit:
                carry = []
            cur = carry
        cur.append(unit)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_note(
    path: str,
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    count: Callable[[str], int] = count_tokens,
) -> list[Chunk]:
    """Split a note into chunks whose embed_text stays within max_tokens.

    `count` measures text length in tokens; pass the embedding model's real tokenizer
    (Embedder.count_tokens) to size chunks exactly. max_tokens should leave a little
    room under the model limit for its special tokens and prefix.
    """
    fm_tags, body = parse_frontmatter(text)
    tags = _merge_tags(fm_tags, _inline_tags(body))

    def limit_for(heading_path: str) -> int:
        # the tags/heading lines are embedded too, so they use part of the budget;
        # the floor keeps a pathologically long header from starving the body entirely
        header = count(_header(path, tags, heading_path))
        return max(max_tokens - header, max_tokens // 4)

    sections = _merge_small(_split_sections(body), limit_for, count, min_tokens)

    chunks: list[Chunk] = []
    for heading_path, sec_text in sections:
        limit = limit_for(heading_path)
        pieces = (
            _split_long(sec_text, limit, overlap_tokens, count)
            if count(sec_text) > limit
            else [sec_text]
        )
        for piece in pieces:
            chunks.append(
                Chunk(path=path, index=len(chunks), heading_path=heading_path,
                      text=piece.strip(), tags=tags)
            )
    return chunks
