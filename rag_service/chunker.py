"""Split a markdown note into overlapping, heading-aware chunks."""
import math
import re
from dataclasses import dataclass

DEFAULT_MAX_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50
DEFAULT_MIN_TOKENS = 60

# Rough estimate until the real embedding tokenizer is wired in (kept in one place on purpose).
_TOKENS_PER_WORD = 1.5

_FRONTMATTER = re.compile(r"---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)", re.S)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
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
        """What the embedding model and BM25 should see: tags + heading path + body."""
        parts = []
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags))
        if self.heading_path:
            parts.append("Section: " + self.heading_path)
        parts.append(self.text)
        return "\n".join(parts)


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


def _merge_small(sections: list[tuple[str, str]], max_tokens: int, min_tokens: int):
    merged: list[tuple[str, str]] = []
    carry: tuple[str, str] | None = None
    for path, text in sections:
        if carry:
            joined = carry[1] + "\n\n" + text
            if count_tokens(joined) <= max_tokens:
                # the section with more content names the merged chunk
                if count_tokens(carry[1]) >= count_tokens(text):
                    path = carry[0]
                text = joined
            else:
                merged.append(carry)
            carry = None
        if count_tokens(text) < min_tokens:
            carry = (path, text)
        else:
            merged.append((path, text))
    if carry:
        if merged and count_tokens(merged[-1][1] + "\n\n" + carry[1]) <= max_tokens:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n\n" + carry[1])
        else:
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


def _window_words(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Fallback for one paragraph longer than max_tokens: overlapping word windows."""
    words = text.split()
    size = max(1, int(max_tokens / _TOKENS_PER_WORD))
    overlap = min(int(overlap_tokens / _TOKENS_PER_WORD), size - 1)
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def _split_long(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    units: list[str] = []
    for para in _paragraphs(text):
        if count_tokens(para) > max_tokens:
            units.extend(_window_words(para, max_tokens, overlap_tokens))
        else:
            units.append(para)

    chunks: list[str] = []
    cur: list[str] = []
    for unit in units:
        if cur and count_tokens("\n\n".join(cur + [unit])) > max_tokens:
            chunks.append("\n\n".join(cur))
            carry: list[str] = []
            for prev in reversed(cur):
                if count_tokens("\n\n".join([prev] + carry)) > overlap_tokens:
                    break
                carry.insert(0, prev)
            if carry and count_tokens("\n\n".join(carry + [unit])) > max_tokens:
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
) -> list[Chunk]:
    fm_tags, body = parse_frontmatter(text)
    tags = _merge_tags(fm_tags, _inline_tags(body))
    sections = _merge_small(_split_sections(body), max_tokens, min_tokens)

    chunks: list[Chunk] = []
    for heading_path, sec_text in sections:
        pieces = (
            _split_long(sec_text, max_tokens, overlap_tokens)
            if count_tokens(sec_text) > max_tokens
            else [sec_text]
        )
        for piece in pieces:
            chunks.append(
                Chunk(path=path, index=len(chunks), heading_path=heading_path,
                      text=piece.strip(), tags=tags)
            )
    return chunks
