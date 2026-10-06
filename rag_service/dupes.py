"""Find notes that are copies of other notes (or mostly made of other notes' text).

Works on the stored index only: it compares runs of words, not embeddings, because the
embedding text starts with the note's name and folder, which would push two copies apart.
Nothing here changes anything; it only reports. See audit.py for the command.

Method: every note becomes the set of its 8-word runs ("shingles"). Two notes are linked when
the smaller one has at least `threshold` of its runs inside the other. Linked notes form a
group; the suggestion is which of them could be left out while their text stays covered by the
rest. Weaker links (`similar_threshold`) only produce "look at this" groups, with no suggestion.
"""
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from rag_service.chunker import note_title
from rag_service.manifest import Snapshot
from rag_service.store import IndexData

SHINGLE_WORDS = 8
DEFAULT_THRESHOLD = 0.9  # "overlap": nearly all of the smaller note is in the other
DEFAULT_SIMILAR = 0.6  # "similar": most of it is, but it has real differences
NAME_OVERLAP = 0.2  # same name (up to letters) AND at least this much shared text
SUGGEST_BAR = 0.8  # a note may be suggested for leaving out when this much of it is covered
MIN_WORDS = 30  # shorter notes say too little to call them copies
MAX_SHARERS = 25  # a run found in more notes than this is boilerplate, not evidence

_KIND_ORDER = {"exact": 0, "overlap": 1, "similar": 2, "name": 3}
_FOLD = str.maketrans("İIıĞğÜüŞşÖöÇç", "iiiggu" + "ussoocc")
_WORD = re.compile(r"[^\W_]+")
_LINK = re.compile(r"\[\[([^\]|#\n]+)")
_HINTS = frozenset({"yedek", "backup", "kopya", "copy", "eski", "old", "arsiv", "archive"})


@dataclass(frozen=True)
class NoteInfo:
    path: str
    chunks: int
    words: int
    modified: date | None  # last change of the file, if the manifest knows it
    inbound: int  # how many other notes link to it with [[...]]
    hints: tuple[str, ...]  # words in its path that suggest a leftover ("yedek", "old", ...)
    covered: float  # share of its text that is also in the other notes of the group


@dataclass(frozen=True)
class DupGroup:
    key: str  # stable id (same notes -> same key), for remembering decisions
    kind: str  # "exact" | "overlap" | "similar" | "name"
    notes: tuple[NoteInfo, ...]
    detail: str
    keep: tuple[str, ...]  # paths suggested to stay
    exclude: tuple[str, ...]  # paths suggested to leave out ((): no suggestion)


def _fold(text: str) -> str:
    return text.translate(_FOLD).lower()


def _words(text: str) -> list[str]:
    return _WORD.findall(_fold(text))


def _shingles(words: list[str]) -> np.ndarray:
    n = SHINGLE_WORDS
    return np.unique(np.fromiter(
        (hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)), dtype=np.int64,
        count=max(len(words) - n + 1, 0)))


def _stem(path: str) -> str:
    return note_title(path)


def _group_key(paths) -> str:
    return hashlib.sha1("|".join(sorted(paths)).encode("utf-8")).hexdigest()[:12]


class _Union:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _pair_overlaps(arrays: list[np.ndarray]) -> dict[tuple[int, int], int]:
    """For every pair of notes, how many runs they share (boilerplate runs ignored)."""
    owners = np.concatenate([np.full(len(a), i, dtype=np.int64) for i, a in enumerate(arrays)])
    hashes = np.concatenate(arrays)
    order = np.argsort(hashes, kind="stable")
    hashes, owners = hashes[order], owners[order]
    cuts = np.flatnonzero(np.diff(hashes)) + 1
    starts = np.concatenate(([0], cuts))
    sizes = np.diff(np.concatenate((starts, [len(hashes)])))

    counts: dict[tuple[int, int], int] = {}
    two = starts[sizes == 2]  # the common case, done in bulk
    if len(two):
        lo, hi = owners[two], owners[two + 1]
        codes, n = np.unique(np.minimum(lo, hi) * len(arrays) + np.maximum(lo, hi), return_counts=True)
        for code, c in zip(codes.tolist(), n.tolist()):
            counts[divmod(code, len(arrays))] = c
    for start, size in zip(starts[(sizes > 2) & (sizes <= MAX_SHARERS)].tolist(),
                           sizes[(sizes > 2) & (sizes <= MAX_SHARERS)].tolist()):
        group = sorted(owners[start:start + size].tolist())
        for x in range(size):
            for y in range(x + 1, size):
                counts[(group[x], group[y])] = counts.get((group[x], group[y]), 0) + 1
    return counts


def find_duplicates(data: IndexData, manifest: Snapshot, *, threshold: float = DEFAULT_THRESHOLD,
                    similar_threshold: float = DEFAULT_SIMILAR,
                    min_words: int = MIN_WORDS) -> list[DupGroup]:
    """Groups of notes that copy each other, strongest first. Empty list: nothing found."""
    texts: dict[str, list[str]] = {}
    chunk_count: dict[str, int] = {}
    for c in sorted(data.chunks, key=lambda c: (c.path, c.index)):
        texts.setdefault(c.path, []).append(c.text)
        chunk_count[c.path] = chunk_count.get(c.path, 0) + 1
    paths = sorted(texts)
    raw = {p: "\n".join(texts[p]) for p in paths}
    words = {p: _words(raw[p]) for p in paths}

    inbound = _inbound_links(paths, raw)
    analysed = [p for p in paths if len(words[p]) >= min_words]
    shingles = {p: _shingles(words[p]) for p in analysed}
    full_hash = {p: hashlib.sha1(" ".join(words[p]).encode("utf-8")).digest() for p in analysed}
    index = {p: i for i, p in enumerate(analysed)}
    pairs = _pair_overlaps([shingles[p] for p in analysed]) if len(analysed) >= 2 else {}

    def ratio(i: int, j: int, shared: int) -> float:
        return shared / min(len(shingles[analysed[i]]), len(shingles[analysed[j]]))

    def components(edges) -> list[list[str]]:
        union = _Union(len(analysed))
        touched: set[int] = set()
        for i, j in edges:
            union.join(i, j)
            touched.update((i, j))
        found: dict[int, list[str]] = {}
        for i in sorted(touched):
            found.setdefault(union.find(i), []).append(analysed[i])
        return [m for m in found.values() if len(m) >= 2]

    groups: list[DupGroup] = []
    taken: set[str] = set()  # notes already in a strong group

    for members in components(e for e, c in pairs.items() if ratio(*e, c) >= threshold):
        exact = len({full_hash[p] for p in members}) == 1
        groups.append(_build(
            "exact" if exact else "overlap",
            "birebir aynı metin" if exact
            else "metinleri büyük ölçüde örtüşüyor (biri ötekinin içinde ya da küçük farklarla kopyası)",
            members, shingles, words, chunk_count, manifest, inbound, suggest=True))
        taken.update(members)

    weak = (e for e, c in pairs.items()
            if ratio(*e, c) >= similar_threshold and analysed[e[0]] not in taken and analysed[e[1]] not in taken)
    for members in components(weak):
        groups.append(_build("similar", "metinlerin çoğu ortak ama gerçek farklar da var (ör. eski/yeni sürüm); elle bak",
                             members, shingles, words, chunk_count, manifest, inbound, suggest=False))

    by_name: dict[str, list[str]] = {}
    for p in analysed:
        by_name.setdefault(_fold(_stem(p)), []).append(p)
    name_edges = []
    for members in by_name.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                if _stem(a).casefold() == _stem(b).casefold():
                    continue  # same name in different folders is normal
                i, j = sorted((index[a], index[b]))
                if ratio(i, j, pairs.get((i, j), 0)) >= NAME_OVERLAP:
                    name_edges.append((i, j))
    already = [{n.path for n in g.notes} for g in groups]
    for members in components(name_edges):
        if any(set(members) <= g for g in already):
            continue
        groups.append(_build("name", "adlar yalnızca harf farkıyla ayrılıyor (ör. bağlam / baglam) ve metinleri kısmen ortak; elle bak",
                             members, shingles, words, chunk_count, manifest, inbound, suggest=False))

    groups.sort(key=lambda g: (_KIND_ORDER[g.kind], -sum(n.words for n in g.notes), g.key))
    return groups


def _inbound_links(paths: list[str], raw: dict[str, str]) -> dict[str, int]:
    by_title: dict[str, list[str]] = {}
    for p in paths:
        by_title.setdefault(_fold(_stem(p)), []).append(p)
    linkers: dict[str, set[str]] = {p: set() for p in paths}
    for source in paths:
        for target in _LINK.findall(raw[source]):
            name = target.strip().replace("\\", "/").rsplit("/", 1)[-1]
            if name.lower().endswith(".md"):
                name = name[:-3]
            for dest in by_title.get(_fold(name.strip()), ()):
                if dest != source:
                    linkers[dest].add(source)
    return {p: len(s) for p, s in linkers.items()}


def _covered(path: str, others: list[str], shingles: dict[str, np.ndarray]) -> float:
    mine = shingles.get(path)
    if mine is None or len(mine) == 0 or not others:
        return 0.0
    parts = [shingles[o] for o in others if o in shingles]
    if not parts:
        return 0.0
    return float(np.isin(mine, np.concatenate(parts)).mean())


def _hints(path: str) -> tuple[str, ...]:
    return tuple(sorted(set(_WORD.findall(_fold(path))) & _HINTS))


def _build(kind, detail, members, shingles, words, chunk_count, manifest, inbound, *, suggest):
    members = sorted(members)
    infos = []
    for p in members:
        ns = manifest.get(p, {}).get("mtime_ns")
        infos.append(NoteInfo(
            path=p, chunks=chunk_count[p], words=len(words[p]),
            modified=datetime.fromtimestamp(ns / 1e9).date() if ns else None,
            inbound=inbound.get(p, 0), hints=_hints(p),
            covered=_covered(p, [o for o in members if o != p], shingles),
        ))
    excluded: list[str] = []
    if suggest:
        by_path = {i.path: i for i in infos}

        def drop_order(p: str):  # first = best to leave out
            i = by_path[p]
            return (0 if i.hints else 1, i.inbound, ns_of(p), -i.covered, -i.words, p)

        def ns_of(p: str) -> int:
            return manifest.get(p, {}).get("mtime_ns") or 0

        while True:
            left = [p for p in members if p not in excluded]
            ready = [p for p in left if _covered(p, [o for o in left if o != p], shingles) >= SUGGEST_BAR]
            if not ready:
                break
            excluded.append(min(ready, key=drop_order))
    keep = tuple(p for p in members if p not in excluded)
    return DupGroup(key=_group_key(members), kind=kind, notes=tuple(infos), detail=detail,
                    keep=keep if suggest else (), exclude=tuple(sorted(excluded)))
