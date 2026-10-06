"""Find notes that crowd search results without being the right answer ("noise").

Two kinds of evidence, combined:
  - behaviour: run many test queries (note titles, and sentences taken from note bodies) and
    count how often each note comes back for queries that were NOT about it. A note that comes
    back far more often than the typical note is a "hub".
  - features: very long, no links in or out, no headings. Not noise on their own; they only
    tell a harmful hub from a useful one (an index note is a hub too, but it is linked to).
A hub with two or more features is a "strong" candidate; any other hub is "look". Within each
kind, the notes with the most features come first, then the most often returned.
Nothing here changes anything; it only reports. See audit.py for the command.
"""
import random
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rag_service.chunker import note_title
from rag_service.dupes import link_counts, path_hints, words_of
from rag_service.store import IndexData

DEFAULT_FACTOR = 3.0  # a hub is returned at least this many times more often than the median note
DEFAULT_MIN_RATE = 0.005  # ...and in at least this share of all queries
DEFAULT_LONG_WORDS = 3000

_DATE_IN_TITLE = re.compile(r"\s*\(\d{4}-\d{2}-\d{2}\)")


@dataclass(frozen=True)
class Query:
    text: str
    exclude: frozenset[str]  # notes this query is about: returning them is not "noise" (entries ending in "/" = folders)


@dataclass(frozen=True)
class NoiseCandidate:
    path: str
    appearances: int  # times it came back for queries that were not about it
    rate: float  # appearances / number of queries
    flags: tuple[str, ...]  # "long", "orphan", "unstructured"
    inbound: int
    outbound: int
    words: int
    chunks: int
    hints: tuple[str, ...]
    strength: str  # "strong" | "look"


def make_queries(data: IndexData, *, snippets_per_note: int = 2, snippet_words: int = 12, seed: int = 7,
                 extra: Iterable[tuple[str, Iterable[str]]] = ()) -> list[Query]:
    """Test queries: each note's title, a few sentences from its own text, and any extra
    (question, expected notes) pairs. The same data and seed always give the same queries."""
    chunks: dict[str, list[str]] = {}
    for c in sorted(data.chunks, key=lambda c: (c.path, c.index)):
        chunks.setdefault(c.path, []).append(c.text)
    queries: list[Query] = []
    for path in sorted(chunks):
        own = frozenset({path})
        title = _DATE_IN_TITLE.sub("", note_title(path)).strip()
        if title:
            queries.append(Query(title, own))
        rng = random.Random(f"{seed}|{path}")
        usable = [t.split() for t in chunks[path] if len(t.split()) >= snippet_words]
        for _ in range(snippets_per_note if usable else 0):
            words = rng.choice(usable)
            start = rng.randrange(0, len(words) - snippet_words + 1)
            queries.append(Query(" ".join(words[start:start + snippet_words]), own))
    queries.extend(Query(text, frozenset(expected)) for text, expected in extra)
    return queries


def count_appearances(search: Callable[[str], list[str]], queries: list[Query], k: int = 5) -> Counter:
    """How often each note is in the top k for queries that are not about it.
    `search` takes a query text and returns note paths, best first."""
    counts: Counter = Counter()
    for q in queries:
        for path in search(q.text)[:k]:
            if path in q.exclude or any(e.endswith("/") and path.startswith(e) for e in q.exclude):
                continue
            counts[path] += 1
    return counts


def find_noise(data: IndexData, counts: Counter, n_queries: int, *, factor: float = DEFAULT_FACTOR,
               min_rate: float = DEFAULT_MIN_RATE, long_words: int = DEFAULT_LONG_WORDS) -> list[NoiseCandidate]:
    """Hubs among the notes, strong candidates first. Empty list: nothing stands out."""
    if n_queries <= 0 or not counts:
        return []
    rates = {p: c / n_queries for p, c in counts.items() if c > 0}
    if not rates:
        return []
    threshold = max(min_rate, factor * statistics.median(rates.values()))
    hubs = [p for p, r in rates.items() if r >= threshold]
    if not hubs:
        return []

    texts: dict[str, list[str]] = {}
    headed: set[str] = set()
    chunk_count: Counter = Counter()
    for c in sorted(data.chunks, key=lambda c: (c.path, c.index)):
        texts.setdefault(c.path, []).append(c.text)
        chunk_count[c.path] += 1
        if c.heading_path:
            headed.add(c.path)
    paths = sorted(texts)
    raw = {p: "\n".join(texts[p]) for p in paths}
    inbound, outbound = link_counts(paths, raw)

    found = []
    for p in hubs:
        if p not in texts:
            continue  # the index changed since the counts were taken
        n_words = len(words_of(raw[p]))
        flags = tuple(f for f, on in (("long", n_words >= long_words),
                                      ("orphan", inbound[p] == 0 and outbound[p] == 0),
                                      ("unstructured", p not in headed)) if on)
        found.append(NoiseCandidate(
            path=p, appearances=counts[p], rate=rates[p], flags=flags, inbound=inbound[p],
            outbound=outbound[p], words=n_words, chunks=chunk_count[p], hints=path_hints(p),
            strength="strong" if len(flags) >= 2 else "look"))
    found.sort(key=lambda n: (n.strength != "strong", -len(n.flags), -n.rate, n.path))
    return found
