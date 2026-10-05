"""Date awareness: read dates from note names and tell when a question is about 'now'."""
import re
from datetime import date

from rag_service.bm25 import tokenize

# Folded (lowercase, no diacritics) words that mean "I want the current state".
RECENCY_WORDS = frozenset({"guncel", "son", "simdi", "neredeyiz", "acil", "yeni", "bugun"})
RECENCY_PHRASES = (("su", "an"),)

DEFAULT_HALF_LIFE_DAYS = 90

_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")


def note_date(path: str) -> date | None:
    """The first valid YYYY-MM-DD in the file name (folders are ignored), else None."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    for y, m, d in _DATE.findall(name):
        try:
            return date(int(y), int(m), int(d))
        except ValueError:
            continue
    return None


def is_recency_query(query: str) -> bool:
    tokens = tokenize(query)
    if any(t in RECENCY_WORDS for t in tokens):
        return True
    pairs = list(zip(tokens, tokens[1:]))
    return any(phrase in pairs for phrase in RECENCY_PHRASES)


def freshness(d: date | None, today: date, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """1.0 for today (or the future), halving every half_life_days; 0.0 when the date is unknown."""
    if d is None:
        return 0.0
    age = max((today - d).days, 0)
    return 0.5 ** (age / half_life_days)
