"""Date awareness: read dates from note names and tell when a question is about 'now'."""
import re
from datetime import date

from rag_service.bm25 import tokenize
from rag_service.chunker import note_title

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


# "Where are we" questions are a narrower kind of recency question: the answer is the note that
# sums a project up, not just any new note. Kept narrow on purpose: "güncel" or "en son" alone
# also appear in specific questions ("güncel sürüm", "en son videoda ...").
#   - "neredeyiz" (and its endings: neredeyim ...)
#   - "durum" (and endings: durumu, durumumuz) together with a recency word: "güncel durum", "son durum"
#   - "şu an"
#   - "en son" followed by a question word: "en son ne yaptık", "en son hangi ..."
STATE_ASK_WORDS = frozenset({"ne", "hangi", "neler", "nelerdi", "neydi"})


def is_state_query(query: str) -> bool:
    tokens = tokenize(query)
    if any(t.startswith("neredey") for t in tokens):
        return True
    if any(t.startswith("durum") for t in tokens) and is_recency_query(query):
        return True
    for a, b, *rest in zip(tokens, tokens[1:], tokens[2:] + [""]):
        if (a, b) == ("su", "an") or ((a, b) == ("en", "son") and rest and rest[0] in STATE_ASK_WORDS):
            return True
    return False


def freshness(d: date | None, today: date, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """1.0 for today (or the future), halving every half_life_days; 0.0 when the date is unknown."""
    if d is None:
        return 0.0
    age = max((today - d).days, 0)
    return 0.5 ** (age / half_life_days)


# Folded title words of notes that sum up where a project stands ("X - İndeks", "Devam Notu").
STATUS_WORDS = frozenset({"indeks", "index", "devam"})


def is_status_note(path: str) -> bool:
    """True for notes whose title says they describe the current state: an index, a
    "Devam ..." (continue) note, or "Güncel Durum"."""
    words = set(tokenize(note_title(path)))
    return bool(words & STATUS_WORDS) or {"guncel", "durum"} <= words


def folder_units(path: str) -> frozenset[str]:
    """The folder names above a note, lowercased (the note's file name is not included)."""
    return frozenset(part.casefold() for part in path.replace("\\", "/").split("/")[:-1] if part)
