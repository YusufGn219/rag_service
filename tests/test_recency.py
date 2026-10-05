from datetime import date

import pytest

from rag_service.recency import freshness, is_recency_query, note_date


# ---- note_date ----

def test_date_in_parentheses_in_title():
    p = "Claude_Code/creative-arv/11 ACİL - İstek Sınırı (2026-09-25).md"
    assert note_date(p) == date(2026, 9, 25)


def test_date_anywhere_in_file_name_and_not_in_folders():
    assert note_date("a/2026-02-19 Gün 1 - (Perşembe).md") == date(2026, 2, 19)
    assert note_date("2026-01-01/not.md") is None


def test_no_or_impossible_date_gives_none():
    assert note_date("a/kedi.md") is None
    assert note_date("a/not (2026-13-45).md") is None
    assert note_date("a/not 12026-09-251.md") is None


def test_backslash_paths_are_understood():
    assert note_date("a\\b\\not (2026-09-25).md") == date(2026, 9, 25)


# ---- is_recency_query ----

@pytest.mark.parametrize("q", [
    "Creative arv projesinde güncel durumumuz nedir",
    "Tryess projesinde neredeyiz",
    "yapılması bekleyen acil işlerimiz",
    "şu an ne yapıyoruz",
    "ŞU AN durum",
    "en son ne yaptık",
    "bugün ne var",
    "GÜNCEL durum",
])
def test_recency_queries_detected(q):
    assert is_recency_query(q)


@pytest.mark.parametrize("q", [
    "sargonetik felsefesi hakkında ne düşünüyorsun",
    "iki faktörlü doğrulama nasıl kuruldu",
    "sonuç bölümü",  # 'sonuç' is not 'son'
    "yenilik algoritması",  # 'yenilik' is not 'yeni'
    "",
])
def test_ordinary_queries_not_flagged(q):
    assert not is_recency_query(q)


# ---- freshness ----

def test_freshness_is_one_today_and_halves_each_half_life():
    today = date(2026, 10, 5)
    assert freshness(today, today, half_life_days=90) == pytest.approx(1.0)
    assert freshness(date(2026, 7, 7), today, half_life_days=90) == pytest.approx(0.5)


def test_freshness_unknown_date_is_zero_and_future_is_capped():
    today = date(2026, 10, 5)
    assert freshness(None, today) == 0.0
    assert freshness(date(2027, 1, 1), today) == pytest.approx(1.0)
