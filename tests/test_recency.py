from datetime import date

import pytest

from rag_service.recency import (
    folder_units,
    freshness,
    is_recency_query,
    is_state_query,
    is_status_note,
    note_date,
)


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


# ---- is_state_query ("where are we") ----

@pytest.mark.parametrize("q", [
    "Tryess projesinde neredeyiz",
    "neredeyim ben",
    "Creative arv projesinde güncel durumumuz nedir",
    "vitapuls projesinin güncel durumu",
    "Tryess için son durum nedir",
    "Tryess'te şu an hangi işler bekliyor",
    "Chef.LLM'de en son ne yaptık",
    "Aegis'te en son hangi değişikliği yaptık",
    "EN SON NELER OLDU",
])
def test_state_queries_detected(q):
    assert is_state_query(q)


@pytest.mark.parametrize("q", [
    "Son olarak, videoda hep siyah kare çıkıyordu",
    "En son, videoda hep siyah kare çıkıyordu",  # "en son" only counts before a question word
    "Güncel Docker sürümü nedir",  # "güncel" alone is a specific question too
    "yeni eklenen özellikler neydi",
    "yapılması bekleyen acil işlerimiz",
    "durumu özetle",  # "durum" needs a recency word with it
    "Docker konteyner durumu nasıl kontrol edilir",
    "en son",
    "sonuç bölümü",
    "",
])
def test_specific_or_plain_queries_are_not_state_queries(q):
    assert not is_state_query(q)


def test_a_state_query_is_always_a_recency_query_except_durum_without_a_recency_word():
    for q in ["Tryess projesinde neredeyiz", "güncel durum", "en son ne yaptık", "şu an ne durumda"]:
        assert is_state_query(q) and is_recency_query(q)


# ---- is_status_note / folder_units ----

@pytest.mark.parametrize("path", [
    "Claude_Code/ar-kiyafet/ar-kiyafet - İndeks.md",
    "Claude_Code/x/Proje Index.md",
    "Claude_Code/ar-kiyafet/37 Devam Adımları ve Ertelenenler (2026-10-04).md",
    "Claude_Code/creative-arv/04 Güncel Durum ve Devam Notu (2026-07-15).md",
    "a/Güncel Durum.md",
])
def test_status_notes(path):
    assert is_status_note(path)


@pytest.mark.parametrize("path", [
    "Claude_Code/ar-kiyafet/03 Backend Yeniden Yapılanma Planı.md",
    "a/Durum Raporu.md",  # "durum" alone is not enough
    "a/indeks_klasoru/not.md",  # only the file name counts
    "a/Güncel Sürümler.md",
])
def test_ordinary_notes_are_not_status_notes(path):
    assert not is_status_note(path)


def test_folder_units_are_the_lowercased_folders_without_the_file_name():
    assert folder_units("Claude_Code/Ar-Kiyafet/37 Not.md") == frozenset({"claude_code", "ar-kiyafet"})
    assert folder_units("a\\B\\not.md") == frozenset({"a", "b"})
    assert folder_units("not.md") == frozenset()


# ---- freshness ----

def test_freshness_is_one_today_and_halves_each_half_life():
    today = date(2026, 10, 5)
    assert freshness(today, today, half_life_days=90) == pytest.approx(1.0)
    assert freshness(date(2026, 7, 7), today, half_life_days=90) == pytest.approx(0.5)


def test_freshness_unknown_date_is_zero_and_future_is_capped():
    today = date(2026, 10, 5)
    assert freshness(None, today) == 0.0
    assert freshness(date(2027, 1, 1), today) == pytest.approx(1.0)
