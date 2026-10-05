from rag_service.chunker import chunk_note, count_tokens, parse_frontmatter


def _words(n, word="kelime"):
    return " ".join([word] * n)


# ---- frontmatter ----

def test_frontmatter_inline_tags_removed_from_body():
    text = "---\ntags: [proje, mimari]\ndurum: x\n---\n# Baslik\nicerik"
    tags, body = parse_frontmatter(text)
    assert tags == ["proje", "mimari"]
    assert body.startswith("# Baslik")
    assert "durum" not in body


def test_frontmatter_dash_list_tags():
    tags, _ = parse_frontmatter("---\ntags:\n  - a\n  - b\n---\nbody")
    assert tags == ["a", "b"]


def test_frontmatter_comma_tags_and_crlf():
    tags, body = parse_frontmatter("---\r\ntags: a, b\r\n---\r\nbody")
    assert tags == ["a", "b"]
    assert body == "body"


def test_no_frontmatter_returns_text_unchanged():
    assert parse_frontmatter("# Hi\ntext") == ([], "# Hi\ntext")


def test_unclosed_frontmatter_is_not_stripped():
    tags, body = parse_frontmatter("---\ntags: [a]\nno closing")
    assert tags == []
    assert body.startswith("---")


# ---- counting ----

def test_count_tokens_grows_with_words():
    assert count_tokens("") == 0
    assert count_tokens(_words(100)) > count_tokens(_words(10)) > 0


# ---- chunking ----

def test_empty_note_has_no_chunks():
    assert chunk_note("a.md", "") == []
    assert chunk_note("a.md", "---\ntags: [x]\n---\n\n   \n") == []


def test_short_note_single_chunk_with_metadata():
    chunks = chunk_note("v/a.md", "---\ntags: [x, y]\n---\n# Baslik\nmerhaba dunya")
    assert len(chunks) == 1
    c = chunks[0]
    assert c.path == "v/a.md"
    assert c.heading_path == "Baslik"
    assert c.tags == ("x", "y")
    assert "merhaba dunya" in c.text
    assert "tags" not in c.text


def test_sections_split_by_headings_with_nested_path():
    a, b, c = _words(80, "alfa"), _words(80, "beta"), _words(80, "gama")
    text = f"# Ana\n{a}\n## Alt\n{b}\n# Ikinci\n{c}"
    chunks = chunk_note("n.md", text, max_tokens=400, min_tokens=20)
    assert [x.heading_path for x in chunks] == ["Ana", "Ana > Alt", "Ikinci"]
    assert "beta" in chunks[1].text and "alfa" not in chunks[1].text


def test_heading_inside_code_fence_is_not_a_heading():
    text = "# Ana\n" + _words(60) + "\n```\n# yorum satiri\n```\n" + _words(60)
    chunks = chunk_note("n.md", text, min_tokens=20)
    assert [x.heading_path for x in chunks] == ["Ana"]


def test_text_before_first_heading_has_empty_path():
    chunks = chunk_note("n.md", _words(80) + "\n# Sonra\n" + _words(80), min_tokens=20)
    assert chunks[0].heading_path == ""
    assert chunks[1].heading_path == "Sonra"


def test_tiny_sections_merged_with_next():
    text = "# A\nkisa\n# B\n" + _words(100)
    chunks = chunk_note("n.md", text, min_tokens=60)
    assert len(chunks) == 1
    assert "kisa" in chunks[0].text and "kelime" in chunks[0].text


def test_trailing_tiny_section_merged_with_previous():
    text = "# A\n" + _words(100) + "\n# B\nkisa"
    chunks = chunk_note("n.md", text, min_tokens=60)
    assert len(chunks) == 1
    assert "kisa" in chunks[0].text


def test_long_section_split_on_paragraphs_within_limit():
    paras = [_words(100, f"p{i}x") for i in range(8)]
    text = "# Uzun\n" + "\n\n".join(paras)
    chunks = chunk_note("n.md", text, max_tokens=400, overlap_tokens=0, min_tokens=20)
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= 400 for c in chunks)
    assert all(c.heading_path == "Uzun" for c in chunks)
    joined = " ".join(c.text for c in chunks)
    for i in range(8):
        assert f"p{i}x" in joined


def test_overlap_repeats_trailing_paragraph():
    paras = [_words(60, f"p{i}x") for i in range(10)]
    text = "# Uzun\n" + "\n\n".join(paras)
    chunks = chunk_note("n.md", text, max_tokens=250, overlap_tokens=120, min_tokens=20)
    assert len(chunks) > 1
    # last paragraph of chunk 0 appears again at the start of chunk 1
    last_word_0 = chunks[0].text.split()[-1]
    assert last_word_0 in chunks[1].text.split()[:70]


def test_single_giant_paragraph_is_windowed():
    text = "# G\n" + _words(2000)
    chunks = chunk_note("n.md", text, max_tokens=400, overlap_tokens=40)
    assert len(chunks) > 3
    assert all(count_tokens(c.text) <= 400 for c in chunks)


def test_chunk_indexes_are_sequential_per_note():
    text = "# G\n" + _words(2000)
    chunks = chunk_note("n.md", text, max_tokens=400)
    assert [c.index for c in chunks] == list(range(len(chunks)))


# ---- inline tags + embedding text ----

def test_inline_tags_collected_and_merged_with_frontmatter():
    text = "---\ntags: [proje]\n---\n# A\nBu bir #mimari notu, ayrica #proje ve #ar-kiyafet/v2.\n" + _words(60)
    chunks = chunk_note("n.md", text, min_tokens=20)
    assert chunks[0].tags == ("proje", "mimari", "ar-kiyafet/v2")


def test_inline_tag_ignores_headings_urls_code_and_fences():
    text = (
        "# Baslik\n"
        "link https://x.com/#anchor ve `#fff` renk\n"
        "```\n#yorum\n```\n"
        "& #sayfa1 degil mi? C# dili\n" + _words(60)
    )
    chunks = chunk_note("n.md", text, min_tokens=20)
    assert chunks[0].tags == ("sayfa1",)


def test_tags_apply_to_every_chunk_of_the_note():
    text = "#etiket\n# A\n" + _words(80) + "\n# B\n" + _words(80)
    chunks = chunk_note("n.md", text, min_tokens=20)
    assert len(chunks) == 2
    assert all(c.tags == ("etiket",) for c in chunks)


def test_note_without_tags_is_fine():
    chunks = chunk_note("n.md", "# A\n" + _words(80))
    assert chunks[0].tags == ()


def test_embed_text_contains_tags_heading_and_body():
    text = "---\ntags: [proje, mimari]\n---\n# Ana\n## Alt\n" + _words(80, "icerik")
    c = chunk_note("n.md", text)[0]
    assert "proje" in c.embed_text and "mimari" in c.embed_text
    assert "Ana > Alt" in c.embed_text
    assert "icerik" in c.embed_text
    assert c.text in c.embed_text


def test_embed_text_without_tags_or_heading_is_note_title_plus_text():
    c = chunk_note("n.md", "sadece duz metin burada")[0]
    assert c.embed_text == "Note: n\n" + c.text


def test_embed_text_contains_note_title_and_folder():
    path = "Claude_Code/creative-arv/04 Güncel Durum (2026-07-15).md"
    c = chunk_note(path, "# Ozet\n" + _words(80))[0]
    assert "Note: 04 Güncel Durum (2026-07-15)\n" in c.embed_text
    assert "Folder: Claude_Code > creative-arv\n" in c.embed_text
    assert ".md" not in c.embed_text


def test_embed_text_has_no_folder_line_for_top_level_note():
    c = chunk_note("n.md", "# A\n" + _words(80))[0]
    assert "Folder:" not in c.embed_text


def test_long_note_path_still_fits_budget():
    path = "klasor_xxxxxxxxxx/alt_klasor_xxxxxxxxxx/Cok Uzun Bir Not Adi (2026-09-25).md"
    text = "# Baslik\n" + "\n\n".join(_words(30, f"w{i}") for i in range(12))
    chunks = chunk_note(path, text, **_kw())
    assert len(chunks) > 1
    assert all(len(c.embed_text) <= 200 for c in chunks)


# ---- custom (real-tokenizer-style) counter ----

def _kw(**over):
    kw = dict(max_tokens=200, overlap_tokens=20, min_tokens=20, count=len)
    kw.update(over)
    return kw


def test_every_embed_text_fits_with_custom_counter():
    text = (
        "---\ntags: [alfa, beta, gama, delta]\n---\n"
        "# Ana Baslik\n## Alt Baslik Uzun Bir Isim\n"
        + "\n\n".join(_words(30, f"w{i}") for i in range(12))
        + "\n```\n" + "kod_satiri_uzun " * 40 + "\n```\n"
        + "9" * 1500
    )
    chunks = chunk_note("n.md", text, **_kw())
    assert len(chunks) > 5
    assert all(len(c.embed_text) <= 200 for c in chunks)


def test_unbroken_run_split_without_loss():
    run = "x" * 1000
    chunks = chunk_note("n.md", run, **_kw(overlap_tokens=0))
    assert len(chunks) >= 5
    assert all(len(c.text) <= 200 for c in chunks)
    assert "".join(c.text for c in chunks) == run


def test_no_word_dropped_with_custom_counter():
    words = [f"w{i:03d}" for i in range(300)]
    text = "# T\n" + "\n\n".join(" ".join(words[i:i + 25]) for i in range(0, 300, 25))
    chunks = chunk_note("n.md", text, **_kw())
    seen = set(" ".join(c.text for c in chunks).split())
    assert set(words) <= seen


def test_heavy_header_shrinks_body_budget_but_still_fits():
    tags = ", ".join(f"etiket{i}" for i in range(10))
    body = "# Baslik\n" + _words(120, "kelime")
    with_tags = chunk_note("n.md", f"---\ntags: [{tags}]\n---\n{body}", **_kw())
    without = chunk_note("n.md", body, **_kw())
    assert all(len(c.embed_text) <= 200 for c in with_tags)
    assert len(with_tags) >= len(without)


def test_window_overlap_with_custom_counter():
    words = [f"w{i:03d}" for i in range(200)]
    chunks = chunk_note("n.md", " ".join(words), **_kw(overlap_tokens=30))
    assert len(chunks) > 1
    first_tail = chunks[0].text.split()[-1]
    assert first_tail in chunks[1].text.split()


def test_default_counter_still_estimates_by_words():
    chunks = chunk_note("n.md", "# A\n" + _words(2000), max_tokens=400)
    assert all(count_tokens(c.text) <= 400 for c in chunks)


def test_overlong_line_starting_with_hashes_is_body_not_heading():
    text = "## " + " ".join(f"w{i}" for i in range(200))  # whole note flattened onto one line
    chunks = chunk_note("n.md", text, **_kw(max_tokens=300))
    assert all(c.heading_path == "" for c in chunks)
    assert all(len(c.embed_text) <= 300 for c in chunks)
    assert {f"w{i}" for i in range(200)} <= set(" ".join(c.text for c in chunks).split())


def test_normal_long_title_still_a_heading():
    title = "Bu oldukca uzun ama gercek bir baslik metnidir"
    chunks = chunk_note("n.md", f"# {title}\n" + _words(80))
    assert chunks[0].heading_path == title
