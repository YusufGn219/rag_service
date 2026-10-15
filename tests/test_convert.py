import os

import anyio
import pytest
from mcp import Client

from rag_service import convert, sidecar
from rag_service.bridge import create_bridge
from rag_service.config import load_config
from rag_service.errors import ServiceError


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    return root


@pytest.fixture
def cfg(vault, tmp_path):
    return load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "index")})


def _txt(folder, name="Not.txt", text="ilk satır\nikinci satır"):
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / name
    f.write_text(text, encoding="utf-8")
    return f


def test_convert_creates_the_note_next_to_the_source(vault, cfg):
    src = _txt(vault / "Okul")
    before = src.read_bytes()
    res = convert.convert_file(cfg, str(src))
    note = vault / "Okul" / "Not.txt.md"
    assert res["action"] == convert.WRITTEN and res["note"] == "Okul/Not.txt.md"
    assert note.exists() and "ikinci satır" in note.read_text(encoding="utf-8")
    assert sidecar.is_ours(note.read_text(encoding="utf-8"))
    assert src.read_bytes() == before  # the source is untouched


def test_relative_path_as_search_results_give_it(vault, cfg):
    _txt(vault / "Okul")
    assert convert.convert_file(cfg, "Okul/Not.txt")["action"] == convert.WRITTEN


def test_second_run_skips_and_a_changed_source_refreshes(vault, cfg):
    src = _txt(vault)
    convert.convert_file(cfg, str(src))
    again = convert.convert_file(cfg, str(src))
    assert again["action"] == convert.SKIPPED and "güncel" in again["reason"]

    src.write_text("tamamen yeni içerik burada", encoding="utf-8")
    res = convert.convert_file(cfg, str(src))
    assert res["action"] == convert.REFRESHED
    assert "tamamen yeni" in (vault / "Not.txt.md").read_text(encoding="utf-8")
    assert not list(vault.glob(".*.tmp"))  # no temp file left behind


def test_hand_written_note_is_never_touched(vault, cfg):
    src = _txt(vault)
    mine = vault / "Not.txt.md"
    mine.write_text("benim elle yazdığım not", encoding="utf-8")
    res = convert.convert_file(cfg, str(src))
    assert res["action"] == convert.SKIPPED
    # even an explicit markdown from an agent must not replace it
    res = convert.convert_file(cfg, str(src), markdown="# başka\n\nmetin")
    assert res["action"] == convert.SKIPPED
    assert mine.read_text(encoding="utf-8") == "benim elle yazdığım not"


def test_hard_file_is_not_written_until_text_is_supplied(vault, cfg):
    src = _txt(vault, "Boş.txt", text="  \n")
    res = convert.convert_file(cfg, str(src))
    assert res["action"] == convert.HARD and res["reason"]
    assert not (vault / "Boş.txt.md").exists()

    res = convert.convert_file(cfg, str(src), markdown="Görselden okunan metin.")
    assert res["action"] == convert.WRITTEN
    text = (vault / "Boş.txt.md").read_text(encoding="utf-8")
    assert "okuma: agent" in text and "# Boş" in text and "Görselden okunan metin." in text
    assert sidecar.is_ours(text)


def test_empty_markdown_is_refused(vault, cfg):
    src = _txt(vault)
    with pytest.raises(ServiceError):
        convert.convert_file(cfg, str(src), markdown="   ")


def test_paths_outside_the_vault_or_in_excluded_dirs_are_refused(vault, cfg, tmp_path):
    outside = _txt(tmp_path / "baska")
    with pytest.raises(ServiceError, match="vault root"):
        convert.convert_file(cfg, str(outside))
    hidden = _txt(vault / ".git")
    with pytest.raises(ServiceError, match="excluded"):
        convert.convert_file(cfg, str(hidden))
    assert not (tmp_path / "baska" / "Not.txt.md").exists()
    assert not (vault / ".git" / "Not.txt.md").exists()


def test_unsupported_missing_and_dotdot_are_refused(vault, cfg, tmp_path):
    md = vault / "a.md"
    md.write_text("x")
    with pytest.raises(ServiceError, match="not a supported"):
        convert.convert_file(cfg, str(md))
    with pytest.raises(ServiceError, match="not a file"):
        convert.convert_file(cfg, str(vault / "yok.txt"))
    outside = _txt(tmp_path / "baska")
    with pytest.raises(ServiceError, match="vault root"):
        convert.convert_file(cfg, str(vault / ".." / "baska" / "Not.txt"))
    assert outside.read_text(encoding="utf-8").startswith("ilk")


def test_too_large_file_is_refused(vault, cfg, monkeypatch):
    src = _txt(vault)
    monkeypatch.setattr(convert, "MAX_BYTES", 5)
    with pytest.raises(ServiceError, match="larger"):
        convert.convert_file(cfg, str(src))


def test_unreadable_file_reports_a_service_error(vault, cfg):
    bad = vault / "Bozuk.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(ServiceError):
        convert.convert_file(cfg, str(bad))
    assert not (vault / "Bozuk.pdf.md").exists()


def test_inspect_writes_nothing(vault, cfg):
    src = _txt(vault)
    info = convert.inspect_file(cfg, str(src))
    assert info["kind"] == "txt" and info["status"] == sidecar.NEW and not info["hard"]
    assert info["note"] == "Not.txt.md"
    assert not (vault / "Not.txt.md").exists()
    assert info["preview"] == "ilk satır ikinci satır"  # the "# Not" title line is not part of it
    assert info["modified"].endswith("+00:00") and info["modified"].startswith("20")


def test_inspect_preview_skips_page_markers_and_is_cut(vault, cfg):
    src = _txt(vault, "Uzun.txt", text="Başlık metni " + "kelime " * 200)
    info = convert.inspect_file(cfg, str(src))
    assert info["preview"].startswith("Başlık metni kelime")
    assert len(info["preview"]) <= convert.PREVIEW_CHARS + 1 and info["preview"].endswith("…")
    assert convert._preview("# Rapor\n\n## Sayfa 1\n\nmetin bir\n\n## Sayfa 2\n\nmetin iki\n") == "metin bir metin iki"
    assert convert._preview("# Boş\n") == ""


def test_scan_orders_actionable_first_and_counts(vault, cfg):
    a = _txt(vault / "a", "yeni.txt")
    b = _txt(vault / "b", "eski.txt")
    c = _txt(vault / "c", "guncel.txt")
    d = _txt(vault / "d", "elle.txt")
    convert.convert_file(cfg, str(b))
    convert.convert_file(cfg, str(c))
    b.write_text("değişti", encoding="utf-8")
    (vault / "d" / "elle.txt.md").write_text("elle", encoding="utf-8")
    _txt(vault / ".git", "x.txt")  # excluded
    (vault / "not.md").write_text("x")  # not a convertible type

    out = convert.scan(cfg)
    assert out["total"] == 4
    assert out["by_status"] == {"yeni": 1, "yenile": 1, "güncel": 1, "atla": 1}
    assert [f["status"] for f in out["files"]] == ["yeni", "yenile", "atla", "güncel"]
    assert out["files"][0]["path"] == "a/yeni.txt"

    sub = convert.scan(cfg, "a")
    assert sub["total"] == 1 and sub["files"][0]["path"] == "a/yeni.txt"

    short = convert.scan(cfg, limit=2)
    assert len(short["files"]) == 2 and short["truncated"]
    assert a.exists() and d.exists()


def test_status_has_counts_only(vault, cfg):
    _txt(vault)
    out = convert.status(cfg)
    assert out == {"total": 1, "by_status": {"yeni": 1}, "by_kind": {"txt": 1}}


def test_scan_rejects_a_file_as_folder(vault, cfg):
    src = _txt(vault)
    with pytest.raises(ServiceError, match="not a folder"):
        convert.scan(cfg, str(src))


def test_converted_docx_note_is_picked_up_by_the_scanner(vault, cfg):
    import docx

    d = docx.Document()
    d.add_paragraph("Merhaba dünya")
    f = vault / "Belge.docx"
    d.save(str(f))
    convert.convert_file(cfg, str(f))
    from rag_service.scanner import scan_notes

    assert [p.name for p in scan_notes(cfg)] == ["Belge.docx.md"]


def test_mtime_change_alone_triggers_refresh(vault, cfg):
    src = _txt(vault)
    convert.convert_file(cfg, str(src))
    st = src.stat()
    os.utime(src, (st.st_atime, st.st_mtime + 10))
    assert convert.convert_file(cfg, str(src))["action"] == convert.REFRESHED


# ------------------------------------------------------------------ the MCP tools


def _run(fn):
    anyio.run(fn)


class _NoClient:
    pass


def test_bridge_lists_convert_tools_only_with_a_config(cfg):
    async def go():
        async with Client(create_bridge(_NoClient())) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert "convert_file" not in names
        async with Client(create_bridge(_NoClient(), cfg)) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"inspect_file", "convert_file", "scan_files", "convert_status"} <= names

    _run(go)


def test_bridge_convert_tools_work_and_errors_reach_the_caller(vault, cfg):
    src = _txt(vault)

    async def go():
        async with Client(create_bridge(_NoClient(), cfg)) as c:
            res = await c.call_tool("convert_file", {"path": str(src)})
            assert not res.is_error and "yazıldı" in str(res.content)
            res = await c.call_tool("scan_files", {})
            assert not res.is_error and "Not.txt" in str(res.content)
            res = await c.call_tool("convert_status", {})
            assert not res.is_error
            res = await c.call_tool("convert_file", {"path": str(vault / "yok.txt")})
            assert res.is_error and "not a file" in str(res.content)

    _run(go)
    assert (vault / "Not.txt.md").exists()
