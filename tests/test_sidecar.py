import os

from rag_service import sidecar
from rag_service.chunker import parse_frontmatter


def _source(tmp_path, name="Taslak.pdf", data=b"abc"):
    f = tmp_path / name
    f.write_bytes(data)
    return f


def test_sidecar_path_keeps_the_extension(tmp_path):
    assert sidecar.sidecar_path(tmp_path / "Taslak.pdf") == tmp_path / "Taslak.pdf.md"


def test_render_header_and_tags_are_readable_by_the_chunker(tmp_path):
    src = _source(tmp_path)
    text = sidecar.render(src, "# Taslak\n\nmetin\n", "pdf", pages=4, hard=True)
    header = sidecar.read_header(text)
    assert header["uretici"] == "rag_service"
    assert header["kaynak"] == "Taslak.pdf"
    assert header["tur"] == "pdf" and header["sayfa"] == "4" and header["zor"] == "evet"
    tags, body = parse_frontmatter(text)
    assert tags == ["pdf", "kaynak-dosya"]
    assert body.lstrip().startswith("# Taslak")
    assert sidecar.is_ours(text)


def test_hand_written_note_is_not_ours():
    assert not sidecar.is_ours("---\ntags: [a]\n---\n\nmerhaba")
    assert not sidecar.is_ours("düz not")
    assert sidecar.read_header("düz not") == {}


def test_plan_new_current_refresh_foreign(tmp_path):
    src = _source(tmp_path)
    assert sidecar.plan(src) == sidecar.NEW

    target = sidecar.sidecar_path(src)
    target.write_text(sidecar.render(src, "# T\n", "pdf"), encoding="utf-8")
    assert sidecar.plan(src) == sidecar.CURRENT

    src.write_bytes(b"abc changed")  # size changes
    assert sidecar.plan(src) == sidecar.REFRESH

    target.write_text(sidecar.render(src, "# T\n", "pdf"), encoding="utf-8")
    assert sidecar.plan(src) == sidecar.CURRENT
    stat = src.stat()
    os.utime(src, (stat.st_atime, stat.st_mtime + 5))  # same size, newer time
    assert sidecar.plan(src) == sidecar.REFRESH

    target.write_text("---\ntags: [a]\n---\n\nelle yazılmış", encoding="utf-8")
    assert sidecar.plan(src) == sidecar.FOREIGN
