"""Turn PDF, DOCX, XLSX, PPTX, CSV, HTML and TXT files into markdown text.

`extract(path)` never writes anything. It returns the markdown body plus a verdict:
`hard=True` means the file gave little or no usable text (a scanned PDF, an empty sheet,
an unreadable file) and a person or an agent should look at it.
"""
import csv
import io
from dataclasses import dataclass
from pathlib import Path

MAX_SHEET_ROWS = 2000  # rows kept per spreadsheet sheet; the rest is dropped with a note
MIN_CHARS_PER_PAGE = 30  # a PDF with fewer characters per page than this counts as scanned
_MAX_HEADING = 6
_SHORT_LINE = 20  # a PDF page whose median line is shorter than this is reflowed into paragraphs

SUPPORTED = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
    ".csv": "csv",
    ".tsv": "csv",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
}


class ReadError(RuntimeError):
    """The file could not be opened or parsed."""


@dataclass(frozen=True)
class Extracted:
    markdown: str  # body only; no frontmatter, starts with "# <file name>"
    kind: str  # "pdf", "docx", ...
    pages: int | None = None  # pages / slides / sheets, when the format has them
    hard: bool = False
    reason: str | None = None  # why it is hard (Turkish, shown to the user)


def kind_of(path: Path) -> str | None:
    """The reader kind for a file name, or None when the extension is not supported."""
    return SUPPORTED.get(Path(path).suffix.lower())


# ---------------------------------------------------------------- helpers


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1254"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return " ".join(str(value).replace("|", "\\|").split())


def md_table(rows: list[list]) -> str:
    """A markdown table; the first row is the header. Empty when there are no usable rows."""
    rows = [[_cell(c) for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    while width > 1 and not any(len(r) >= width and r[width - 1] for r in rows):
        width -= 1  # drop trailing empty columns
    rows = [(r + [""] * width)[:width] for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _heading(level: int, text: str) -> str:
    # The note title is "#", so document headings start one level down.
    return "#" * min(max(level, 1) + 1, _MAX_HEADING) + " " + text


def _join(blocks: list[str]) -> str:
    return "\n\n".join(b for b in blocks if b and b.strip())


# ---------------------------------------------------------------- readers


def _reflow(text: str) -> str:
    """Some PDFs come out one word per line (with blank lines between). When the lines are that
    short, glue them back into one running paragraph; otherwise keep the text as it is."""
    lines = [ln.strip() for ln in text.split("\n")]
    lengths = sorted(len(ln) for ln in lines if ln)
    if not lengths or lengths[len(lengths) // 2] >= _SHORT_LINE:
        return text
    return " ".join(ln for ln in lines if ln)


def _read_pdf(path: Path) -> tuple[str, int, str | None]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            return "", 0, "PDF şifreli"
        texts = [_reflow((page.extract_text() or "").strip()) for page in reader.pages]
    except PdfReadError as exc:
        raise ReadError(f"PDF okunamadı: {exc}") from exc
    pages = len(texts)
    blocks = [f"## Sayfa {i}\n\n{t}" for i, t in enumerate(texts, 1) if t]
    chars = sum(len(t) for t in texts)
    reason = None
    if pages and chars < MIN_CHARS_PER_PAGE * pages:
        reason = "metin yok ya da çok az (taranmış PDF olabilir)"
    return _join(blocks), pages, reason


def _docx_paragraph(par) -> str:
    text = par.text.strip()
    if not text:
        return ""
    style = (par.style.name if par.style is not None else "") or ""
    low = style.lower()
    if low == "title":
        return _heading(1, text)
    for prefix in ("heading ", "başlık "):
        if low.startswith(prefix) and low[len(prefix):].isdigit():
            return _heading(int(low[len(prefix):]), text)
    if "list" in low or "liste" in low:
        return "- " + text
    return text


def _read_docx(path: Path) -> tuple[str, int, str | None]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(str(path))
    except Exception as exc:  # python-docx raises several unrelated types on bad files
        raise ReadError(f"DOCX okunamadı: {exc}") from exc
    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            blocks.append(_docx_paragraph(Paragraph(child, document)))
        elif tag == "tbl":
            table = Table(child, document)
            blocks.append(md_table([[c.text for c in row.cells] for row in table.rows]))
    return _join(blocks), None, None


def _read_xlsx(path: Path) -> tuple[str, int, str | None]:
    import openpyxl

    try:
        book = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:
        raise ReadError(f"XLSX okunamadı: {exc}") from exc
    blocks: list[str] = []
    try:
        for sheet in book.worksheets:
            rows: list[list] = []
            cut = False
            for row in sheet.iter_rows(values_only=True):
                if len(rows) >= MAX_SHEET_ROWS:
                    cut = True
                    break
                rows.append(list(row))
            table = md_table(rows)
            if not table:
                continue
            note = f"\n\n*(İlk {MAX_SHEET_ROWS} satır alındı, gerisi kesildi.)*" if cut else ""
            blocks.append(f"## {sheet.title}\n\n{table}{note}")
        count = len(book.worksheets)
    finally:
        book.close()
    return _join(blocks), count, None


def _read_pptx(path: Path) -> tuple[str, int, str | None]:
    from pptx import Presentation

    try:
        deck = Presentation(str(path))
    except Exception as exc:
        raise ReadError(f"PPTX okunamadı: {exc}") from exc
    blocks: list[str] = []
    count = 0
    for number, slide in enumerate(deck.slides, 1):
        count = number
        title_shape = slide.shapes.title
        title = title_shape.text.strip() if title_shape is not None and title_shape.has_text_frame else ""
        parts: list[str] = []
        for shape in slide.shapes:
            if shape is title_shape:
                continue
            if shape.has_text_frame:
                lines = [p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()]
                parts.append("\n".join("- " + line for line in lines))
            elif getattr(shape, "has_table", False) and shape.has_table:
                parts.append(md_table([[c.text for c in row.cells] for row in shape.table.rows]))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append("Not: " + notes)
        body = _join(parts)
        if title or body:
            blocks.append(f"## Slayt {number}" + (f": {title}" if title else "") + ("\n\n" + body if body else ""))
    return _join(blocks), count, None


def _read_csv(path: Path) -> tuple[str, int, str | None]:
    text = _decode(path.read_bytes())
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = []
    for row in csv.reader(io.StringIO(text), dialect):
        if len(rows) >= MAX_SHEET_ROWS:
            rows.append([f"(İlk {MAX_SHEET_ROWS} satır alındı, gerisi kesildi.)"])
            break
        rows.append(row)
    return md_table(rows), None, None


def _read_txt(path: Path) -> tuple[str, int, str | None]:
    return _decode(path.read_bytes()).replace("\r\n", "\n").replace("\r", "\n").strip(), None, None


_HTML_DROP = ("script", "style", "noscript", "svg", "template", "nav", "footer", "head")
_HTML_INLINE = frozenset({
    "a", "b", "i", "em", "strong", "span", "u", "small", "sub", "sup", "code", "abbr", "mark",
    "label", "font", "s", "del", "ins", "cite", "q", "kbd", "time", "br", "img",
})
_HTML_SKIP_STRINGS = ("Comment", "Doctype", "Declaration", "CData", "ProcessingInstruction")


def _html_blocks(node, out: list[str]) -> None:
    """Append the markdown blocks of `node`'s children. Text and inline tags between block
    tags are collected into one paragraph, so `<p>a <b>b</b></p>` stays "a b"."""
    from bs4 import NavigableString, Tag

    run: list[str] = []

    def flush() -> None:
        text = " ".join("".join(run).split())
        run.clear()
        if text:
            out.append(text)

    for child in node.children:
        if isinstance(child, NavigableString):
            if type(child).__name__ not in _HTML_SKIP_STRINGS:
                run.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in _HTML_DROP:
            continue
        if name in _HTML_INLINE:
            run.append(child.get_text() if name != "br" else " ")
            continue
        flush()
        if len(name) == 2 and name[0] == "h" and name[1].isdigit():
            text = " ".join(child.get_text().split())
            if text:
                out.append(_heading(int(name[1]), text))
        elif name == "table":
            rows = [[c.get_text(" ") for c in tr.find_all(["th", "td"])] for tr in child.find_all("tr")]
            out.append(md_table(rows))
        elif name in ("ul", "ol"):
            items = ["- " + " ".join(li.get_text().split()) for li in child.find_all("li")]
            out.append("\n".join(i for i in items if i != "- "))
        elif name == "pre":
            out.append("```\n" + child.get_text().strip("\n") + "\n```")
        else:  # p, div, section, ... and anything unknown: a container
            _html_blocks(child, out)
    flush()


def _read_html(path: Path) -> tuple[str, int, str | None]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(path.read_bytes()), "html.parser")
    root = soup.body or soup
    out: list[str] = []
    _html_blocks(root, out)
    title = " ".join(soup.title.get_text().split()) if soup.title else ""
    blocks = ([_heading(1, title)] if title else []) + out
    return _join(blocks), None, None


_READERS = {
    "pdf": _read_pdf,
    "docx": _read_docx,
    "xlsx": _read_xlsx,
    "pptx": _read_pptx,
    "csv": _read_csv,
    "html": _read_html,
    "txt": _read_txt,
}


def extract(path: Path) -> Extracted:
    """Read one file and return its markdown. Raises ReadError when the file cannot be parsed
    and ValueError when the extension is not supported."""
    path = Path(path)
    kind = kind_of(path)
    if kind is None:
        raise ValueError(f"desteklenmeyen dosya türü: {path.suffix or path.name}")
    try:
        body, pages, reason = _READERS[kind](path)
    except ReadError:
        raise
    except OSError as exc:
        raise ReadError(f"dosya okunamadı: {exc}") from exc
    if not body.strip() and reason is None:
        reason = "dosyadan metin çıkmadı"
    markdown = f"# {path.stem}\n\n{body}\n" if body.strip() else f"# {path.stem}\n"
    return Extracted(markdown=markdown, kind=kind, pages=pages, hard=reason is not None, reason=reason)
