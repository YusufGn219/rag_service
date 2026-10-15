# rag_service

Local hybrid search for a folder of Markdown notes (an Obsidian vault, for example).
It combines meaning-based search (embeddings) with keyword search (BM25), merges the two
rankings, and optionally re-reads the best candidates with a cross-encoder reranker.

Everything runs on your machine, on CPU. No cloud service, no API key for a model provider.

It can be used three ways:

- **Command line:** `python -m rag_service.search "your question"`
- **HTTP service:** a small FastAPI server on `127.0.0.1`
- **MCP tools for Claude Code:** `search_notes`, `read_section` and `read_note`, through a thin stdio bridge. The same bridge has tools that turn PDF, Word, Excel, PowerPoint, CSV, HTML and text files into searchable notes (see [Other file types](#other-file-types))

## How it works

1. **Index:** notes are split into overlapping chunks of about 400-512 tokens. Each chunk
   gets an embedding ([multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small),
   ONNX) and goes into a BM25 index. Only changed notes are re-embedded on later runs.
2. **Search:** the question is searched in both indexes and the two rankings are merged with
   Reciprocal Rank Fusion (RRF).
3. **Rerank (optional):** the top candidates are re-scored by a cross-encoder
   ([mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)).
   This adds roughly 2-3 seconds per question and improves accuracy.
4. **Recency:** for "what is the current state of X" type questions, newer dated notes and
   project index notes are ranked higher.

Notes are multilingual; the author uses it with Turkish and English notes.

## Other file types

PDF, Word (`.docx`), Excel (`.xlsx`), PowerPoint (`.pptx`), CSV/TSV, HTML and plain text files
can be made searchable too. The index only reads Markdown, so such a file is first converted to
a note that is written **next to the original**: `report.pdf` becomes `report.pdf.md` in the same
folder. Links and embeds that point at the original keep working, and the normal indexing run
picks the new note up.

- Each converted note starts with a small header (`uretici: rag_service`, the source name, and
  the source's size and modified time). A later run uses it to tell whether the note is up to date.
- A converted note is refreshed when its source changes. A note you wrote yourself is never
  touched, even if it has the same name. The original file is never changed.
- Files are only read and written inside the folders of `RAG_VAULT_ROOT`, and never inside
  excluded folders.
- A file that gives no usable text (a scanned PDF, an empty sheet) is not written. It is reported
  as hard, and an agent can read it and supply the text (see below).

Limits: no OCR; a spreadsheet sheet is cut after 2000 rows; files over 200 MB are refused; the
old `.doc`, `.xls` and `.ppt` formats are not supported.

The conversion is offered as MCP tools only (see [Claude Code (MCP)](#claude-code-mcp)); there is no
command-line command for it.

## Requirements

- Python 3.10 or newer
- About 500 MB of disk for the embedding model, about 135 MB more for the reranker
- About 1.4 GB of RAM while the models are loaded (about 0.1 GB after they are unloaded)

## Install

```
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Models

The embedding model is downloaded once (about 490 MB):

```
python -c "from pathlib import Path; from rag_service.embeddings import download_model; download_model(Path('data/models/multilingual-e5-small'))"
```

The reranker is optional. If `data/models/reranker/` does not contain a model, search
simply runs without reranking. To enable it, save these two files into `data/models/reranker/`:

- `onnx/model_quint8_avx2.onnx` as `model.onnx`
- `tokenizer.json`

both from `https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/resolve/main/<file>`.

### Configure

Copy `.env.example` to `.env` and set at least the folder that holds your notes:

```
RAG_VAULT_ROOT=/path/to/your/notes
```

Settings are read from environment variables, and from `.env` for anything the environment
lacks. `.env.example` documents every option (in English and Turkish):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_VAULT_ROOT` | required | Notes folder. Several folders: separate with `;` (Windows) or `:` (Linux/macOS) |
| `RAG_INDEX_DIR` | `data/index` | Where the index is stored. Must be outside the notes folder |
| `RAG_MODEL_DIR` | `data/models/multilingual-e5-small` | Embedding model folder |
| `RAG_RERANK_DIR` | `data/models/reranker` | Reranker folder. Empty value turns reranking off |
| `RAG_EXCLUDE_DIRS` | `.git,.obsidian,.trash` | Folders or files to skip. Setting it replaces the default |
| `RAG_PORT` | `2190` | HTTP service port (listens on `127.0.0.1` only) |
| `RAG_IDLE_MINUTES` | `10` | Unload models after this many idle minutes (`0` = never) |
| `RAG_REINDEX_MINUTES` | `10` | Update the index this often (`0` = never) |
| `RAG_API_KEY` | none | If set, HTTP requests must send it as `X-API-Key` or `Authorization: Bearer ...` |
| `RAG_PARENT_CHARS` | `3000` | Longest parent section text returned (by `read_section`, or by `/search` with `include_parent`); longer ones are cut (`0` = parent sections off) |

## Use

### Command line

Build or update the index, then search:

```
python -m rag_service.indexer
python -m rag_service.search "what did we decide about the database?"
```

Useful options of `search`: `-k N` (number of results), `--notes` (one result per note),
`--mode hybrid|dense|bm25`, `--rerank auto|off`, `--recency auto|on|off`, `--full`.

### HTTP service

```
python -m rag_service.server            # --port N overrides RAG_PORT
```

| Endpoint | Purpose |
|---|---|
| `POST /search` | body `{"query": "...", "k": 5}`; each result has `parent` `{id, section, chars}`, the wider section its best passage sits in (`null` if it adds nothing). Add `"include_parent": true` to get the section text in it too (about 2.5x bigger answers) |
| `GET /section?path=...&id=...` | text of that parent section (`path` and `parent.id` from a search result), cut after `RAG_PARENT_CHARS` |
| `GET /note?path=...` | full text of a note, `path` as returned by search |
| `POST /reindex` | update the index now |
| `GET /health` | status, version, process id |

The service reindexes on its own timer and frees memory when idle, so it can stay running.

### Claude Code (MCP)

Register the bridge once. Use forward slashes in the paths:

```
claude mcp add rag --scope user -e PYTHONPATH=<project> -- <project>/.venv/Scripts/python.exe -m rag_service.bridge
```

`PYTHONPATH` is required, because Claude Code does not start the bridge from the project
folder. The bridge starts the HTTP service in the background on the first tool call
(log: `<RAG_INDEX_DIR>/server.log`). It offers three search tools:

- `search_notes(query, k=5)`
- `read_section(path, id)`: the wider section of a result (its `parent.id`), shorter than the whole note
- `read_note(path)`

and four tools that convert files into notes (see [Other file types](#other-file-types)). `path` is
absolute, or relative to a vault root as in search results:

- `inspect_file(path)`: look at a file without writing anything: type, size, pages, whether text comes out
- `convert_file(path, markdown=None)`: write the note next to the file. With `markdown`, that text is
  saved instead (for a file that gave no text on its own)
- `scan_files(folder=None, limit=200)`: list convertible files with their state
- `convert_status()`: counts by state and by type

The states are `yeni` (not converted yet), `yenile` (the source changed), `güncel` (up to date) and
`atla` (a hand-written note has that name). `convert_file` answers `yazıldı` (written), `yenilendi`
(refreshed), `atlandı` (skipped) or `zor` (hard, not written).

The conversion agent in `agents/dosya-cevirici.md` (written in Turkish) uses these tools. It scans a
folder, skips personal documents, old versions and code templates, converts the rest and reads
scanned PDFs itself. To use it in Claude Code, copy the file to `.claude/agents/` in a project, or to
`~/.claude/agents/` for all projects. It is plain Markdown with a small header, so its text can be
reused with other agent frameworks.

Check it with `claude mcp get rag`, then restart Claude Code.

## Extra tools

- `python -m rag_service.evaluate` measures search quality against your own list of
  questions and the notes that should answer them (`--add "question" --expect note.md`
  adds one). Keep that list private, it contains your note names.
- `python -m rag_service.audit` finds duplicate and noisy notes that hurt search results,
  and can add them to `RAG_EXCLUDE_DIRS`.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

The tests use fake models and do not need the downloads.

## Privacy

The index, models, evaluation questions and audit reports live under `data/` and are
listed in `.gitignore`, so they are not committed. Do not commit your `.env`.

Converted notes are new files inside your notes folders. If one of those folders is a git
repository you publish, review the converted notes before committing them.

## Status

Built for personal use and working well there. Search quality depends on your notes, so
measure it on your own with `python -m rag_service.evaluate`.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE). You may use, copy and modify the software for
personal, hobby, educational, research and other non-commercial purposes. Commercial use is not
permitted under this license. Note that this is a source-available license, not an OSI-approved
open source license.

---

# Türkçe

## rag_service nedir?

Markdown notlarından oluşan bir klasör (örneğin bir Obsidian vault'u) için yerel çalışan
hibrit arama servisi. Anlama dayalı aramayı (embedding) anahtar kelime aramasıyla (BM25)
birleştirir, iki sıralamayı RRF ile tek listeye indirir ve istersen en iyi adayları bir
yeniden sıralayıcıyla (reranker) bir kez daha okuyup puanlar.

Her şey kendi bilgisayarında, işlemci (CPU) üzerinde çalışır. Bulut servisi ya da model
sağlayıcısı anahtarı gerekmez. Türkçe ve İngilizce notlarla birlikte kullanılabilir.

Üç şekilde kullanılır:

- **Komut satırı:** `python -m rag_service.search "sorun"`
- **HTTP servisi:** `127.0.0.1` üzerinde küçük bir FastAPI sunucusu
- **Claude Code için MCP araçları:** `search_notes`, `read_section` ve `read_note`. Aynı köprüde PDF, Word, Excel, PowerPoint, CSV, HTML ve metin dosyalarını aranabilir nota çeviren araçlar da var ([Diğer dosya türleri](#diğer-dosya-türleri))

## Nasıl çalışır?

1. **İndeks:** Notlar yaklaşık 400-512 tokenlik, birbiriyle örtüşen parçalara bölünür. Her
   parça için bir embedding üretilir ve BM25 indeksine eklenir. Sonraki çalıştırmalarda
   yalnızca değişen notlar yeniden işlenir.
2. **Arama:** Soru iki indekste de aranır, iki sıralama RRF ile birleştirilir.
3. **Yeniden sıralama (isteğe bağlı):** En iyi adaylar bir cross-encoder ile yeniden
   puanlanır. Soru başına yaklaşık 2-3 saniye ekler, doğruluğu artırır.
4. **Güncellik:** "X'in son durumu ne" türü sorularda daha yeni tarihli notlar ve proje
   indeks notları öne çıkarılır.

## Diğer dosya türleri

PDF, Word (`.docx`), Excel (`.xlsx`), PowerPoint (`.pptx`), CSV/TSV, HTML ve düz metin dosyaları da aranabilir hale getirilebilir. 
İndeks yalnızca Markdown okuduğu için böyle bir dosya önce **orijinalin yanına** yazılan bir nota çevrilir: `rapor.pdf`, aynı klasörde `rapor.pdf.md` olur.
Orijinale giden bağlantılar ve gömülü yollar bozulmaz, normal indeksleme turu yeni notu kendisi alır.

- Her çevrilmiş not küçük bir künyeyle başlar (`uretici: rag_service`, kaynağın adı, boyutu ve
  değişim zamanı). Sonraki çalıştırma notun güncel olup olmadığını buradan anlar.
- Kaynak değişince çevrilmiş not yenilenir. Senin elle yazdığın bir nota, aynı ada sahip olsa
  bile, dokunulmaz. Orijinal dosya hiçbir zaman değiştirilmez.
- Dosyalar yalnızca `RAG_VAULT_ROOT` klasörlerinin içinde okunur ve yazılır, hariç tutulan
  klasörlere hiç girilmez.
- Kullanılabilir metin vermeyen dosya (taranmış PDF, boş sayfa) yazılmaz, "zor" diye bildirilir;
  bir agent onu okuyup metni kendisi verebilir (aşağıya bak).

Sınırlar: OCR yok; bir tablo sayfası 2000 satırdan sonra kesilir; 200 MB'tan büyük dosya
reddedilir; eski `.doc`, `.xls` ve `.ppt` biçimleri desteklenmez.

Çevirme yalnızca MCP araçlarıyla yapılır (aşağıdaki Claude Code bölümüne bak); komut satırı
karşılığı yoktur.

## Gereksinimler

- Python 3.10 veya üstü
- Embedding modeli için yaklaşık 500 MB, reranker için yaklaşık 135 MB disk
- Modeller yüklüyken yaklaşık 1,4 GB RAM (boşaltılınca yaklaşık 0,1 GB)

## Kurulum

```
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Embedding modelini bir kez indir (yaklaşık 490 MB):

```
python -c "from pathlib import Path; from rag_service.embeddings import download_model; download_model(Path('data/models/multilingual-e5-small'))"
```

Reranker isteğe bağlıdır. `data/models/reranker/` klasöründe model yoksa arama
yeniden sıralama olmadan çalışır. Açmak için şu iki dosyayı o klasöre kaydet:

- `onnx/model_quint8_avx2.onnx` dosyasını `model.onnx` adıyla
- `tokenizer.json`

İkisi de `https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/resolve/main/<dosya>`
adresinden indirilir.

## Ayarlar

`.env.example` dosyasını `.env` olarak kopyala ve en az notlarının klasörünü yaz:

```
RAG_VAULT_ROOT=/notlarinin/yolu
```

Tüm seçenekler İngilizce ve Türkçe açıklamalarıyla `.env.example` içinde, özet tablo da
yukarıdaki "Configure" bölümünde var.

## Kullanım

Komut satırı:

```
python -m rag_service.indexer
python -m rag_service.search "veritabanı hakkında ne karar vermiştik?"
```

HTTP servisi:

```
python -m rag_service.server            # --port N, RAG_PORT'u ezer
```

Uçlar: `POST /search`, `GET /section?path=...&id=...`, `GET /note?path=...`, `POST /reindex`, `GET /health`.
Her arama sonucundaki `parent` (`id`, `section`, `chars`), eşleşen parçanın bağlı olduğu daha geniş bölümü
söyler; metni `/section` ile çekilir. `/search` isteğine `"include_parent": true` eklenirse metin sonuca
da girer (yanıt yaklaşık 2,5 kat büyür).
Servis kendi zamanlayıcısıyla indeksi günceller, boşta kalınca belleği boşaltır; açık
bırakılabilir.

Claude Code'a bağlamak için (bir kez, yolları `/` ile yaz):

```
claude mcp add rag --scope user -e PYTHONPATH=<proje> -- <proje>/.venv/Scripts/python.exe -m rag_service.bridge
```

`PYTHONPATH` şarttır, çünkü Claude Code köprüyü proje klasöründen başlatmaz. Köprü, ilk
araç çağrısında HTTP servisini arka planda kendisi başlatır. Kontrol için
`claude mcp get rag` çalıştır, sonra Claude Code'u yeniden başlat.

Arama araçlarına ek olarak dosyaları nota çeviren dört araç gelir. `path` mutlak yol ya da arama
sonuçlarındaki gibi vault köküne göreli yol olabilir:

- `inspect_file(path)`: hiçbir şey yazmadan dosyaya bakar (tür, boyut, sayfa sayısı, metin çıkıp çıkmadığı)
- `convert_file(path, markdown=None)`: notu dosyanın yanına yazar. `markdown` verilirse okuyucunun
  çıktısı yerine o metin kaydedilir (kendi başına metin vermeyen dosyalar için)
- `scan_files(folder=None, limit=200)`: çevrilebilir dosyaları durumlarıyla listeler
- `convert_status()`: duruma ve türe göre sayılar

Durumlar: `yeni` (henüz çevrilmedi), `yenile` (kaynak değişti), `güncel`, `atla` (aynı adlı elle
yazılmış not var). `convert_file` şunlardan birini söyler: `yazıldı`, `yenilendi`, `atlandı`,
`zor` (yazılmadı).

`agents/dosya-cevirici.md` (Türkçe yazılmış) bu araçları kullanan çeviri agent'ıdır. Bir klasörü
tarar, kişisel belgeleri, eski sürümleri ve kod şablonlarını atlar, kalanını çevirir, taranmış
PDF'leri kendisi okur. Claude Code'da kullanmak için dosyayı bir projedeki `.claude/agents/`
klasörüne ya da tüm projeler için `~/.claude/agents/` klasörüne kopyala. Düz Markdown ve küçük bir
başlıktan oluştuğu için metni başka agent çerçevelerinde de kullanılabilir.

## Ek araçlar ve testler

- `python -m rag_service.evaluate`: kendi soru listene karşı arama kalitesini sayıyla
  ölçer. Soru listesi not adlarını içerdiği için gizli tutulmalıdır.
- `python -m rag_service.audit`: aramayı bozan kopya ve gürültülü notları bulur.
- Testler: `pip install -r requirements-dev.txt` sonra `pytest`.

## Gizlilik

İndeks, modeller, değerlendirme soruları ve denetim raporları `data/` altındadır ve
`.gitignore` içindedir, commit edilmez. `.env` dosyanı da commit etme.

Çevrilmiş notlar, notlarının klasörlerinde oluşan yeni dosyalardır. Bu klasörlerden biri
yayınladığın bir git deposuysa, commit etmeden önce çevrilmiş notlara bak.

## Lisans

[PolyForm Noncommercial License 1.0.0](LICENSE). Yazılımı kişisel, hobi, eğitim, araştırma ve
diğer ticari olmayan amaçlarla kullanabilir, kopyalayabilir ve değiştirebilirsin. Ticari
kullanıma bu lisansla izin verilmez. Bu, OSI onaylı bir açık kaynak lisansı değil, kaynağı
açık (source-available) bir lisanstır.
