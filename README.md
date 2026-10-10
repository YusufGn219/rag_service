# rag_service

Local hybrid search for a folder of Markdown notes (an Obsidian vault, for example).
It combines meaning-based search (embeddings) with keyword search (BM25), merges the two
rankings, and optionally re-reads the best candidates with a cross-encoder reranker.

Everything runs on your machine, on CPU. No cloud service, no API key for a model provider.

It can be used three ways:

- **Command line:** `python -m rag_service.search "your question"`
- **HTTP service:** a small FastAPI server on `127.0.0.1`
- **MCP tools for Claude Code:** `search_notes` and `read_note`, through a thin stdio bridge

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
| `POST /search` | body `{"query": "...", "k": 5}` |
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
(log: `<RAG_INDEX_DIR>/server.log`). It offers two tools:

- `search_notes(query, k=5)`
- `read_note(path)`

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
- **Claude Code için MCP araçları:** `search_notes` ve `read_note`

## Nasıl çalışır?

1. **İndeks:** Notlar yaklaşık 400-512 tokenlik, birbiriyle örtüşen parçalara bölünür. Her
   parça için bir embedding üretilir ve BM25 indeksine eklenir. Sonraki çalıştırmalarda
   yalnızca değişen notlar yeniden işlenir.
2. **Arama:** Soru iki indekste de aranır, iki sıralama RRF ile birleştirilir.
3. **Yeniden sıralama (isteğe bağlı):** En iyi adaylar bir cross-encoder ile yeniden
   puanlanır. Soru başına yaklaşık 2-3 saniye ekler, doğruluğu artırır.
4. **Güncellik:** "X'in son durumu ne" türü sorularda daha yeni tarihli notlar ve proje
   indeks notları öne çıkarılır.

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

Uçlar: `POST /search`, `GET /note?path=...`, `POST /reindex`, `GET /health`.
Servis kendi zamanlayıcısıyla indeksi günceller, boşta kalınca belleği boşaltır; açık
bırakılabilir.

Claude Code'a bağlamak için (bir kez, yolları `/` ile yaz):

```
claude mcp add rag --scope user -e PYTHONPATH=<proje> -- <proje>/.venv/Scripts/python.exe -m rag_service.bridge
```

`PYTHONPATH` şarttır, çünkü Claude Code köprüyü proje klasöründen başlatmaz. Köprü, ilk
araç çağrısında HTTP servisini arka planda kendisi başlatır. Kontrol için
`claude mcp get rag` çalıştır, sonra Claude Code'u yeniden başlat.

## Ek araçlar ve testler

- `python -m rag_service.evaluate`: kendi soru listene karşı arama kalitesini sayıyla
  ölçer. Soru listesi not adlarını içerdiği için gizli tutulmalıdır.
- `python -m rag_service.audit`: aramayı bozan kopya ve gürültülü notları bulur.
- Testler: `pip install -r requirements-dev.txt` sonra `pytest`.

## Gizlilik

İndeks, modeller, değerlendirme soruları ve denetim raporları `data/` altındadır ve
`.gitignore` içindedir, commit edilmez. `.env` dosyanı da commit etme.

## Lisans

[PolyForm Noncommercial License 1.0.0](LICENSE). Yazılımı kişisel, hobi, eğitim, araştırma ve
diğer ticari olmayan amaçlarla kullanabilir, kopyalayabilir ve değiştirebilirsin. Ticari
kullanıma bu lisansla izin verilmez. Bu, OSI onaylı bir açık kaynak lisansı değil, kaynağı
açık (source-available) bir lisanstır.
