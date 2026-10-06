"""Audit the index for copies and noise, and act on the answers.

Run it:  python -m rag_service.audit              report (copies + noise), decided items hidden
         python -m rag_service.audit keep ID...   "this stays": never shown again
         python -m rag_service.audit exclude ID... [--apply]   leave notes out of the index
         python -m rag_service.audit baseline     save search quality (hit@1/5, MRR) before a cleanup
         python -m rag_service.audit check        measure again and show what got better / worse

Nothing is removed on its own: `exclude` only edits RAG_EXCLUDE_DIRS in .env, and only with
--apply. Excluded notes leave the index at the next update (the service does it every 10
minutes, or POST /reindex); removing the entry from .env brings them back.
"""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from rag_service.config import Config, ConfigError
from rag_service.dupes import DupGroup
from rag_service.noise import NoiseCandidate
from rag_service.scanner import scan_notes

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DECISIONS_FILE = _PROJECT_ROOT / "data" / "audit_kararlar.json"
LAST_REPORT_FILE = _PROJECT_ROOT / "data" / "audit_son_rapor.json"
BASELINE_FILE = _PROJECT_ROOT / "data" / "audit_baseline.json"
REPORT_TEXT_FILE = _PROJECT_ROOT / "data" / "audit_rapor.txt"
ENV_FILE = _PROJECT_ROOT / ".env"
DEFAULT_LIMIT_LOOK = 20
_EXCLUDE_KEY = "RAG_EXCLUDE_DIRS"


class ExclusionError(RuntimeError):
    """An exclusion that cannot be done safely (the message says why)."""


# ---- decisions: what the user already answered ----

def _empty_decisions() -> dict:
    return {"kept_groups": {}, "kept_notes": {}, "excluded": {}}


def load_decisions(path: Path) -> dict:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_decisions()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not readable ({exc}); fix or delete it") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a decisions file; fix or delete it")
    return {k: dict(raw.get(k) or {}) for k in _empty_decisions()}


def save_decisions(path: Path, decisions: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(decisions, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def mark_kept(decisions: dict, *, group: str | None = None, note: str | None = None, today: date | None = None) -> None:
    stamp = (today or date.today()).isoformat()
    if group:
        decisions["kept_groups"][group] = stamp
    if note:
        decisions["kept_notes"][note] = stamp


def mark_excluded(decisions: dict, paths, today: date | None = None) -> None:
    stamp = (today or date.today()).isoformat()
    for p in paths:
        decisions["excluded"][p] = stamp


# ---- the report ----

def noise_id(path: str) -> str:
    return "n" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]


def _note_line(mark: str, n) -> str:
    when = f", {n.modified}" if n.modified else ""
    hint = f", ipucu: {'/'.join(n.hints)}" if n.hints else ""
    return f"   {mark:<6} {n.path}  ({n.words} kelime, {n.chunks} parça{when}, {n.inbound} bağlantı{hint})"


def format_report(groups: list[DupGroup], candidates: list[NoiseCandidate], decisions: dict, *,
                  limit_look: int = DEFAULT_LIMIT_LOOK, show_all: bool = False) -> tuple[str, dict]:
    """(text for the screen, machine-readable ids for keep/exclude). Decided items are left out."""
    excluded = set(decisions["excluded"])
    shown_groups = [g for g in groups if g.key not in decisions["kept_groups"]
                    and sum(1 for n in g.notes if n.path not in excluded) >= 2]
    shown_noise = [c for c in candidates if c.path not in decisions["kept_notes"] and c.path not in excluded]
    hidden_g, hidden_n = len(groups) - len(shown_groups), len(candidates) - len(shown_noise)

    strong = [c for c in shown_noise if c.strength == "strong"]
    look = [c for c in shown_noise if c.strength != "strong"]
    look_shown = look if show_all else look[:limit_look]

    data: dict = {"groups": {}, "noise": {}}
    lines: list[str] = []
    if shown_groups:
        lines.append(f"== KOPYA GRUPLARI ({len(shown_groups)}) ==")
        for g in shown_groups:
            data["groups"][g.key] = {"kind": g.kind, "detail": g.detail, "paths": [n.path for n in g.notes],
                                     "exclude": list(g.exclude)}
            lines.append(f"[{g.key}] {g.kind}: {g.detail}")
            for n in g.notes:
                mark = "ÇIKAR" if n.path in g.exclude else ("kalsın" if g.exclude else "bak")
                lines.append(_note_line(mark, n))
            if g.exclude:
                lines.append(f"   -> python -m rag_service.audit exclude {g.key}   |   audit keep {g.key}")
            else:
                lines.append(f"   -> öneri yok (elle bak). Karar: audit keep {g.key}   |   audit exclude --path <not yolu>")
            lines.append("")
    if strong or look_shown:
        lines.append(f"== GÜRÜLTÜ ADAYLARI ({len(strong)} güçlü, {len(look)} bak) ==")
        for c in strong + look_shown:
            nid = noise_id(c.path)
            data["noise"][nid] = {"path": c.path, "strength": c.strength}
            tag = "GÜÇLÜ" if c.strength == "strong" else "bak"
            extra = f", ipucu: {'/'.join(c.hints)}" if c.hints else ""
            lines.append(f"[{nid}] {tag:<5} %{c.rate * 100:.1f} ({c.appearances} kez)  {','.join(c.flags) or '-'}  "
                         f"{c.words} kelime, {c.inbound} gelen/{c.outbound} giden bağlantı{extra}")
            lines.append(f"   {c.path}")
            lines.append(f"   -> audit exclude {nid}   |   audit keep {nid}")
        if len(look) > len(look_shown):
            lines.append(f"(+{len(look) - len(look_shown)} daha 'bak' adayı; hepsi için --all)")
        lines.append("")
    if not data["groups"] and not data["noise"]:
        lines.append("Karar bekleyen kopya ya da gürültü adayı bulunamadı.")
    if hidden_g or hidden_n:
        lines.append(f"({hidden_g} kopya grubu, {hidden_n} gürültü adayı daha önce karara bağlandığı için gösterilmedi)")
    return "\n".join(lines).rstrip() + "\n", data


# ---- leaving notes out of the index ----

@dataclass(frozen=True)
class ExclusionPlan:
    entries: list[str]  # lines to add to RAG_EXCLUDE_DIRS
    removed: list[str]  # note keys that would leave the index
    extra: list[str]  # ...of which were not asked for (an entry also matched them)


def _norm(entry: str) -> str:
    return entry.replace("\\", "/").strip("/")


def entry_for_key(cfg: Config, key: str) -> str:
    """The RAG_EXCLUDE_DIRS entry that leaves exactly this note out: its path inside its vault root."""
    try:
        cfg.resolve_key(key)
    except ValueError as exc:
        raise ExclusionError(str(exc)) from exc
    rel = key if len(cfg.vault_roots) == 1 else key.partition("/")[2]
    rel = _norm(rel)
    if "/" not in rel:
        raise ExclusionError(
            f"{key!r} kök klasörde duruyor; böyle bir not yalnızca ad olarak yazılabilir ve aynı adlı "
            "her notu her yerde dışarıda bırakırdı. Notu bir alt klasöre taşıyın ya da elle karar verin.")
    if "," in rel:
        raise ExclusionError(f"{key!r} adında virgül var; RAG_EXCLUDE_DIRS virgülle ayrıldığı için yazılamaz")
    return rel


def plan_exclusion(cfg: Config, keys: list[str]) -> ExclusionPlan:
    existing = {_norm(e) for e in cfg.exclude_dirs}
    entries: list[str] = []
    for key in keys:
        entry = entry_for_key(cfg, key)
        if entry not in existing and entry not in entries:
            entries.append(entry)
    if not entries:
        return ExclusionPlan([], [], [])
    before = {cfg.note_key(p) for p in scan_notes(cfg)}
    after = {cfg.note_key(p) for p in scan_notes(replace(cfg, exclude_dirs=cfg.exclude_dirs + tuple(entries)))}
    removed = sorted(before - after)
    return ExclusionPlan(entries, removed, sorted(set(removed) - set(keys)))


def apply_env_exclusions(env_path: Path, cfg: Config, entries: list[str]) -> str:
    """Add entries to the RAG_EXCLUDE_DIRS line of .env (created with the current list if missing)."""
    if _EXCLUDE_KEY in os.environ:
        raise ExclusionError(f"{_EXCLUDE_KEY} bir ortam değişkeni olarak ayarlı; .env'e yazmak etkisiz kalır. "
                             "Önce ortam değişkenini kaldırın.")
    env_path = Path(env_path)
    raw = env_path.read_bytes().decode("utf-8") if env_path.exists() else ""
    nl = "\r\n" if "\r\n" in raw else "\n"
    current = list(cfg.exclude_dirs)
    known = {_norm(e) for e in current}
    new_line = f"{_EXCLUDE_KEY}=" + ",".join(current + [e for e in entries if _norm(e) not in known])
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(_EXCLUDE_KEY + "="):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    tmp = env_path.with_name(env_path.name + ".tmp")
    tmp.write_bytes((nl.join(lines) + nl).encode("utf-8"))
    tmp.replace(env_path)
    return new_line


# ---- measuring before / after ----

def eval_snapshot(searcher, questions, *, notes: int) -> dict:
    from rag_service.evaluate import run_eval

    report = run_eval(searcher, questions)
    return {"n": report.n, "notes": notes, "hit1": report.hit1, "hit5": report.hit5, "mrr": report.mrr,
            "ranks": {r.question.id: r.rank for r in report.results},
            "questions": {r.question.id: r.question.text for r in report.results}}


def _rank_text(rank) -> str:
    return "yok" if rank is None else f"#{rank}"


def compare_snapshots(old: dict, new: dict) -> str:
    lines = [f"İndeksteki not sayısı: {old['notes']} -> {new['notes']}",
             f"hit@1 {old['hit1']:.0%} -> {new['hit1']:.0%} | hit@5 {old['hit5']:.0%} -> {new['hit5']:.0%} | "
             f"MRR {old['mrr']:.3f} -> {new['mrr']:.3f}"]
    worse, better = [], []
    for qid, new_rank in new["ranks"].items():
        if qid not in old["ranks"]:
            continue
        old_rank = old["ranks"][qid]
        key_old = old_rank if old_rank is not None else 10**9
        key_new = new_rank if new_rank is not None else 10**9
        text = new["questions"].get(qid) or old["questions"].get(qid, "")
        line = f"   {qid}  {text}  : {_rank_text(old_rank)} -> {_rank_text(new_rank)}"
        if key_new > key_old:
            worse.append(line)
        elif key_new < key_old:
            better.append(line)
    if worse:
        lines += [f"KÖTÜLEŞEN ({len(worse)}):", *worse]
    if better:
        lines += [f"İYİLEŞEN ({len(better)}):", *better]
    if not worse and not better:
        lines.append("Sonuç: hiçbir sorunun sırası değişmedi.")
    return "\n".join(lines) + "\n"


# ---- command line ----

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag_service.audit", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd")
    rep = sub.add_parser("report", help="show copies and noise (default)")
    rep.add_argument("--all", action="store_true", help="show every 'bak' candidate, not just the first 20")
    rep.add_argument("--no-noise", action="store_true", help="skip the noise measurement (needs the models, ~30 s)")
    keep = sub.add_parser("keep", help="this stays; do not show it again")
    keep.add_argument("ids", nargs="+", help="group id, noise id, or a note path from the report")
    exc = sub.add_parser("exclude", help="leave notes out of the index (dry run unless --apply)")
    exc.add_argument("ids", nargs="*", help="group id (its suggested notes) or noise id")
    exc.add_argument("--path", action="append", default=[], help="a specific note path, as shown in the report")
    exc.add_argument("--apply", action="store_true", help="write the change to .env")
    sub.add_parser("baseline", help="save search quality before a cleanup")
    chk = sub.add_parser("check", help="measure again and compare with the baseline")
    chk.add_argument("--save", action="store_true", help="make this measurement the new baseline")
    return parser


def _load_last() -> dict:
    try:
        return json.loads(LAST_REPORT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit("error: önce `python -m rag_service.audit` ile rapor çıkarın (id'ler son rapordan okunur)")


def _cmd_report(args, cfg) -> int:
    from rag_service.dupes import find_duplicates
    from rag_service.manifest import load_manifest
    from rag_service.store import load_index

    data = load_index(cfg)
    if data is None:
        print("error: indeks yok; önce indeksleyin (python -m rag_service.indexer)", file=sys.stderr)
        return 1
    groups = find_duplicates(data, load_manifest(cfg))
    candidates: list[NoiseCandidate] = []
    if not args.no_noise:
        from rag_service.embeddings import Embedder
        from rag_service.evaluate import DEFAULT_FILE, load_questions
        from rag_service.noise import count_appearances, find_noise, make_queries
        from rag_service.search import Searcher

        searcher = Searcher.from_config(cfg, Embedder.from_dir(cfg.model_dir), reranker=None)
        extra = []
        if DEFAULT_FILE.exists():
            extra = [(q.text, q.expected) for q in load_questions(DEFAULT_FILE)]
        queries = make_queries(data, extra=extra)
        print(f"gürültü ölçülüyor ({len(queries)} sorgu)...", file=sys.stderr, flush=True)
        counts = count_appearances(lambda q: [n.path for n in searcher.search_notes(q, k=5)], queries)
        candidates = find_noise(data, counts, len(queries))
    text, machine = format_report(groups, candidates, load_decisions(DECISIONS_FILE), show_all=args.all)
    print(text)
    LAST_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_REPORT_FILE.write_text(json.dumps(machine, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT_TEXT_FILE.write_text(text, encoding="utf-8")
    return 0


def _cmd_keep(args) -> int:
    last, decisions = _load_last(), load_decisions(DECISIONS_FILE)
    for ident in args.ids:
        if ident in last["groups"]:
            mark_kept(decisions, group=ident)
        elif ident in last["noise"]:
            mark_kept(decisions, note=last["noise"][ident]["path"])
        elif "/" in ident or ident.lower().endswith(".md"):
            mark_kept(decisions, note=ident)
        else:
            print(f"error: {ident!r} son raporda yok", file=sys.stderr)
            return 1
        print(f"kalsın olarak kaydedildi: {ident}")
    save_decisions(DECISIONS_FILE, decisions)
    return 0


def _cmd_exclude(args, cfg) -> int:
    keys: list[str] = list(args.path)
    if args.ids:
        last = _load_last()
        for ident in args.ids:
            if ident in last["groups"]:
                suggested = last["groups"][ident]["exclude"]
                if not suggested:
                    print(f"error: {ident} grubunda öneri yok; --path ile hangi notu çıkaracağınızı yazın",
                          file=sys.stderr)
                    return 1
                keys += suggested
            elif ident in last["noise"]:
                keys.append(last["noise"][ident]["path"])
            else:
                print(f"error: {ident!r} son raporda yok", file=sys.stderr)
                return 1
    if not keys:
        print("error: çıkarılacak bir id ya da --path verin", file=sys.stderr)
        return 1
    try:
        plan = plan_exclusion(cfg, list(dict.fromkeys(keys)))
    except ExclusionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not plan.entries:
        print("Bunlar zaten dışarıda bırakılmış; yapılacak bir şey yok.")
        return 0
    if plan.extra:
        print("error: istenmeyen notlar da dışarıda kalırdı (aynı yol başka bir kökte de var):", file=sys.stderr)
        for k in plan.extra:
            print(f"   {k}", file=sys.stderr)
        return 1
    print(f"{_EXCLUDE_KEY} satırına eklenecek ({len(plan.entries)}):")
    for e in plan.entries:
        print(f"   {e}")
    print(f"İndeksten düşecek notlar ({len(plan.removed)}):")
    for k in plan.removed:
        print(f"   {k}")
    if not args.apply:
        print("\nKuru çalıştırma: hiçbir şey yazılmadı. Uygulamak için aynı komuta --apply ekleyin.")
        return 0
    try:
        apply_env_exclusions(ENV_FILE, cfg, plan.entries)
    except ExclusionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    decisions = load_decisions(DECISIONS_FILE)
    mark_excluded(decisions, plan.removed)
    save_decisions(DECISIONS_FILE, decisions)
    print("\n.env güncellendi. İndeks 10 dk içinde ya da `POST /reindex` ile güncellenir (servis açıksa).")
    print("Sonra `python -m rag_service.audit check` ile arama kalitesini kontrol edin.")
    return 0


def _snapshot(cfg) -> dict:
    from rag_service.embeddings import Embedder
    from rag_service.evaluate import DEFAULT_FILE, load_questions
    from rag_service.manifest import load_manifest
    from rag_service.search import Searcher

    searcher = Searcher.from_config(cfg, Embedder.from_dir(cfg.model_dir))
    return eval_snapshot(searcher, load_questions(DEFAULT_FILE), notes=len(load_manifest(cfg)))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"report", "keep", "exclude", "baseline", "check"}
    if not argv or argv[0] not in commands and argv[0] not in ("-h", "--help"):
        argv.insert(0, "report")
    args = _build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    from rag_service.config import load_config

    try:
        cfg = load_config()
        if args.cmd == "keep":
            return _cmd_keep(args)
        if args.cmd == "exclude":
            return _cmd_exclude(args, cfg)
        if args.cmd == "baseline":
            snap = _snapshot(cfg)
            BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            BASELINE_FILE.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"baseline kaydedildi: {snap['n']} soru, hit@1 {snap['hit1']:.0%}, hit@5 {snap['hit5']:.0%}, "
                  f"MRR {snap['mrr']:.3f}, {snap['notes']} not")
            return 0
        if args.cmd == "check":
            try:
                old = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                print("error: baseline yok; temizlikten önce `audit baseline` çalıştırın", file=sys.stderr)
                return 1
            new = _snapshot(cfg)
            print(compare_snapshots(old, new))
            if args.save:
                BASELINE_FILE.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
                print("yeni ölçüm baseline olarak kaydedildi")
            return 0
        return _cmd_report(args, cfg)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
