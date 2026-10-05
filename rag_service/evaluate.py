"""Measure search quality: how often does the right note come out on top?

Questions live in a JSONL file (default data/eval/questions.jsonl), one per line:
    {"id": "optional", "question": "...", "expected": ["Folder/Note.md", "Folder/Sub/"]}
An expected entry ending in "/" accepts any note under that folder. Questions with no
expected notes are listed but not scored.

Run it:  python -m rag_service.evaluate
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "data" / "eval" / "questions.jsonl"
DEFAULT_K = 10


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    expected: tuple[str, ...]


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    rank: int | None  # 1-based position of the first right note; None = not in the top k
    top: tuple[str, ...]  # the notes that came back, best first


@dataclass(frozen=True)
class Report:
    results: tuple[QuestionResult, ...]  # scored questions only
    unlabeled: tuple[Question, ...]

    @property
    def n(self) -> int:
        return len(self.results)

    def _rate(self, ok) -> float:
        return sum(1 for r in self.results if r.rank is not None and ok(r.rank)) / self.n if self.n else 0.0

    @property
    def hit1(self) -> float:
        return self._rate(lambda rank: rank == 1)

    @property
    def hit5(self) -> float:
        return self._rate(lambda rank: rank <= 5)

    @property
    def mrr(self) -> float:
        """Mean of 1/rank (0 for a miss): 1.0 = always first, 0.5 = typically second."""
        if not self.n:
            return 0.0
        return sum(1 / r.rank for r in self.results if r.rank is not None) / self.n


def matches(path: str, expected: tuple[str, ...]) -> bool:
    return any(path.startswith(e) if e.endswith("/") else path == e for e in expected)


def load_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {lineno} is not valid JSON ({exc.msg})") from exc
        text = row.get("question") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: line {lineno} has no 'question' text")
        expected = row.get("expected") or []
        if isinstance(expected, str):
            expected = [expected]
        questions.append(Question(row.get("id") or f"q{len(questions) + 1}", text, tuple(expected)))
    return questions


def add_question(path: Path, text: str, expected: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"question": text, "expected": list(expected)}, ensure_ascii=False) + "\n")


def run_eval(searcher, questions: list[Question], *, k: int = DEFAULT_K, **search_kwargs) -> Report:
    results: list[QuestionResult] = []
    unlabeled: list[Question] = []
    for q in questions:
        if not q.expected:
            unlabeled.append(q)
            continue
        top = tuple(n.path for n in searcher.search_notes(q.text, k=k, **search_kwargs))
        rank = next((i for i, p in enumerate(top, start=1) if matches(p, q.expected)), None)
        results.append(QuestionResult(q, rank, top))
    return Report(tuple(results), tuple(unlabeled))


def format_report(report: Report) -> str:
    lines = []
    for r in report.results:
        mark = f"#{r.rank}" if r.rank is not None else "MISS"
        lines.append(f"{mark:>5}  {r.question.id}  {r.question.text}")
        if r.rank != 1:
            for p in r.top[:3]:
                lines.append(f"         got: {p}")
    for q in report.unlabeled:
        lines.append(f"  n/a  {q.id}  {q.text}  (no expected note yet)")
    lines.append("")
    lines.append(
        f"scored {report.n} | hit@1 {report.hit1:.0%} | hit@5 {report.hit5:.0%} | MRR {report.mrr:.3f}"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag_service.evaluate")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--mode", choices=("hybrid", "dense", "bm25"), default="hybrid")
    parser.add_argument("--recency", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--add", metavar="QUESTION", help="append a question to the file and exit")
    parser.add_argument("--expect", nargs="*", default=[], help="with --add: the right note(s)")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if args.add:
        add_question(args.file, args.add, args.expect)
        print(f"added to {args.file}")
        return 0

    from rag_service.config import ConfigError, load_config
    from rag_service.embeddings import Embedder
    from rag_service.search import Searcher

    try:
        questions = load_questions(args.file)
        cfg = load_config()
        searcher = Searcher.from_config(cfg, Embedder.from_dir(cfg.model_dir))
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    recency = {"auto": None, "on": True, "off": False}[args.recency]
    print(format_report(run_eval(searcher, questions, k=args.k, mode=args.mode, recency=recency)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
