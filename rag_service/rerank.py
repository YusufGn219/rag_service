"""Re-score search candidates with a cross-encoder: the question and a passage are read together.

Unlike the embedding model, which turns each text into a vector on its own, a cross-encoder
sees both at once, so it can tell "mentions the topic" from "answers the question". It is
slower, so it only runs on the few dozen candidates the first stage found.
"""
from pathlib import Path

import numpy as np

_MODEL_NAME = "model.onnx"
_TOKENIZER_NAME = "tokenizer.json"


class Reranker:
    def __init__(self, session, tokenizer, batch_size: int = 8, max_length: int = 512):
        self._session = session
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
        pad = next((i for i in map(tokenizer.token_to_id, ("<pad>", "[PAD]")) if i is not None), 0)
        self._pad_id = pad

        names = {i.name for i in session.get_inputs()}
        if not {"input_ids", "attention_mask"} <= names:
            raise ValueError(f"model must take input_ids and attention_mask, got {sorted(names)}")
        self._uses_token_type = "token_type_ids" in names

    @classmethod
    def from_dir(cls, model_dir: Path, **kwargs) -> "Reranker":
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        model_path, tok_path = model_dir / _MODEL_NAME, model_dir / _TOKENIZER_NAME
        if not model_path.is_file() or not tok_path.is_file():
            raise FileNotFoundError(f"reranker model files missing in {model_dir}")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return cls(session, Tokenizer.from_file(str(tok_path)), **kwargs)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage, in input order. Higher = better; only the order matters."""
        if not passages:
            return []
        encs = [self._tokenizer.encode(query, p) for p in passages]
        order = sorted(range(len(passages)), key=lambda i: len(encs[i].ids))  # similar lengths together
        out = [0.0] * len(passages)
        for start in range(0, len(order), self._batch_size):
            idx = order[start:start + self._batch_size]
            logits = self._run_batch([encs[i] for i in idx])
            for i, value in zip(idx, logits):
                out[i] = float(value)
        return out

    def _run_batch(self, encs) -> np.ndarray:
        width = max(len(e.ids) for e in encs)
        input_ids = np.full((len(encs), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(encs), width), dtype=np.int64)
        types = np.zeros((len(encs), width), dtype=np.int64)
        for row, e in enumerate(encs):
            n = len(e.ids)
            input_ids[row, :n] = e.ids
            mask[row, :n] = 1
            types[row, :n] = e.type_ids
        feeds = {"input_ids": input_ids, "attention_mask": mask}
        if self._uses_token_type:
            feeds["token_type_ids"] = types
        logits = np.asarray(self._session.run(None, feeds)[0])
        return logits.reshape(len(encs), -1)[:, 0]


def load_reranker(cfg) -> "Reranker | None":
    """The reranker named by the config, or None when reranking is off.

    Off means: RAG_RERANK_DIR is empty, or it is unset and the default folder has no model.
    A folder the user named explicitly must hold a model, otherwise this raises.
    """
    if cfg.rerank_dir is None:
        return None
    present = (cfg.rerank_dir / _MODEL_NAME).is_file() and (cfg.rerank_dir / _TOKENIZER_NAME).is_file()
    if not present:
        if cfg.rerank_required:
            raise FileNotFoundError(f"reranker model files missing in {cfg.rerank_dir}")
        return None
    return Reranker.from_dir(cfg.rerank_dir)
