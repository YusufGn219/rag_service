"""Turn text into unit-length vectors with an ONNX embedding model (E5 family)."""
from pathlib import Path

import numpy as np

MODEL_REPO = "intfloat/multilingual-e5-small"
_ONNX_FILE = "onnx/model.onnx"
_TOKENIZER_FILE = "onnx/tokenizer.json"

# E5 models expect these prefixes; leaving them out silently lowers quality.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

_MODEL_NAME = "model.onnx"
_TOKENIZER_NAME = "tokenizer.json"


def download_model(dest_dir: Path) -> Path:
    """Fetch the ONNX model + tokenizer (~490 MB) into dest_dir. Only runs when called."""
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for remote, local in ((_ONNX_FILE, _MODEL_NAME), (_TOKENIZER_FILE, _TOKENIZER_NAME)):
        cached = hf_hub_download(MODEL_REPO, remote)
        (dest_dir / local).write_bytes(Path(cached).read_bytes())
    return dest_dir


class Embedder:
    def __init__(self, session, tokenizer, batch_size: int = 32, max_length: int = 512, pad_id=None):
        self._session = session
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self._pad_id = pad_id if pad_id is not None else (tokenizer.token_to_id("<pad>") or 0)

        names = {i.name for i in session.get_inputs()}
        if not {"input_ids", "attention_mask"} <= names:
            raise ValueError(f"model must take input_ids and attention_mask, got {sorted(names)}")
        self._uses_token_type = "token_type_ids" in names

    @classmethod
    def from_dir(cls, model_dir: Path, **kwargs) -> "Embedder":
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        model_path, tok_path = model_dir / _MODEL_NAME, model_dir / _TOKENIZER_NAME
        if not model_path.is_file() or not tok_path.is_file():
            raise FileNotFoundError(f"model files missing in {model_dir} (run download_model first)")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return cls(session, Tokenizer.from_file(str(tok_path)), **kwargs)

    # -- public API ---------------------------------------------------------

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + text])[0]

    def count_tokens(self, text: str) -> int:
        """Token count of text alone (no special tokens, no prefix); pass to chunk_note."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Real token count of each passage as the model would see it, before truncation."""
        encs = self._tokenizer.encode_batch([PASSAGE_PREFIX + t for t in texts])
        return [len(e.ids) for e in encs]

    # -- internals ----------------------------------------------------------

    def _ids(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text).ids
        if len(ids) > self._max_length:
            ids = ids[: self._max_length - 1] + [ids[-1]]  # keep the closing special token
        return ids

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        ids = [self._ids(t) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: len(ids[i]))  # similar lengths per batch
        out: list[np.ndarray | None] = [None] * len(texts)
        for start in range(0, len(order), self._batch_size):
            idx = order[start:start + self._batch_size]
            vecs = self._run_batch([ids[i] for i in idx])
            for i, v in zip(idx, vecs):
                out[i] = v
        return np.stack(out).astype(np.float32)

    def _run_batch(self, batch: list[list[int]]) -> np.ndarray:
        width = max(len(b) for b in batch)
        input_ids = np.full((len(batch), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=np.int64)
        for row, seq in enumerate(batch):
            input_ids[row, : len(seq)] = seq
            mask[row, : len(seq)] = 1
        feeds = {"input_ids": input_ids, "attention_mask": mask}
        if self._uses_token_type:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        hidden = self._session.run(None, feeds)[0]
        return _mean_pool_normalize(hidden, mask)


def _mean_pool_normalize(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None].astype(np.float32)
    pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norm, 1e-12)
