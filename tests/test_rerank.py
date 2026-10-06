import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from rag_service.rerank import Reranker

_WORDS = ["<unk>", "<pad>", "a", "b", "c", "d", "e", "f"]
_VOCAB = {w: i for i, w in enumerate(_WORDS)}
A = _VOCAB["a"]


def _tokenizer():
    tok = Tokenizer(models.WordLevel(_VOCAB, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


class _Input:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Logit of a pair = how many 'a' tokens it contains (padding excluded via the mask)."""

    def __init__(self, token_type=False, flat=False):
        self.names = ["input_ids", "attention_mask"] + (["token_type_ids"] if token_type else [])
        self.flat = flat
        self.calls = []

    def get_inputs(self):
        return [_Input(n) for n in self.names]

    def run(self, _outputs, feeds):
        self.calls.append(feeds)
        hits = ((feeds["input_ids"] == A) & (feeds["attention_mask"] == 1)).sum(axis=1)
        logits = hits.astype(np.float32)
        return [logits if self.flat else logits[:, None]]


def _rr(**kw):
    sess = FakeSession(token_type=kw.pop("token_type", False), flat=kw.pop("flat", False))
    return Reranker(sess, _tokenizer(), **kw), sess


def test_one_score_per_passage_in_input_order():
    rr, _ = _rr(batch_size=2)
    scores = rr.score("b", ["a a a", "c", "a", "a a", "d e f"])
    assert scores == [3.0, 0.0, 1.0, 2.0, 0.0]


def test_query_tokens_count_too():
    rr, _ = _rr()
    assert rr.score("a", ["b"]) == [1.0]  # the 'a' came from the query


def test_empty_passage_list():
    rr, sess = _rr()
    assert rr.score("a", []) == []
    assert sess.calls == []


def test_padding_does_not_change_scores():
    rr, sess = _rr(batch_size=8)
    alone = rr.score("b", ["a"])[0]
    mixed = rr.score("b", ["a", "c " * 30])[0]
    assert alone == mixed == 1.0
    assert sess.calls[-1]["input_ids"].shape[0] == 2


def test_flat_logit_output_is_accepted():
    rr, _ = _rr(flat=True)
    assert rr.score("b", ["a a", "c"]) == [2.0, 0.0]


def test_long_passage_is_truncated_to_max_length_but_query_survives():
    rr, sess = _rr(max_length=16)
    rr.score("a", ["b " * 100])
    assert sess.calls[0]["input_ids"].shape[1] <= 16
    assert A in sess.calls[0]["input_ids"][0]  # the query is not cut off


def test_token_type_ids_only_sent_when_model_wants_them():
    rr, sess = _rr(token_type=True)
    rr.score("a", ["b"])
    assert "token_type_ids" in sess.calls[0]
    rr2, sess2 = _rr()
    rr2.score("a", ["b"])
    assert "token_type_ids" not in sess2.calls[0]


def test_model_without_required_inputs_rejected():
    class Bad(FakeSession):
        def get_inputs(self):
            return [_Input("x")]

    with pytest.raises(ValueError, match="input_ids"):
        Reranker(Bad(), _tokenizer())


def test_from_dir_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="model"):
        Reranker.from_dir(tmp_path)


# ---- loading from config ----

def _cfg(tmp_path, **extra):
    from rag_service.config import load_config

    vault = tmp_path / "v"
    vault.mkdir(exist_ok=True)
    return load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx"), **extra})


def test_load_reranker_disabled_when_dir_is_empty_string(tmp_path):
    from rag_service.rerank import load_reranker

    assert load_reranker(_cfg(tmp_path, RAG_RERANK_DIR="")) is None


def test_load_reranker_default_location_without_model_is_simply_off(tmp_path, monkeypatch):
    import dataclasses

    from rag_service.rerank import load_reranker

    cfg = dataclasses.replace(_cfg(tmp_path), rerank_dir=tmp_path / "nothing-here")
    assert cfg.rerank_required is False
    assert load_reranker(cfg) is None


def test_load_reranker_explicit_missing_dir_is_an_error(tmp_path):
    from rag_service.rerank import load_reranker

    with pytest.raises(FileNotFoundError, match="reranker"):
        load_reranker(_cfg(tmp_path, RAG_RERANK_DIR=str(tmp_path / "missing")))
