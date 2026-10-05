import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from rag_service.embeddings import Embedder

_WORDS = ["<unk>", "<pad>", "passage", "query", ":", "a", "b", "c", "d", "e", "f"]
_VOCAB = {w: i for i, w in enumerate(_WORDS)}
_DIM = 8


def _tokenizer():
    tok = Tokenizer(models.WordLevel(_VOCAB, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


class _Input:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Stands in for onnxruntime: hidden state = fixed lookup table of the token ids."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.table = rng.normal(size=(len(_VOCAB), _DIM)).astype(np.float32)
        self.calls = []

    def get_inputs(self):
        return [_Input("input_ids"), _Input("attention_mask")]

    def run(self, _outputs, feeds):
        self.calls.append(feeds)
        return [self.table[feeds["input_ids"]]]


def _embedder(**kw):
    sess = FakeSession()
    return Embedder(sess, _tokenizer(), **kw), sess


def test_output_shape_dtype_and_unit_norm():
    emb, _ = _embedder()
    out = emb.embed_passages(["a b c", "d e"])
    assert out.shape == (2, _DIM)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_empty_input_returns_empty_array():
    emb, _ = _embedder()
    assert emb.embed_passages([]).shape[0] == 0


def test_passage_and_query_use_different_prefixes():
    emb, sess = _embedder()
    emb.embed_passages(["a"])
    emb.embed_query("a")
    passage_ids = sess.calls[0]["input_ids"][0]
    query_ids = sess.calls[1]["input_ids"][0]
    assert passage_ids[0] == _VOCAB["passage"]
    assert query_ids[0] == _VOCAB["query"]


def test_embed_query_returns_single_vector():
    emb, _ = _embedder()
    v = emb.embed_query("a b")
    assert v.shape == (_DIM,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_padding_does_not_change_a_texts_embedding():
    emb, _ = _embedder()
    alone = emb.embed_passages(["a b"])[0]
    batched = emb.embed_passages(["a b", "a b c d e f a b c d e f"])[0]
    assert np.allclose(alone, batched, atol=1e-6)


def test_order_preserved_across_length_sorted_batches():
    emb, _ = _embedder(batch_size=2)
    texts = ["a b c d e f", "a", "b c", "d e f a b c d", "f"]
    together = emb.embed_passages(texts)
    for i, t in enumerate(texts):
        assert np.allclose(together[i], emb.embed_passages([t])[0], atol=1e-6)


def test_long_text_truncated_to_max_length():
    emb, sess = _embedder(max_length=6)
    emb.embed_passages(["a b c d e f a b c d e f"])
    assert sess.calls[0]["input_ids"].shape[1] == 6


def test_token_lengths_are_untruncated_and_include_prefix():
    emb, _ = _embedder(max_length=4)
    # "passage" ":" + 6 words = 8 tokens, even though max_length is 4
    assert emb.token_lengths(["a b c d e f"]) == [8]


def test_rejects_session_without_expected_inputs():
    class Bad(FakeSession):
        def get_inputs(self):
            return [_Input("something_else")]

    with pytest.raises(ValueError, match="input_ids"):
        Embedder(Bad(), _tokenizer())
