from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from rag_service.api import create_app
from rag_service.lock import index_lock
from test_service import _make


def _client(tmp_path, *, api_key=None, **kw):
    svc, cfg, vault, clock = _make(tmp_path, **kw)
    cfg = replace(cfg, api_key=api_key)
    return TestClient(create_app(svc, cfg)), svc, cfg, vault


def test_search(tmp_path):
    c, *_ = _client(tmp_path)
    r = c.post("/search", json={"query": "kedi süt", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["path"] == "kedi.md"
    assert "süt" in body["results"][0]["text"]


def test_search_default_k_and_validation(tmp_path):
    c, *_ = _client(tmp_path)
    assert c.post("/search", json={"query": "kedi"}).status_code == 200
    assert c.post("/search", json={}).status_code == 422
    assert c.post("/search", json={"query": "kedi", "k": 0}).status_code == 422
    assert c.post("/search", json={"query": "kedi", "k": 99}).status_code == 422


def test_search_without_index_is_503(tmp_path):
    c, *_ = _client(tmp_path, build=False)
    r = c.post("/search", json={"query": "kedi"})
    assert r.status_code == 503
    assert "reindex" in r.json()["detail"]


def test_note(tmp_path):
    c, *_ = _client(tmp_path)
    r = c.get("/note", params={"path": "kedi.md"})
    assert r.status_code == 200
    assert r.json()["path"] == "kedi.md" and "kedi süt" in r.json()["text"]


def test_note_errors(tmp_path):
    c, *_ = _client(tmp_path)
    assert c.get("/note", params={"path": "../x.md"}).status_code == 404
    assert c.get("/note", params={"path": "yok.md"}).status_code == 404
    assert c.get("/note").status_code == 422


def test_health(tmp_path):
    c, *_ = _client(tmp_path)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["index_present"] is True and r.json()["loaded"] is False


def test_reindex(tmp_path):
    c, _, _, vault = _client(tmp_path)
    (vault / "yeni.md").write_text("# Yeni\n\nbalık tutmak güzel\n", encoding="utf-8")
    r = c.post("/reindex")
    assert r.status_code == 200 and r.json()["changed"] == 1
    assert c.post("/search", json={"query": "balık tutmak"}).json()["results"][0]["path"] == "yeni.md"


def test_reindex_busy_is_409(tmp_path):
    c, _, cfg, _ = _client(tmp_path)
    with index_lock(cfg):
        assert c.post("/reindex").status_code == 409


def test_no_key_configured_means_open(tmp_path):
    c, *_ = _client(tmp_path, api_key=None)
    assert c.get("/health").status_code == 200


@pytest.mark.parametrize("headers,ok", [
    ({}, False),
    ({"X-API-Key": "wrong"}, False),
    ({"X-API-Key": "s3cret"}, True),
    ({"Authorization": "Bearer s3cret"}, True),
    ({"Authorization": "Bearer wrong"}, False),
    ({"Authorization": "s3cret"}, False),
])
def test_api_key(tmp_path, headers, ok):
    c, *_ = _client(tmp_path, api_key="s3cret")
    for call in (lambda: c.get("/health", headers=headers),
                 lambda: c.post("/search", json={"query": "kedi"}, headers=headers),
                 lambda: c.get("/note", params={"path": "kedi.md"}, headers=headers),
                 lambda: c.post("/reindex", headers=headers)):
        assert (call().status_code != 401) is ok
