"""Tests for the ingest pipeline and Corpus."""

import tempfile

import sunder


def _sample_docs():
    return [
        (
            "Kubernetes is a container orchestration platform that automates "
            "the deployment and scaling of containerized applications.\n\n"
            "The control plane includes the API server, scheduler, and etcd. "
            "Worker nodes run kubelet and the container runtime.\n\n"
            "Pods are the smallest deployable units. Each pod contains one or "
            "more containers with shared networking."
        ),
        (
            "PostgreSQL is an advanced relational database system. It supports "
            "ACID transactions, JSON columns, and full-text search.\n\n"
            "Connection pooling with PgBouncer reduces overhead for "
            "high-concurrency workloads.\n\n"
            "The VACUUM process reclaims storage from deleted rows. "
            "Autovacuum runs automatically in the background."
        ),
    ]


def test_ingest_raw_text():
    cfg = sunder.SunderConfig(chunking_mode="flat", chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        assert corpus.index.ntotal > 0
        assert len(corpus.chunks) > 0


def test_search_returns_results():
    cfg = sunder.SunderConfig(chunking_mode="flat", chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        hits = corpus.search("container orchestration", top_k=3)
        assert len(hits) > 0
        assert hits[0].score > 0


def test_search_unique_chunk_ids():
    cfg = sunder.SunderConfig(chunking_mode="flat", chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        hits = corpus.search("database", top_k=5)
        chunk_ids = [h.chunk_id for h in hits]
        assert len(chunk_ids) == len(set(chunk_ids))


def test_search_scores_sorted():
    cfg = sunder.SunderConfig(chunking_mode="flat", chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        hits = corpus.search("query", top_k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


def test_search_snippets():
    cfg = sunder.SunderConfig(chunking_mode="flat", chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        hits = corpus.search("query", top_k=3)
        for h in hits:
            assert 1 <= len(h.snippets) <= 3
            assert all(isinstance(s, str) and len(s) > 0 for s in h.snippets)


def test_get_chunk():
    cfg = sunder.SunderConfig(chunking_mode="flat", chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        hits = corpus.search("query", top_k=1)
        chunk = corpus.get_chunk(hits[0].chunk_id)
        assert chunk is not None
        assert "text" in chunk
        assert "doc_id" in chunk
        assert len(chunk["text"]) > 0


def test_get_chunk_unknown():
    cfg = sunder.SunderConfig(chunking_mode="flat", chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        assert corpus.get_chunk("nonexistent") is None


def test_corpus_load():
    cfg = sunder.SunderConfig(chunking_mode="flat", chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        total = corpus.index.ntotal

        loaded = sunder.Corpus.load(tmpdir)
        assert loaded.index.ntotal == total

        hits = loaded.search("container", top_k=2)
        assert len(hits) > 0
        assert all(len(h.snippets) > 0 for h in hits)

        chunk = loaded.get_chunk(hits[0].chunk_id)
        assert chunk is not None


def test_resume_no_duplicates():
    cfg = sunder.SunderConfig(chunking_mode="flat", chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus1 = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        n = corpus1.index.ntotal

        corpus2 = sunder.ingest(_sample_docs(), tmpdir, config=cfg)
        assert corpus2.index.ntotal == n
