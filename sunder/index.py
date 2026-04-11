"""
sunder.index -- Sentence-level embedding index and search.

Embeds sentences (not chunks) into a FAISS IndexFlatL2 index.
Search returns ranked hits grouped by parent chunk.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .types import Chunk, SearchHit, SunderConfig, Triple

# Suppress noisy model load reports
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

# File names within the corpus output directory
_INDEX_FILE = "faiss.index"
_METADATA_FILE = "metadata.pkl"
_CONFIG_FILE = "config.json"
_PROGRESS_FILE = "progress.json"
_TRIPLES_FILE = "triples.jsonl"
_CHUNKS_FILE = "chunks.jsonl"


def _load_model(model_name: str):
    """Lazy-load the sentence-transformers model."""
    import os
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _embed_batch(
    texts: list[str],
    config: SunderConfig,
    st_model=None,
) -> np.ndarray:
    """Embed a batch of texts using the configured provider.

    For "sentence-transformers", st_model must be a pre-loaded model to avoid
    reloading per batch. For "openai", st_model is ignored.
    """
    if config.embedding_provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=config.openai_api_key)
        response = client.embeddings.create(
            model=config.embedding_model,
            input=texts,
        )
        return np.array([e.embedding for e in response.data], dtype=np.float32)
    else:
        vectors = st_model.encode(texts, normalize_embeddings=False)
        return np.array(vectors, dtype=np.float32)


def _load_progress(output_dir: Path) -> set[str]:
    """Load the set of already-embedded sentence_ids from progress checkpoint."""
    progress_path = output_dir / _PROGRESS_FILE
    if progress_path.exists():
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        return set(data.keys())
    return set()


def _save_progress(output_dir: Path, embedded_ids: set[str]):
    """Save progress checkpoint."""
    progress_path = output_dir / _PROGRESS_FILE
    progress_path.write_text(
        json.dumps({sid: True for sid in sorted(embedded_ids)}),
        encoding="utf-8",
    )


def _load_config(output_dir: Path) -> SunderConfig | None:
    """Load saved config from output_dir, or None if not found."""
    config_path = output_dir / _CONFIG_FILE
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return SunderConfig(**data)
    return None


def _save_config(output_dir: Path, config: SunderConfig):
    """Save config alongside the index."""
    config_path = output_dir / _CONFIG_FILE
    config_path.write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )


def _load_triples(output_dir: Path) -> list[Triple]:
    """Load triples from JSONL file, or empty list if not found."""
    triples_path = output_dir / _TRIPLES_FILE
    if not triples_path.exists():
        return []
    triples = []
    for line in triples_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        triples.append(Triple(
            subject=d["subject"],
            subject_type=d["subject_type"],
            relation=d["relation"],
            object=d["object"],
            object_type=d["object_type"],
            triple_id=d["triple_id"],
            doc_id=d["doc_id"],
            chunk_id=d["chunk_id"],
            chunk_index=d.get("chunk_index"),
            sentence_id=d.get("sentence_id"),
        ))
    return triples


def _save_triples(output_dir: Path, triples: list[Triple]):
    """Save triples as JSONL -- one JSON object per line."""
    triples_path = output_dir / _TRIPLES_FILE
    lines = []
    for t in triples:
        lines.append(json.dumps({
            "subject": t.subject,
            "subject_type": t.subject_type,
            "relation": t.relation,
            "object": t.object,
            "object_type": t.object_type,
            "triple_id": t.triple_id,
            "doc_id": t.doc_id,
            "chunk_id": t.chunk_id,
            "chunk_index": t.chunk_index,
            "sentence_id": t.sentence_id,
        }, ensure_ascii=False))
    triples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_chunks_jsonl(output_dir: Path) -> dict[str, dict]:
    """Load chunk data from JSONL into a dict keyed by chunk_id."""
    chunks_path = output_dir / _CHUNKS_FILE
    if not chunks_path.exists():
        return {}
    lookup = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        lookup[d["chunk_id"]] = d
    return lookup


def _save_chunks_jsonl(output_dir: Path, chunks: list[Chunk]):
    """Save chunk text and metadata as JSONL."""
    chunks_path = output_dir / _CHUNKS_FILE
    lines = []
    for c in chunks:
        lines.append(json.dumps({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "token_count": c.token_count,
            "chunk_index": c.chunk_index,
            "page": c.page,
            "topic_title": c.topic_title,
            "topic_summary": c.topic_summary,
        }, ensure_ascii=False))
    chunks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def index_embeddings(
    chunks: list[Chunk],
    output_dir: str,
    config: SunderConfig | None = None,
) -> "Corpus":
    """Embed all sentences and build a FAISS index.

    Pre-flight checks:
      - All chunks must have sentences populated
      - On resume: embedding_model and embedding_dim must match saved config

    Persistence (four files in output_dir):
      - faiss.index      -- FAISS IndexFlatL2
      - metadata.pkl     -- list[dict] parallel to index vectors (includes text)
      - config.json      -- serialized SunderConfig
      - progress.json    -- {sentence_id: true} checkpoint

    Args:
        chunks: Chunks with sentences populated (call segment_sentences first).
        output_dir: Directory for index files.
        config: Pipeline config. Uses SunderConfig() defaults if None.

    Returns:
        A Corpus object ready for search.
    """
    cfg = config or SunderConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- Pre-flight: sentences must exist --
    unsegmented = [c.chunk_id for c in chunks if len(c.sentences) == 0]
    if unsegmented:
        raise ValueError(
            f"All chunks must have sentences populated. "
            f"{len(unsegmented)} chunk(s) have no sentences "
            f"(first: {unsegmented[0]}). "
            f"Call segment_sentences(chunks) before index_embeddings()."
        )

    # -- Pre-flight: config compatibility on resume --
    saved_config = _load_config(out)
    if saved_config is not None:
        if saved_config.embedding_model != cfg.embedding_model:
            raise ValueError(
                f"Cannot resume: index built with {saved_config.embedding_model!r}, "
                f"current config uses {cfg.embedding_model!r}. "
                f"Delete the existing index or use a matching model."
            )
        if saved_config.embedding_dim != cfg.embedding_dim:
            raise ValueError(
                f"Cannot resume: index built with embedding_dim={saved_config.embedding_dim}, "
                f"current config uses embedding_dim={cfg.embedding_dim}. "
                f"Delete the existing index or use a matching dimension."
            )

    # -- Load existing state or create fresh --
    index_path = out / _INDEX_FILE
    meta_path = out / _METADATA_FILE

    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(cfg.embedding_dim)
        metadata = []

    already_embedded = _load_progress(out)

    # -- Collect sentences to embed --
    to_embed = []
    to_embed_meta = []
    for chunk in chunks:
        for sent in chunk.sentences:
            if sent.sentence_id in already_embedded:
                continue
            to_embed.append(sent.text)
            to_embed_meta.append({
                "sentence_id": sent.sentence_id,
                "chunk_id": sent.chunk_id,
                "doc_id": sent.doc_id,
                "sent_index_in_chunk": sent.sent_index_in_chunk,
                "text": sent.text,
            })

    if to_embed:
        # -- Embed in batches --
        st_model = _load_model(cfg.embedding_model) if cfg.embedding_provider == "sentence-transformers" else None
        batch_size = cfg.embedding_batch_size
        checkpoint_interval = 10  # save every 10 batches

        for batch_start in range(0, len(to_embed), batch_size):
            batch_end = min(batch_start + batch_size, len(to_embed))
            batch_texts = to_embed[batch_start:batch_end]
            batch_meta = to_embed_meta[batch_start:batch_end]

            vectors = _embed_batch(batch_texts, cfg, st_model=st_model)

            index.add(vectors)
            metadata.extend(batch_meta)

            for m in batch_meta:
                already_embedded.add(m["sentence_id"])

            # Checkpoint every N batches
            batch_num = batch_start // batch_size
            if (batch_num + 1) % checkpoint_interval == 0:
                faiss.write_index(index, str(index_path))
                with open(meta_path, "wb") as f:
                    pickle.dump(metadata, f)
                _save_progress(out, already_embedded)

        # -- Final save --
        faiss.write_index(index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        _save_progress(out, already_embedded)

    # -- Save config and chunks (always, so they exist even on first run) --
    _save_config(out, cfg)
    _save_chunks_jsonl(out, chunks)

    return Corpus(
        output_dir=str(out),
        chunks=chunks,
        config=cfg,
        _index=index,
        _metadata=metadata,
    )


class Corpus:
    """A searchable corpus of sentence embeddings backed by FAISS.

    Created by index_embeddings() or loaded from disk via Corpus.load().
    Fully self-contained: snippets, chunk text, and topic metadata are all
    persisted — no in-memory chunks required after loading.
    """

    def __init__(
        self,
        output_dir: str,
        chunks: list[Chunk],
        config: SunderConfig,
        _index: faiss.IndexFlatL2 | None = None,
        _metadata: list[dict] | None = None,
        _triples: list[Triple] | None = None,
    ):
        self.output_dir = output_dir
        self.chunks = chunks
        self.config = config

        out = Path(output_dir)
        if _index is not None:
            self.index = _index
            self._metadata = _metadata or []
        else:
            self.index = faiss.read_index(str(out / _INDEX_FILE))
            with open(out / _METADATA_FILE, "rb") as f:
                self._metadata = pickle.load(f)

        # Triples: provided directly, loaded from disk, or empty
        if _triples is not None:
            self.triples = _triples
        else:
            self.triples = _load_triples(out)

        # Chunk data: persisted in chunks.jsonl, keyed by chunk_id for O(1) lookup
        self._chunks_data: dict[str, dict] = _load_chunks_jsonl(out)

        # Embedding model: loaded lazily on first search, then cached
        self._model = None

    @classmethod
    def load(cls, output_dir: str, chunks: list[Chunk] | None = None) -> "Corpus":
        """Load a previously built Corpus from disk.

        Fully functional: snippets from metadata, chunk text from chunks.jsonl,
        topic metadata from chunks.jsonl. No in-memory chunks required.

        Args:
            output_dir: Directory containing index files.
            chunks: Optional — ignored (kept for backward compatibility).
        """
        out = Path(output_dir)
        config = _load_config(out)
        if config is None:
            raise FileNotFoundError(
                f"No config.json found in {output_dir}. "
                f"Is this a valid sunder corpus directory?"
            )
        return cls(
            output_dir=output_dir,
            chunks=chunks or [],
            config=config,
        )

    @property
    def stats(self) -> dict:
        """Summary statistics for the corpus."""
        doc_ids = set()
        topic_titles = set()
        for cd in self._chunks_data.values():
            doc_ids.add(cd.get("doc_id", ""))
            t = cd.get("topic_title")
            if t:
                topic_titles.add(t)
        return {
            "documents": len(doc_ids),
            "chunks": len(self._chunks_data),
            "sentences": len(self._metadata),
            "vectors": self.index.ntotal,
            "topics": len(topic_titles),
            "triples": len(self.triples),
        }

    @property
    def structure(self) -> list[dict]:
        """Topic structure extracted from chunks.

        Returns a list of unique topics with their chunk ranges,
        ordered by first appearance. Empty if flat chunking was used.
        """
        topics: dict[str, dict] = {}
        for cd in self._chunks_data.values():
            title = cd.get("topic_title")
            if not title:
                continue
            if title not in topics:
                topics[title] = {
                    "title": title,
                    "summary": cd.get("topic_summary"),
                    "chunk_ids": [],
                }
            topics[title]["chunk_ids"].append(cd["chunk_id"])
        return list(topics.values())

    def get_chunk(self, chunk_id: str) -> dict | None:
        """Retrieve full chunk text and metadata by chunk_id.

        Returns None if chunk_id not found. Works after Corpus.load()
        without in-memory chunks — reads from persisted chunks.jsonl.

        Returned dict keys: chunk_id, doc_id, text, token_count,
        chunk_index, page, topic_title, topic_summary.
        """
        return self._chunks_data.get(chunk_id)

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Search the corpus for sentences matching the query.

        1. Embed query using the same model as the corpus
        2. Search FAISS: fetch top_k * search_oversample raw sentence hits
        3. Group by chunk_id -- best score per chunk, up to 3 snippets
        4. Return top_k SearchHit objects sorted by score descending

        Scores are similarity: higher = more similar, computed as 1/(1+L2_dist).

        Args:
            query: Search query text.
            top_k: Number of chunk-level hits to return.
                   Defaults to config.default_top_k.
        """
        k = top_k or self.config.default_top_k
        oversample = self.config.search_oversample

        if self.index.ntotal == 0:
            return []

        # Embed query (ST model cached after first load; OpenAI has no local model)
        if self.config.embedding_provider == "sentence-transformers" and self._model is None:
            self._model = _load_model(self.config.embedding_model)
        query_vec = _embed_batch([query], self.config, st_model=self._model)

        # Search FAISS -- fetch more than needed, then group by chunk
        n_fetch = min(k * oversample, self.index.ntotal)
        distances, indices = self.index.search(query_vec, n_fetch)

        # Group results by chunk_id
        # chunk_id -> list of (similarity_score, sentence_text, sentence_id)
        chunk_hits: dict[str, list[tuple[float, str, str]]] = {}

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            chunk_id = meta["chunk_id"]
            similarity = 1.0 / (1.0 + float(dist))
            sent_text = meta.get("text", meta["sentence_id"])

            if chunk_id not in chunk_hits:
                chunk_hits[chunk_id] = []
            chunk_hits[chunk_id].append((similarity, sent_text, meta["sentence_id"]))

        # Build SearchHit for each chunk -- best score, up to 3 snippets
        hits: list[SearchHit] = []
        for chunk_id, sentence_results in chunk_hits.items():
            sentence_results.sort(key=lambda x: x[0], reverse=True)
            best_score = sentence_results[0][0]
            top_snippets = sentence_results[:3]

            chunk_data = self._chunks_data.get(chunk_id, {})

            hits.append(SearchHit(
                chunk_id=chunk_id,
                doc_id=chunk_data.get("doc_id", ""),
                score=best_score,
                snippets=[s[1] for s in top_snippets],
                sentence_ids=[s[2] for s in top_snippets],
                topic_title=chunk_data.get("topic_title"),
                topic_summary=chunk_data.get("topic_summary"),
            ))

        # Sort by score descending, take top_k
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
