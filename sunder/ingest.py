"""
sunder.ingest -- Single entry point for the full document-to-searchable-corpus pipeline.

    corpus = sunder.ingest(docs=["doc.pdf", "raw text..."], output_dir="./corpus")
    results = corpus.search("query", top_k=10)
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .chunking import chunk_by_topics, chunk_by_topics_sentence_aware, chunk_text
from .extract import extract_pdf
from .index import Corpus, index_embeddings, _save_triples
from .kg import LLMExtractor, normalize_triples, deduplicate_triples
from .llm import LLMProvider, get_provider
from .sentences import segment_sentences
from .structure import detect_structure
from .types import AgentConfig, Chunk, Ontology, SunderConfig


def _doc_id_from_path(path: str) -> str:
    """Derive a doc_id from a file path: stem without extension."""
    return Path(path).stem


def _doc_id_from_text(text: str, index: int) -> str:
    """Derive a doc_id from raw text: short hash for stability."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"text_{index:03d}_{h}"


def ingest(
    docs: list[str],
    output_dir: str,
    config: SunderConfig | None = None,
    agent_config: AgentConfig | None = None,
    provider: LLMProvider | None = None,
    extract_graph: bool = False,
    pages: str | None = None,
    on_progress: callable | None = None,
    curation_report_path: str | None = None,
) -> Corpus | None:
    """Ingest documents into a searchable corpus.

    Full pipeline: extract -> chunk -> segment sentences -> embed -> index.
    Optionally: -> extract knowledge graph triples.

    Args:
        docs: List of document sources. Each entry is either:
              - A file path ending in .pdf (extracted via PyPDF)
              - A raw text string (used directly, doc_id derived from hash)
        output_dir: Directory for corpus files (faiss.index, metadata, config).
        config: Pipeline configuration. Uses SunderConfig() defaults if None.
        agent_config: Runtime config for LLM provider selection. Used to create
                      a provider if none is given explicitly.
        provider: Pre-initialized LLMProvider. Takes precedence over agent_config.
        extract_graph: If True, run KG extraction on chunks via LLM.
        pages: Optional page range for PDF extraction (e.g. "1-50"). Applied to all PDFs.
        on_progress: Optional callback(stage: str, detail: str) for progress reporting.

    Returns:
        A Corpus object ready for search. If extract_graph=True,
        corpus.triples is populated and triples.jsonl is persisted.
    """
    cfg = config or SunderConfig()

    # -- Enforce curation approval gate --
    # If curation is enabled, a report path is mandatory. Curation and
    # embedding must be separated by a human review step: first run writes
    # the report and stops; second run loads the approved report and embeds.
    if cfg.curate and not curation_report_path:
        raise ValueError(
            "curate=True requires curation_report_path. "
            "Run curation first to generate the report, review it, then run embed."
        )

    # Create LLM provider lazily: only needed for topic chunking or graph extraction
    _provider: LLMProvider | None = provider

    def _get_provider() -> LLMProvider:
        nonlocal _provider
        if _provider is None:
            _provider = get_provider(agent_config)
        return _provider

    def _report(stage: str, detail: str):
        if on_progress:
            on_progress(stage, detail)

    # -- Stage 1: Extract text from each document --
    all_chunks: list[Chunk] = []
    t0 = time.perf_counter()

    for i, doc in enumerate(docs):
        is_pdf = doc.lower().endswith(".pdf")

        if is_pdf:
            doc_id = _doc_id_from_path(doc)
            _report("extract", f"[{i+1}/{len(docs)}] Extracting {Path(doc).name}...")
            text = extract_pdf(doc, pages=pages)
        else:
            doc_id = _doc_id_from_text(doc, i)
            text = doc
            _report("extract", f"[{i+1}/{len(docs)}] Processing raw text ({len(text):,} chars)...")

        if not text.strip():
            _report("extract", f"  Skipped {doc_id}: empty text")
            continue

        # -- Stage 1b (optional): LLM-assisted curation --
        # Strict two-step gate: first invocation writes a report and stops
        # (no embedding). Second invocation loads the approved report and
        # proceeds. curate_text() and apply_curation() are never chained
        # internally -- the human review step between them is mandatory.
        if cfg.curate:
            from .curator import apply_curation, curate_text, load_report, save_report

            report_path = Path(curation_report_path)  # type: ignore[arg-type]
            if not report_path.exists():
                _report("curate", f"  Curating {doc_id} with Claude...")
                report = curate_text(text, doc_id, cfg)
                save_report(report, report_path)
                print(
                    f"\nCuration complete. Review {report_path} then run embed."
                )
                return None

            _report("curate", f"  Loading curation report from {report_path}")
            report = load_report(report_path)
            text = "\n\n".join(apply_curation(report))

            if not text.strip():
                _report("curate", f"  Skipped {doc_id}: empty after curation")
                continue

        # -- Stage 2: Chunk --
        # "topic_sentence" mode builds sentences during chunking (no separate step).
        # "topic" and "flat" modes require a separate segment_sentences() pass.
        sentences_done = False

        if cfg.chunking_mode == "topic_sentence":
            _report("chunk", f"  Topic segmentation + sentence-aware chunking {doc_id}...")
            structure = detect_structure(text, method="topics", doc_title=doc_id, provider=_get_provider())
            _report("chunk", f"  {len(structure.sections)} topics detected")
            chunks = chunk_by_topics_sentence_aware(text, doc_id, structure, config=cfg)
            sentences_done = True
        elif cfg.chunking_mode == "topic":
            _report("chunk", f"  Topic segmentation + chunking {doc_id}...")
            structure = detect_structure(text, method="topics", doc_title=doc_id, provider=_get_provider())
            _report("chunk", f"  {len(structure.sections)} topics detected")
            chunks = chunk_by_topics(text, doc_id, structure, config=cfg)
        else:
            _report("chunk", f"  Chunking {doc_id} (flat)...")
            chunks = chunk_text(text, doc_id, config=cfg)

        if not chunks:
            _report("chunk", f"  Skipped {doc_id}: no chunks after filtering")
            continue

        # -- Stage 3: Sentence segmentation (skipped for topic_sentence) --
        if not sentences_done:
            _report("segment", f"  Segmenting {doc_id}: {len(chunks)} chunks...")
            segment_sentences(chunks, config=cfg)

        sent_count = sum(len(c.sentences) for c in chunks)
        _report("segment", f"  {doc_id}: {len(chunks)} chunks, {sent_count} sentences")

        all_chunks.extend(chunks)

    extract_elapsed = time.perf_counter() - t0
    total_sents = sum(len(c.sentences) for c in all_chunks)
    _report("extract", f"Extraction complete: {len(all_chunks)} chunks, {total_sents} sentences in {extract_elapsed:.1f}s")

    if not all_chunks:
        raise ValueError("No chunks produced from any document. Check input documents and config.min_chunk_size.")

    # -- Stage 4: Embed and index --
    _report("index", f"Indexing {total_sents} sentences...")
    t1 = time.perf_counter()
    corpus = index_embeddings(all_chunks, output_dir, config=cfg)
    index_elapsed = time.perf_counter() - t1
    _report("index", f"Indexed in {index_elapsed:.1f}s ({corpus.index.ntotal} vectors)")

    # -- Stage 5 (optional): Graph extraction --
    if extract_graph:
        graph_chunks = all_chunks
        if cfg.graph_max_chunks is not None:
            graph_chunks = all_chunks[:cfg.graph_max_chunks]
        _report("graph", f"Extracting knowledge graph from {len(graph_chunks)} chunks...")
        t2 = time.perf_counter()

        ontology = Ontology()
        extractor = LLMExtractor(_get_provider())
        result = extractor.extract_from_chunks(graph_chunks, ontology)

        result.triples = normalize_triples(result.triples)
        result.triples = deduplicate_triples(result.triples)

        graph_elapsed = time.perf_counter() - t2
        _report("graph",
            f"  {len(result.triples)} triples in {graph_elapsed:.1f}s "
            f"(${result.cost_usd:.4f}, {result.api_calls} API calls)")

        # Persist and attach to corpus
        out = Path(output_dir)
        _save_triples(out, result.triples)
        corpus.triples = result.triples

    return corpus
