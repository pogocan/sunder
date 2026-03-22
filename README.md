# sunder

PDF to searchable knowledge in three lines.

## Install

```bash
uv add git+https://github.com/pogocan/sunder.git
```

For knowledge graph extraction (requires Anthropic API key):

```bash
uv add "sunderdoc[graph] @ git+https://github.com/pogocan/sunder.git"
```

## Usage

```python
import sunder

# Ingest PDFs or raw text into a searchable corpus
corpus = sunder.ingest(["doc.pdf"], output_dir="./corpus")

# Search returns ranked hits with sentence snippets
hits = corpus.search("your query", top_k=5)

# Retrieve full chunk text
chunk = corpus.get_chunk(hits[0].chunk_id)
```

## LLM requirement

Topic-guided chunking (default) uses an LLM to model document structure
before chunking. Set your API key:

```bash
export ANTHROPIC_API_KEY=your_key
```

To run without an LLM, use flat chunking:

```python
config = sunder.SunderConfig(chunking_mode="flat")
corpus = sunder.ingest(["doc.pdf"], output_dir="./corpus", config=config)
```

Flat mode uses paragraph-boundary chunking only -- no API calls,
no topic metadata on chunks.

## What it does

- Extracts and cleans text from PDFs (dehyphenation, whitespace normalization)
- Chunks by paragraph boundaries (never splits mid-sentence)
- Segments into sentences, embeds with sentence-transformers
- Builds a FAISS index for fast semantic search
- Returns sentence-level snippets grouped by parent chunk
- Optionally extracts a knowledge graph via Anthropic API

## Three-tier retrieval

Sunder uses a three-tier model:

- **Topic** -- structural metadata (which section this belongs to)
- **Chunk** -- readable context unit (what a caller opens and reads)
- **Sentence** -- retrieval unit (what gets embedded and searched)

Search finds relevant sentences, then groups results by parent chunk. The caller decides which chunks to fully read.

## Configuration

```python
config = sunder.SunderConfig(
    chunk_size=200,                # words per chunk
    chunk_overlap=30,              # overlap between chunks
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    default_top_k=10,
)
corpus = sunder.ingest(["doc.pdf"], output_dir="./corpus", config=config)
```

## Knowledge graph extraction

```python
# Requires: uv add "sunderdoc[graph] @ git+https://github.com/pogocan/sunder.git"
# Requires: ANTHROPIC_API_KEY environment variable
corpus = sunder.ingest(
    ["doc.pdf"],
    output_dir="./corpus",
    extract_graph=True,
)
print(f"{len(corpus.triples)} triples extracted")
```

## Raw text input

Works without PDFs -- pass text strings directly:

```python
corpus = sunder.ingest(
    ["This is raw text content to index..."],
    output_dir="./corpus",
)
```

## Persistence

A corpus is fully self-contained on disk. Load and search without re-indexing:

```python
corpus = sunder.Corpus.load("./corpus")
hits = corpus.search("query")
chunk = corpus.get_chunk(hits[0].chunk_id)
```

## License

MIT
